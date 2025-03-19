#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coordenador de Treinamento para o sistema distribuído.
Gerencia o processo de treinamento distribuído e coordena os nós.
"""

import os
import sys
import time
import json
import logging
import threading
import queue
import math
from typing import Dict, List, Optional, Union
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta

# Importações condicionais
try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TrainingStatus(Enum):
    """Status possíveis do treinamento."""
    IDLE = 0
    PREPARING = 1
    TRAINING = 2
    EVALUATING = 3
    COMPLETED = 4
    FAILED = 5
    PAUSED = 6

class TaskType(Enum):
    """Tipos de tarefas que podem ser atribuídas aos nós."""
    DOWNLOAD_DATASET = 0
    PREPROCESS_DATA = 1
    TRAIN_SHARD = 2
    EVALUATE = 3
    MERGE_MODEL = 4
    UPLOAD_MODEL = 5
    CLEANUP = 6

class TrainingTask:
    """Representa uma tarefa de treinamento."""
    
    def __init__(self, task_id, task_type, params=None, node_id=None):
        self.id = task_id
        self.type = task_type
        self.params = params or {}
        self.node_id = node_id
        self.status = TrainingStatus.IDLE
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.result = None
        self.error = None
    
    def to_dict(self):
        """Converte o objeto para dicionário."""
        return {
            "id": self.id,
            "type": self.type.name,
            "params": self.params,
            "node_id": self.node_id,
            "status": self.status.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "error": self.error
        }
    
    def start(self, node_id=None):
        """Marca a tarefa como iniciada."""
        self.status = TrainingStatus.TRAINING
        self.started_at = datetime.now()
        if node_id:
            self.node_id = node_id
    
    def complete(self, result=None):
        """Marca a tarefa como concluída."""
        self.status = TrainingStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 100.0
        self.result = result
    
    def fail(self, error):
        """Marca a tarefa como falha."""
        self.status = TrainingStatus.FAILED
        self.completed_at = datetime.now()
        self.error = str(error)
    
    def update_progress(self, progress):
        """Atualiza o progresso da tarefa."""
        self.progress = float(progress)
    
    @property
    def duration(self):
        """Retorna a duração da tarefa."""
        if not self.started_at:
            return timedelta(0)
        
        end_time = self.completed_at or datetime.now()
        return end_time - self.started_at

class TrainingCoordinator:
    """Coordenador central do treinamento distribuído."""
    
    def __init__(self, config, node_manager, dataset_handler):
        self.config = config
        self.node_manager = node_manager
        self.dataset_handler = dataset_handler
        self.logger = logging.getLogger(__name__)
        
        # Status e controle do treinamento
        self.status = TrainingStatus.IDLE
        self.start_time = None
        self.end_time = None
        self.last_checkpoint = None
        
        # Filas de tarefas
        self.task_queue = queue.Queue()
        self.completed_tasks = []
        self.failed_tasks = []
        self.active_tasks = {}
        
        # Controle de threads
        self.task_lock = threading.Lock()
        self.running = False
        self.dispatcher_thread = None
        self.progress_thread = None
        
        # Dados do modelo
        self.model_config = config.get("training", {}).get("model", {})
        self.model_dir = Path(config.get("infrastructure", {}).get("master_storage_path", "./storage/model"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Registra todas as máquinas disponíveis
        self.available_nodes = []
        
        # Taxa de progresso
        self.total_work_units = 0
        self.completed_work_units = 0
    
    def start(self):
        """Inicia o coordenador de treinamento."""
        self.logger.info("Iniciando coordenador de treinamento...")
        self.running = True
        self.start_time = datetime.now()
        self.status = TrainingStatus.PREPARING
        
        # Inicia os threads de controle
        self.dispatcher_thread = threading.Thread(target=self._task_dispatcher)
        self.dispatcher_thread.daemon = True
        self.dispatcher_thread.start()
        
        self.progress_thread = threading.Thread(target=self._progress_monitor)
        self.progress_thread.daemon = True
        self.progress_thread.start()
        
        # Inicializa o processo de treinamento
        try:
            self._prepare_training()
            self._start_training()
            
            # Aguarda a conclusão (bloqueante)
            self._wait_for_completion()
            
        except KeyboardInterrupt:
            self.logger.info("Treinamento interrompido pelo usuário.")
            self._cleanup(forced=True)
        except Exception as e:
            self.logger.error(f"Erro durante o treinamento: {str(e)}", exc_info=True)
            self.status = TrainingStatus.FAILED
            self._cleanup(forced=True)
    
    def stop(self):
        """Para o coordenador de treinamento."""
        self.logger.info("Parando coordenador de treinamento...")
        self.running = False
        
        # Cancela tarefas ativas
        with self.task_lock:
            for task_id, task in list(self.active_tasks.items()):
                self.logger.info(f"Cancelando tarefa {task_id}")
                # Aqui seria implementada a lógica para cancelar a tarefa no nó
                task.fail("Cancelado pelo coordenador")
                self.failed_tasks.append(task)
                del self.active_tasks[task_id]
        
        # Aguarda os threads terminarem
        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            self.dispatcher_thread.join(timeout=5)
        
        if self.progress_thread and self.progress_thread.is_alive():
            self.progress_thread.join(timeout=5)
        
        self.end_time = datetime.now()
        self._cleanup()
    
    def _prepare_training(self):
        """Prepara o ambiente para o treinamento."""
        self.logger.info("Preparando ambiente para treinamento...")
        
        # Registra todos os nós disponíveis
        self.available_nodes = self.node_manager.get_nodes(status="ONLINE")
        num_nodes = len(self.available_nodes)
        
        if num_nodes == 0:
            # Se não houver nós externos, usa apenas o nó local
            self.logger.warning("Nenhum nó remoto disponível. Usando apenas o nó local.")
            self.available_nodes = [self.node_manager.local_node]
            num_nodes = 1
        
        self.logger.info(f"Número de nós disponíveis para treinamento: {num_nodes}")
        
        # Baixa e prepara os datasets
        for dataset_name in self.config.get("data", {}).get("datasets", []):
            task_id = f"download_{dataset_name}_{int(time.time())}"
            task = TrainingTask(
                task_id=task_id,
                task_type=TaskType.DOWNLOAD_DATASET,
                params={"dataset_name": dataset_name}
            )
            self.task_queue.put(task)
        
        # Aguarda o download dos datasets (bloqueante)
        self._wait_for_empty_queue()
        
        # Verifica se houve falhas fatais no download
        if self._check_fatal_failures():
            raise Exception("Falhas críticas durante o download dos datasets")
        
        self.logger.info("Preparação concluída com sucesso")
    
    def _start_training(self):
        """Inicia o processo de treinamento."""
        self.logger.info("Iniciando processo de treinamento...")
        self.status = TrainingStatus.TRAINING
        
        # Preprocess datasets and create shards
        datasets = self.dataset_handler.load_all_datasets()
        
        # Filtra por linguagens configuradas
        for name, dataset in datasets.items():
            datasets[name] = self.dataset_handler.filter_by_languages(dataset)
        
        # Calcula o total de trabalho com base no tamanho dos datasets
        total_samples = 0
        for name, dataset in datasets.items():
            if hasattr(dataset, "num_rows"):
                total_samples += dataset.num_rows
            elif hasattr(dataset, "__len__"):
                total_samples += len(dataset)
        
        # Define unidades de trabalho (1 unidade = 1000 amostras)
        self.total_work_units = max(1, total_samples // 1000)
        self.logger.info(f"Total de unidades de trabalho: {self.total_work_units}")
        
        # Para cada dataset, cria shards e distribui para os nós
        for dataset_name in datasets.keys():
            # Cria shards (pedaços) do dataset para processamento distribuído
            shard_size = self.config.get("data", {}).get("processing", {}).get("shard_size", 1000)
            shards = self.dataset_handler.prepare_shards(dataset_name, shard_size=shard_size)
            
            if not shards:
                self.logger.warning(f"Não foi possível criar shards para {dataset_name}")
                continue
            
            # Distribui os shards entre os nós disponíveis
            num_nodes = len(self.available_nodes)
            shards_per_node = math.ceil(len(shards) / num_nodes)
            
            # Otimiza a distribuição com base nas capacidades dos nós
            node_distribution = self.node_manager.optimize_workload_distribution(
                self.available_nodes, len(shards)
            )
            
            # Cria tarefas de treinamento para cada nó
            node_idx = 0
            shard_idx = 0
            
            for node_id, num_shards in node_distribution.items():
                # Obtém shards para este nó
                node_shards = shards[shard_idx:shard_idx + num_shards]
                if not node_shards:
                    continue
                
                # Cria tarefa de treinamento
                task_id = f"train_{dataset_name}_{node_id}_{int(time.time())}"
                task = TrainingTask(
                    task_id=task_id,
                    task_type=TaskType.TRAIN_SHARD,
                    params={
                        "dataset_name": dataset_name,
                        "shards": [str(s) for s in node_shards],
                        "model_config": self.model_config,
                        "training_config": self.config.get("training", {})
                    },
                    node_id=node_id
                )
                
                # Adiciona à fila de tarefas
                self.task_queue.put(task)
                shard_idx += num_shards
                
            self.logger.info(f"Criadas tarefas de treinamento para {dataset_name} em {len(node_distribution)} nós")
        
        # Adiciona tarefa de merge final das partes do modelo
        task_id = f"merge_model_{int(time.time())}"
        task = TrainingTask(
            task_id=task_id,
            task_type=TaskType.MERGE_MODEL,
            params={
                "model_parts": [],  # Será preenchido durante o treinamento
                "model_config": self.model_config,
                "output_dir": str(self.model_dir)
            }
        )
        self.task_queue.put(task)
        
        self.logger.info("Tarefas de treinamento iniciadas com sucesso")
    
    def _task_dispatcher(self):
        """Thread que despacha tarefas para os nós."""
        self.logger.info("Despachador de tarefas iniciado")
        
        while self.running:
            try:
                # Obtém nós disponíveis
                available_nodes = self.node_manager.get_nodes(status="ONLINE")
                busy_nodes = set(task.node_id for task in self.active_tasks.values() if task.node_id)
                free_nodes = [node for node in available_nodes if node.id not in busy_nodes]
                
                if not free_nodes:
                    # Se não há nós livres, aguarda um pouco
                    time.sleep(5)
                    continue
                
                # Pega uma tarefa da fila
                try:
                    task = self.task_queue.get(block=False)
                except queue.Empty:
                    # Não há tarefas pendentes
                    time.sleep(5)
                    continue
                
                # Se a tarefa já tem um nó atribuído e ele não está disponível, reatribui
                if task.node_id and task.node_id not in [n.id for n in available_nodes]:
                    task.node_id = None
                
                # Atribui um nó para a tarefa se necessário
                if not task.node_id:
                    # Escolhe o nó menos carregado
                    chosen_node = min(free_nodes, key=lambda n: n.load_avg)
                    task.node_id = chosen_node.id
                
                # Adiciona à lista de tarefas ativas
                with self.task_lock:
                    self.active_tasks[task.id] = task
                
                # Inicia a tarefa (no futuro implementaremos comunicação com o nó)
                task.start(task.node_id)
                self.logger.info(f"Tarefa {task.id} iniciada no nó {task.node_id}")
                
                # Simula execução da tarefa (placeholder para implementação real)
                if task.type == TaskType.DOWNLOAD_DATASET:
                    threading.Thread(target=self._execute_download_task, args=(task,)).start()
                elif task.type == TaskType.TRAIN_SHARD:
                    threading.Thread(target=self._execute_training_task, args=(task,)).start()
                elif task.type == TaskType.MERGE_MODEL:
                    threading.Thread(target=self._execute_merge_task, args=(task,)).start()
                
                # Marca a tarefa como completa para a fila (não para execução)
                self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Erro no despachador de tarefas: {str(e)}", exc_info=True)
                time.sleep(10)  # Espera antes de tentar novamente após erro
    
    def _execute_download_task(self, task):
        """Executa uma tarefa de download em background."""
        try:
            dataset_name = task.params.get("dataset_name")
            self.logger.info(f"Executando download de {dataset_name}")
            
            # Executa o download usando o dataset handler
            success = self.dataset_handler._download_dataset(dataset_name)
            
            if success:
                # Marca a tarefa como concluída
                task.complete({"status": "success", "dataset": dataset_name})
                self.logger.info(f"Download de {dataset_name} concluído com sucesso")
            else:
                # Marca a tarefa como falha
                task.fail(f"Falha ao baixar dataset {dataset_name}")
                self.logger.error(f"Falha ao baixar dataset {dataset_name}")
        
        except Exception as e:
            # Em caso de erro, marca a tarefa como falha
            task.fail(str(e))
            self.logger.error(f"Erro em tarefa de download: {str(e)}", exc_info=True)
        
        finally:
            # Remove da lista de tarefas ativas e adiciona à lista adequada
            with self.task_lock:
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                
                if task.status == TrainingStatus.COMPLETED:
                    self.completed_tasks.append(task)
                else:
                    self.failed_tasks.append(task)
    
    def _execute_training_task(self, task):
        """Executa uma tarefa de treinamento em background."""
        try:
            dataset_name = task.params.get("dataset_name")
            shards = task.params.get("shards", [])
            
            self.logger.info(f"Executando treinamento para {dataset_name} com {len(shards)} shards")
            
            # Simula progresso de treinamento (placeholder para implementação real)
            for progress in range(0, 101, 10):
                if not self.running:
                    break
                
                task.update_progress(progress)
                self.logger.debug(f"Progresso da tarefa {task.id}: {progress}%")
                time.sleep(3)  # Simulação de tempo de treinamento
            
            # Atualiza unidades de trabalho concluídas
            num_samples = len(shards) * self.config.get("data", {}).get("processing", {}).get("shard_size", 1000)
            work_units = max(1, num_samples // 1000)
            with self.task_lock:
                self.completed_work_units += work_units
            
            # Simula a criação de arquivos de checkpoint (placeholder para implementação real)
            checkpoint_file = self.model_dir / f"checkpoint_{task.id}.pt"
            with open(checkpoint_file, 'w') as f:
                f.write(f"Simulação de checkpoint para {task.id}")
            
            # Marca a tarefa como concluída
            result = {
                "status": "success", 
                "dataset": dataset_name, 
                "checkpoint": str(checkpoint_file)
            }
            task.complete(result)
            self.logger.info(f"Treinamento do shard {task.id} concluído com sucesso")
            
            # Atualiza a lista de partes do modelo para merge posterior
            if TaskType.MERGE_MODEL in [t.type for t in self.task_queue.queue]:
                for qt in self.task_queue.queue:
                    if qt.type == TaskType.MERGE_MODEL:
                        qt.params["model_parts"].append(str(checkpoint_file))
        
        except Exception as e:
            # Em caso de erro, marca a tarefa como falha
            task.fail(str(e))
            self.logger.error(f"Erro em tarefa de treinamento: {str(e)}", exc_info=True)
        
        finally:
            # Remove da lista de tarefas ativas e adiciona à lista adequada
            with self.task_lock:
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                
                if task.status == TrainingStatus.COMPLETED:
                    self.completed_tasks.append(task)
                else:
                    self.failed_tasks.append(task)
    
    def _execute_merge_task(self, task):
        """Executa uma tarefa de mesclagem de modelo em background."""
        try:
            model_parts = task.params.get("model_parts", [])
            output_dir = Path(task.params.get("output_dir", str(self.model_dir)))
            
            self.logger.info(f"Executando mesclagem de modelo com {len(model_parts)} partes")
            
            # Simula progresso de mesclagem (placeholder para implementação real)
            for progress in range(0, 101, 20):
                if not self.running:
                    break
                
                task.update_progress(progress)
                self.logger.debug(f"Progresso da mesclagem: {progress}%")
                time.sleep(2)  # Simulação de tempo de mesclagem
            
            # Simula a criação do modelo final (placeholder para implementação real)
            final_model_file = output_dir / f"model_final_{int(time.time())}.pt"
            with open(final_model_file, 'w') as f:
                f.write(f"Simulação de modelo final com {len(model_parts)} partes")
            
            # Marca a tarefa como concluída
            result = {"status": "success", "model_path": str(final_model_file)}
            task.complete(result)
            self.logger.info(f"Mesclagem de modelo concluída com sucesso: {final_model_file}")
            
            # Marca o treinamento como concluído
            self.status = TrainingStatus.COMPLETED
            self.end_time = datetime.now()
            
        except Exception as e:
            # Em caso de erro, marca a tarefa como falha
            task.fail(str(e))
            self.logger.error(f"Erro em tarefa de mesclagem: {str(e)}", exc_info=True)
            self.status = TrainingStatus.FAILED
        
        finally:
            # Remove da lista de tarefas ativas e adiciona à lista adequada
            with self.task_lock:
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                
                if task.status == TrainingStatus.COMPLETED:
                    self.completed_tasks.append(task)
                else:
                    self.failed_tasks.append(task)
    
    def _progress_monitor(self):
        """Thread que monitora o progresso do treinamento."""
        last_log = time.time()
        log_interval = self.config.get("logging", {}).get("log_progress_interval", 60)
        
        while self.running:
            try:
                now = time.time()
                if now - last_log >= log_interval:
                    # Calcula progresso
                    if self.total_work_units > 0:
                        progress = round(100 * self.completed_work_units / self.total_work_units, 2)
                    else:
                        progress = 0
                    
                    # Calcula tempo estimado
                    active_nodes = len(self.active_tasks)
                    remaining_time = self.node_manager.estimate_completion_time(
                        self.total_work_units,
                        self.completed_work_units,
                        max(1, active_nodes)
                    )
                    
                    if remaining_time == float('inf'):
                        time_msg = "desconhecido"
                    else:
                        remaining_hours = int(remaining_time // 3600)
                        remaining_minutes = int((remaining_time % 3600) // 60)
                        time_msg = f"{remaining_hours}h {remaining_minutes}m"
                    
                    # Log de progresso
                    self.logger.info(
                        f"Progresso: {progress}% | "
                        f"Nós ativos: {active_nodes} | "
                        f"Tarefas concluídas: {len(self.completed_tasks)} | "
                        f"Falhas: {len(self.failed_tasks)} | "
                        f"Tempo restante: {time_msg}"
                    )
                    
                    last_log = now
                
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Erro no monitor de progresso: {str(e)}", exc_info=True)
                time.sleep(30)
    
    def _wait_for_empty_queue(self, timeout=None):
        """Aguarda até que a fila de tarefas esteja vazia, com timeout opcional."""
        start_time = time.time()
        while not self.task_queue.empty() or self.active_tasks:
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Timeout atingido ao aguardar fila vazia")
                break
            
            time.sleep(5)
    
    def _wait_for_completion(self):
        """Aguarda a conclusão de todas as tarefas de treinamento."""
        self.logger.info("Aguardando conclusão de todas as tarefas")
        
        # Aguarda a fila esvaziar
        self._wait_for_empty_queue()
        
        # Verifica o status final
        if self.status == TrainingStatus.COMPLETED:
            self.logger.info("Treinamento concluído com sucesso")
            duration = self.end_time - self.start_time
            self.logger.info(f"Tempo total de treinamento: {duration}")
        else:
            self.logger.info(f"Treinamento finalizado com status: {self.status.name}")
    
    def _check_fatal_failures(self):
        """Verifica se houve falhas fatais que impedem o prosseguimento."""
        # Implementação específica para cada tipo de falha fatal
        # Aqui consideramos fatal se todos os downloads de datasets falharam
        download_failures = [t for t in self.failed_tasks if t.type == TaskType.DOWNLOAD_DATASET]
        download_successes = [t for t in self.completed_tasks if t.type == TaskType.DOWNLOAD_DATASET]
        
        if download_failures and not download_successes:
            self.logger.error("ERRO FATAL: Todos os downloads de datasets falharam")
            return True
        
        return False
    
    def _cleanup(self, forced=False):
        """Realiza limpeza e finalização do treinamento."""
        self.logger.info("Realizando limpeza e finalização")
        
        if forced:
            # Limpeza forçada em caso de interrupção
            self.logger.info("Cancelando tarefas pendentes")
            
            # Esvazia a fila de tarefas
            while not self.task_queue.empty():
                try:
                    self.task_queue.get(block=False)
                    self.task_queue.task_done()
                except queue.Empty:
                    break
        
        # Gera relatório final
        self._generate_report()
    
    def _generate_report(self):
        """Gera um relatório final do treinamento."""
        if not self.start_time:
            return
        
        end_time = self.end_time or datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "status": self.status.name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "work_progress": f"{self.completed_work_units}/{self.total_work_units} unidades",
            "nodes_used": len(set(task.node_id for task in self.completed_tasks if task.node_id))
        }
        
        # Salva o relatório
        report_file = self.model_dir / f"training_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Relatório de treinamento salvo em {report_file}")
        
        # Imprime resumo
        if self.status == TrainingStatus.COMPLETED:
            self.logger.info(
                f"Treinamento concluído com sucesso! "
                f"Duração: {duration}, "
                f"Tarefas: {len(self.completed_tasks)}"
            )
        else:
            self.logger.info(
                f"Treinamento finalizado com status {self.status.name}. "
                f"Duração: {duration}, "
                f"Tarefas concluídas: {len(self.completed_tasks)}, "
                f"Falhas: {len(self.failed_tasks)}"
            ) 