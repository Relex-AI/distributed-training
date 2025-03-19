#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gerenciador de nós para treinamento distribuído.
Responsável por conectar e gerenciar os nós de trabalho.
"""

import os
import sys
import time
import json
import socket
import subprocess
import threading
import logging
import platform
import psutil
from enum import Enum
import grpc
import paramiko
import concurrent.futures
import numpy as np

# Tenta importar módulos específicos de cada SO
try:
    import pywinrm  # Para Windows Remote Management
except ImportError:
    pywinrm = None

try:
    import pyvmomi  # Para gerenciamento de VMware
except ImportError:
    pyvmomi = None

# Enumeração para status do nó
class NodeStatus(Enum):
    OFFLINE = 0
    ONLINE = 1
    BUSY = 2
    ERROR = 3
    INITIALIZING = 4

class Node:
    """Representa um nó de trabalho no sistema distribuído."""
    
    def __init__(self, node_id, ip, port, specs=None):
        self.id = node_id
        self.ip = ip
        self.port = port
        self.status = NodeStatus.OFFLINE
        self.specs = specs or {}
        self.connection = None
        self.last_heartbeat = None
        self.current_task = None
        self.load_avg = 0.0
        self.memory_usage = 0.0
        self.disk_usage = 0.0
        self.connected_since = None
        self.error_message = None
        
    def to_dict(self):
        """Converte o objeto Node para dicionário."""
        return {
            "id": self.id,
            "ip": self.ip,
            "port": self.port,
            "status": self.status.name,
            "specs": self.specs,
            "last_heartbeat": self.last_heartbeat,
            "current_task": self.current_task,
            "load_avg": self.load_avg,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "connected_since": self.connected_since,
            "error_message": self.error_message
        }
    
    def update_specs(self, specs):
        """Atualiza as especificações do nó."""
        self.specs.update(specs)
    
    def update_status(self, status, error_message=None):
        """Atualiza o status do nó."""
        self.status = status
        if error_message:
            self.error_message = error_message
        if status == NodeStatus.ONLINE and not self.connected_since:
            self.connected_since = time.time()
    
    def update_metrics(self, load_avg, memory_usage, disk_usage):
        """Atualiza as métricas de desempenho do nó."""
        self.load_avg = load_avg
        self.memory_usage = memory_usage
        self.disk_usage = disk_usage
        self.last_heartbeat = time.time()

class NodeManager:
    """Gerencia os nós no sistema distribuído."""
    
    def __init__(self, config, mode="standalone"):
        self.config = config
        self.mode = mode
        self.master_ip = config.get("master_ip", "0.0.0.0")
        self.master_port = config.get("master_port", 12345)
        self.worker_port = config.get("worker_port", 12346)
        self.nodes = {}
        self.node_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.server = None
        self.server_thread = None
        self.running = False
        self.local_node = self._create_local_node()
        
        # Configurações para acessos remotos
        self.remote_access = config.get("remote_access", {})
        
        # Executor para tarefas em threads
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
    def _create_local_node(self):
        """Cria um nó para representar a máquina local."""
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        node_id = f"local_{hostname}"
        
        # Coleta informações do sistema
        specs = {
            "os": platform.system(),
            "os_version": platform.version(),
            "hostname": hostname,
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory": psutil.virtual_memory().total / (1024 ** 3),  # GB
            "total_disk": psutil.disk_usage('/').total / (1024 ** 3),  # GB
            "platform": platform.platform(),
            "processor": platform.processor()
        }
        
        # Adiciona informações de GPU se disponível
        try:
            import torch
            specs["cuda_available"] = torch.cuda.is_available()
            specs["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
            specs["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if specs["gpu_count"] > 0:
                specs["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(specs["gpu_count"])]
        except ImportError:
            specs["cuda_available"] = False
            specs["gpu_count"] = 0
        
        node = Node(node_id, ip, self.worker_port, specs)
        node.update_status(NodeStatus.ONLINE)
        return node
    
    def start(self):
        """Inicia o gerenciador de nós."""
        self.running = True
        if self.mode == "master" or self.mode == "standalone":
            self._start_master()
        elif self.mode == "worker":
            self._start_worker()
    
    def stop(self):
        """Para o gerenciador de nós."""
        self.running = False
        if self.server:
            self.server.stop(0)
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
    
    def _start_master(self):
        """Inicia o nó como master."""
        self.logger.info(f"Iniciando nó master na porta {self.master_port}")
        # Implementação do servidor gRPC para o master
        # Este é um placeholder para a implementação real
        self.server_thread = threading.Thread(target=self._run_master_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Inicia o thread de monitoramento de nós
        self.heartbeat_thread = threading.Thread(target=self._monitor_nodes)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def _run_master_server(self):
        """Executa o servidor master."""
        # Placeholder para implementação do servidor gRPC
        self.logger.info("Servidor master em execução")
        
        # Mantem o servidor rodando até o flag running ser False
        while self.running:
            time.sleep(1)
    
    def _start_worker(self):
        """Inicia o nó como worker."""
        self.logger.info(f"Iniciando nó worker na porta {self.worker_port}")
        # Implementação do servidor gRPC para o worker
        # Este é um placeholder para a implementação real
        self.server_thread = threading.Thread(target=self._run_worker_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Conecta ao nó master
        self.connect_to_master()
    
    def _run_worker_server(self):
        """Executa o servidor worker."""
        # Placeholder para implementação do servidor gRPC do worker
        self.logger.info("Servidor worker em execução")
        
        # Mantem o servidor rodando até o flag running ser False
        while self.running:
            time.sleep(1)
    
    def connect_to_master(self):
        """Conecta este nó worker ao master."""
        if self.mode != "worker":
            self.logger.warning("Tentativa de conectar ao master, mas não é um worker")
            return False
        
        self.logger.info(f"Conectando ao master em {self.master_ip}:{self.master_port}")
        
        # Placeholder para implementação da conexão com o master
        # Aqui seria implementada a lógica real de conexão e registro
        # Usando gRPC ou outro protocolo definido
        
        # Simulação de conexão bem-sucedida
        time.sleep(2)
        self.logger.info("Conectado ao master com sucesso")
        return True
    
    def register_node(self, node):
        """Registra um nó no sistema."""
        with self.node_lock:
            if node.id in self.nodes:
                self.logger.warning(f"Nó já registrado: {node.id}")
                return False
            
            self.nodes[node.id] = node
            self.logger.info(f"Nó registrado: {node.id} ({node.ip}:{node.port})")
            return True
    
    def unregister_node(self, node_id):
        """Remove o registro de um nó do sistema."""
        with self.node_lock:
            if node_id not in self.nodes:
                self.logger.warning(f"Nó não encontrado para remoção: {node_id}")
                return False
            
            del self.nodes[node_id]
            self.logger.info(f"Nó removido: {node_id}")
            return True
    
    def get_nodes(self, status=None):
        """Retorna a lista de nós, opcionalmente filtrada por status."""
        with self.node_lock:
            if status is None:
                return list(self.nodes.values())
            
            return [node for node in self.nodes.values() if node.status == status]
    
    def get_node(self, node_id):
        """Retorna um nó específico pelo ID."""
        with self.node_lock:
            return self.nodes.get(node_id)
    
    def _monitor_nodes(self):
        """Thread que monitora os nós conectados."""
        self.logger.info("Iniciando monitoramento de nós")
        
        while self.running:
            try:
                self._check_node_heartbeats()
                time.sleep(self.config.get("heartbeat_interval", 10))
            except Exception as e:
                self.logger.error(f"Erro no monitoramento de nós: {str(e)}", exc_info=True)
    
    def _check_node_heartbeats(self):
        """Verifica os heartbeats dos nós e marca inativos."""
        now = time.time()
        timeout = self.config.get("connection_timeout", 60)
        
        with self.node_lock:
            for node_id, node in list(self.nodes.items()):
                if node.status == NodeStatus.OFFLINE:
                    continue
                
                if node.last_heartbeat and (now - node.last_heartbeat) > timeout:
                    self.logger.warning(f"Nó {node_id} não responde. Último heartbeat: {node.last_heartbeat}")
                    node.update_status(NodeStatus.OFFLINE, "Timeout de heartbeat")
    
    def connect_to_remote_node(self, ip, auth_type, credentials):
        """Conecta a um nó remoto usando um dos protocolos suportados."""
        if auth_type == "ssh":
            return self._connect_ssh(ip, credentials)
        elif auth_type == "rdp":
            return self._connect_rdp(ip, credentials)
        elif auth_type == "vnc":
            return self._connect_vnc(ip, credentials)
        elif auth_type == "spice":
            return self._connect_spice(ip, credentials)
        else:
            self.logger.error(f"Tipo de autenticação não suportado: {auth_type}")
            return False
    
    def _connect_ssh(self, ip, credentials):
        """Conecta a um nó remoto usando SSH."""
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                ip,
                username=credentials.get("username"),
                password=credentials.get("password"),
                key_filename=credentials.get("key_file"),
                timeout=30
            )
            
            # Executa um comando de teste
            stdin, stdout, stderr = client.exec_command("uname -a")
            result = stdout.read().decode("utf-8")
            self.logger.info(f"Conexão SSH estabelecida com {ip}: {result}")
            
            return client
        except Exception as e:
            self.logger.error(f"Erro ao conectar via SSH a {ip}: {str(e)}")
            return None
    
    def _connect_rdp(self, ip, credentials):
        """Conecta a um nó remoto usando RDP (Windows Remote Desktop)."""
        if not pywinrm:
            self.logger.error("Módulo pywinrm não disponível para conexão RDP")
            return None
        
        try:
            # Este é apenas um exemplo de conexão WinRM, não uma conexão RDP real
            # Para RDP real, seria necessário usar um cliente RDP específico
            session = pywinrm.Session(
                ip,
                auth=(credentials.get("username"), credentials.get("password"))
            )
            
            # Executa um comando de teste
            result = session.run_cmd("ipconfig")
            self.logger.info(f"Conexão WinRM estabelecida com {ip}")
            
            return session
        except Exception as e:
            self.logger.error(f"Erro ao conectar via WinRM a {ip}: {str(e)}")
            return None
    
    def _connect_vnc(self, ip, credentials):
        """Conecta a um nó remoto usando VNC."""
        # Implementação de conexão VNC dependeria de uma biblioteca específica
        # ou de execução de cliente VNC externo
        self.logger.warning("Conexão VNC não implementada")
        return None
    
    def _connect_spice(self, ip, credentials):
        """Conecta a um nó remoto usando SPICE."""
        # Implementação de conexão SPICE dependeria de uma biblioteca específica
        # ou de execução de cliente SPICE externo
        self.logger.warning("Conexão SPICE não implementada")
        return None
    
    def execute_command(self, node_id, command):
        """Executa um comando em um nó remoto."""
        node = self.get_node(node_id)
        if not node or node.status != NodeStatus.ONLINE:
            self.logger.error(f"Nó {node_id} não está online para executar comandos")
            return None
        
        if not node.connection:
            self.logger.error(f"Nó {node_id} não tem conexão estabelecida")
            return None
        
        try:
            # Implementação depende do tipo de conexão
            if isinstance(node.connection, paramiko.SSHClient):
                stdin, stdout, stderr = node.connection.exec_command(command)
                return {
                    "stdout": stdout.read().decode("utf-8"),
                    "stderr": stderr.read().decode("utf-8"),
                    "exit_code": stdout.channel.recv_exit_status()
                }
            else:
                self.logger.error(f"Tipo de conexão não suportado para execução de comandos")
                return None
        except Exception as e:
            self.logger.error(f"Erro ao executar comando em {node_id}: {str(e)}")
            return None
    
    def get_system_info(self):
        """Retorna informações do sistema local."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_io": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            },
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        }
    
    def estimate_completion_time(self, total_work, completed_work, num_nodes):
        """Estima o tempo restante para completar o trabalho."""
        if completed_work == 0:
            return float('inf')
        
        work_per_node = total_work / max(1, num_nodes)
        percent_complete = completed_work / total_work
        
        # Assumindo progresso linear
        if percent_complete > 0:
            estimated_total_time = total_work / (completed_work / time.time())
            remaining_time = estimated_total_time * (1 - percent_complete)
            return remaining_time
        
        return float('inf')
    
    def optimize_workload_distribution(self, nodes, total_work):
        """Otimiza a distribuição de trabalho com base nas capacidades dos nós."""
        if not nodes:
            return {}
        
        # Calcula a capacidade relativa de cada nó
        capacities = {}
        total_capacity = 0
        
        for node in nodes:
            # Fator de capacidade baseado em CPU e memória
            cpu_factor = node.specs.get("cpu_count", 1)
            memory_factor = node.specs.get("total_memory", 4)
            
            # Ajusta baseado em uso atual
            cpu_load = max(0.1, 1.0 - node.load_avg / 100.0)
            memory_avail = max(0.1, 1.0 - node.memory_usage / 100.0)
            
            # Capacidade relativa deste nó
            capacity = cpu_factor * cpu_load * memory_factor * memory_avail
            capacities[node.id] = capacity
            total_capacity += capacity
        
        # Distribui trabalho proporcional à capacidade
        distribution = {}
        for node_id, capacity in capacities.items():
            proportion = capacity / total_capacity
            distribution[node_id] = int(total_work * proportion)
        
        return distribution
    
    def start_worker(self):
        """Inicia operação como worker."""
        self.logger.info("Iniciando operação como worker")
        
        # Atualiza o status
        self.local_node.update_status(NodeStatus.ONLINE)
        
        # Loop principal do worker
        while self.running:
            try:
                # Atualiza métricas do sistema
                system_info = self.get_system_info()
                self.local_node.update_metrics(
                    system_info["cpu_percent"],
                    system_info["memory_percent"],
                    system_info["disk_percent"]
                )
                
                # Aguarda comandos do master (implementação real usaria gRPC ou similar)
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Erro na operação do worker: {str(e)}", exc_info=True)
                time.sleep(10)  # Espera antes de tentar novamente após erro
    
    def get_platform_setup_commands(self, platform_name):
        """Retorna os comandos de configuração para uma plataforma específica."""
        platforms = self.config.get("platforms", {})
        platform_config = platforms.get(platform_name)
        
        if not platform_config or not platform_config.get("enabled", False):
            return None
        
        setup_script = platform_config.get("setup_script")
        if not setup_script or not os.path.exists(setup_script):
            return None
        
        with open(setup_script, 'r') as f:
            return f.read()
    
    def setup_platform(self, platform_name, node_id=None):
        """Configura uma plataforma específica em um nó."""
        commands = self.get_platform_setup_commands(platform_name)
        if not commands:
            self.logger.error(f"Comandos de configuração não encontrados para {platform_name}")
            return False
        
        if node_id:
            # Executa no nó remoto
            result = self.execute_command(node_id, commands)
            return result and result.get("exit_code", 1) == 0
        else:
            # Executa localmente
            try:
                result = subprocess.run(
                    commands, 
                    shell=True, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Erro ao configurar {platform_name} localmente: {e}")
                return False 