#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gerenciador de datasets para treinamento de modelos de IA.
Responsável por baixar, processar e gerenciar datasets do Hugging Face e outras fontes.
"""

import os
import sys
import json
import logging
import hashlib
import shutil
import time
import threading
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np

# Importações condicionais para evitar erros caso não estejam instalados
try:
    from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
    from datasets.utils.logging import set_verbosity_error
    set_verbosity_error()
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class DatasetHandler:
    """Gerencia datasets para treinamento distribuído."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurações de armazenamento
        self.storage_limit_gb = config.get("storage_limit_gb", 300)
        self.data_dir = Path(config.get("data_dir", "./storage/data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações de processamento
        self.proc_config = config.get("processing", {})
        self.num_proc = self.proc_config.get("num_proc", 1)
        self.max_tokens = self.proc_config.get("max_tokens", 1024)
        self.shuffle = self.proc_config.get("shuffle", True)
        
        # Lista de datasets a serem usados
        self.datasets_to_use = config.get("datasets", [])
        
        # Configuração de idiomas suportados
        self.languages = config.get("languages", ["python", "cpp"])
        
        # Cache de datasets carregados
        self.loaded_datasets = {}
        self.dataset_sizes = {}
        self.dataset_info = {}
        
        # Controle de sincronização
        self.load_lock = threading.Lock()
        
        # Verifica as dependências
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verifica se as dependências necessárias estão instaladas."""
        if not HF_AVAILABLE:
            self.logger.warning("Biblioteca 'datasets' não encontrada. A funcionalidade de carregamento de datasets será limitada.")
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch não encontrado. Algumas funcionalidades podem ser limitadas.")
    
    def _get_disk_usage(self, path: Path) -> float:
        """Retorna o uso de disco em GB de um diretório."""
        total_size = 0
        if path.exists():
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = Path(dirpath) / f
                    if not fp.is_symlink():
                        total_size += fp.stat().st_size
        
        return total_size / (1024 ** 3)  # Converte para GB
    
    def _clean_cache(self, required_space_gb: float) -> bool:
        """Limpa o cache para liberar espaço."""
        current_usage = self._get_disk_usage(self.data_dir)
        available_space = self.storage_limit_gb - current_usage
        
        if available_space >= required_space_gb:
            return True  # Já tem espaço suficiente
        
        space_to_free = required_space_gb - available_space
        self.logger.info(f"Tentando liberar {space_to_free:.2f} GB de espaço")
        
        # Lista os diretórios de cache por tempo de acesso
        cache_dirs = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and (item / "dataset_info.json").exists():
                try:
                    access_time = os.path.getatime(item)
                    size_gb = self._get_disk_usage(item)
                    cache_dirs.append((item, access_time, size_gb))
                except Exception as e:
                    self.logger.warning(f"Erro ao acessar {item}: {e}")
        
        # Ordena por tempo de acesso (mais antigos primeiro)
        cache_dirs.sort(key=lambda x: x[1])
        
        # Remove diretórios até liberar espaço suficiente
        freed_space = 0
        for item, _, size_gb in cache_dirs:
            if freed_space >= space_to_free:
                break
                
            # Verifica se o dataset não está entre os prioritários
            dataset_info_file = item / "dataset_info.json"
            if dataset_info_file.exists():
                try:
                    with open(dataset_info_file, 'r') as f:
                        info = json.load(f)
                        if info.get("name") in self.datasets_to_use:
                            continue  # Não remove datasets prioritários
                except Exception:
                    pass  # Se não conseguir ler, considera removível
            
            try:
                self.logger.info(f"Removendo cache de {item.name} para liberar {size_gb:.2f} GB")
                shutil.rmtree(item)
                freed_space += size_gb
            except Exception as e:
                self.logger.error(f"Erro ao remover {item}: {e}")
        
        # Verifica se conseguiu liberar espaço suficiente
        new_usage = self._get_disk_usage(self.data_dir)
        return (self.storage_limit_gb - new_usage) >= required_space_gb
    
    def _get_dataset_path(self, dataset_name: str) -> Path:
        """Retorna o caminho onde o dataset será armazenado."""
        # Cria um hash do nome para evitar problemas com caracteres especiais
        hashed_name = hashlib.md5(dataset_name.encode()).hexdigest()
        return self.data_dir / f"{dataset_name.replace('/', '_')}_{hashed_name}"
    
    def _save_dataset_info(self, dataset_name: str, info: Dict):
        """Salva informações sobre o dataset."""
        dataset_path = self._get_dataset_path(dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        info_file = dataset_path / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Carrega informações sobre o dataset."""
        dataset_path = self._get_dataset_path(dataset_name)
        info_file = dataset_path / "dataset_info.json"
        
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Erro ao carregar informações do dataset {dataset_name}: {e}")
        
        return None
    
    def _estimate_dataset_size(self, dataset_name: str) -> float:
        """Estima o tamanho do dataset em GB."""
        # Tenta buscar de informações salvas anteriormente
        info = self._load_dataset_info(dataset_name)
        if info and "size_gb" in info:
            return info["size_gb"]
        
        # Para datasets do HuggingFace, podemos usar a API para estimar o tamanho
        if HF_AVAILABLE and "/" in dataset_name:
            try:
                # Apenas carrega as informações do dataset, não os dados
                from huggingface_hub import HfApi
                api = HfApi()
                dataset_info = api.dataset_info(dataset_name)
                
                # Estima o tamanho com base nos dados da API
                size_bytes = sum(getattr(dataset_info, "size", 0) for file in getattr(dataset_info, "siblings", []))
                size_gb = size_bytes / (1024 ** 3)
                
                # Se for muito pequeno, assume um valor mínimo razoável
                if size_gb < 0.1:
                    size_gb = 0.5
                
                return size_gb
            except Exception as e:
                self.logger.warning(f"Erro ao estimar tamanho do dataset {dataset_name}: {e}")
        
        # Se não conseguiu estimar, usa um valor padrão conservador
        return 5.0  # 5 GB como estimativa conservadora
    
    def _download_dataset(self, dataset_name: str) -> bool:
        """Baixa um dataset."""
        if not HF_AVAILABLE:
            self.logger.error("Biblioteca 'datasets' não disponível para baixar datasets")
            return False
        
        estimated_size = self._estimate_dataset_size(dataset_name)
        self.logger.info(f"Tamanho estimado do dataset {dataset_name}: {estimated_size:.2f} GB")
        
        # Verifica se há espaço disponível
        if not self._clean_cache(estimated_size):
            self.logger.error(f"Espaço insuficiente para baixar {dataset_name}")
            return False
        
        dataset_path = self._get_dataset_path(dataset_name)
        cache_dir = str(dataset_path)
        
        try:
            self.logger.info(f"Baixando dataset {dataset_name}...")
            start_time = time.time()
            
            # Carrega o dataset do Hugging Face
            dataset = load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                num_proc=self.num_proc
            )
            
            download_time = time.time() - start_time
            
            # Salva informações sobre o dataset
            actual_size = self._get_disk_usage(dataset_path)
            info = {
                "name": dataset_name,
                "download_date": time.time(),
                "download_time_seconds": download_time,
                "size_gb": actual_size,
                "splits": list(dataset.keys()) if isinstance(dataset, dict) else ["train"],
                "num_examples": sum(split.num_rows for split in dataset.values()) if isinstance(dataset, dict) else dataset.num_rows
            }
            
            self._save_dataset_info(dataset_name, info)
            self.logger.info(f"Dataset {dataset_name} baixado com sucesso. Tamanho: {actual_size:.2f} GB")
            
            # Armazena no cache de memória
            with self.load_lock:
                self.loaded_datasets[dataset_name] = dataset
                self.dataset_sizes[dataset_name] = actual_size
                self.dataset_info[dataset_name] = info
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao baixar dataset {dataset_name}: {e}", exc_info=True)
            
            # Remove diretório parcial em caso de falha
            if dataset_path.exists():
                try:
                    shutil.rmtree(dataset_path)
                except Exception:
                    pass
                
            return False
    
    def load_dataset(self, dataset_name: str) -> Optional[Union[Dataset, DatasetDict]]:
        """Carrega um dataset específico."""
        # Verifica se já está em memória
        with self.load_lock:
            if dataset_name in self.loaded_datasets:
                return self.loaded_datasets[dataset_name]
        
        dataset_path = self._get_dataset_path(dataset_name)
        if not dataset_path.exists():
            # Se não existe localmente, tenta baixar
            if not self._download_dataset(dataset_name):
                return None
        else:
            # Se existe localmente, carrega do disco
            info = self._load_dataset_info(dataset_name)
            
            if info:
                try:
                    dataset = load_dataset(
                        "json", 
                        data_dir=str(dataset_path),
                        cache_dir=str(dataset_path / "cache")
                    )
                    
                    with self.load_lock:
                        self.loaded_datasets[dataset_name] = dataset
                        self.dataset_sizes[dataset_name] = info.get("size_gb", 0)
                        self.dataset_info[dataset_name] = info
                    
                    return dataset
                    
                except Exception as e:
                    self.logger.error(f"Erro ao carregar dataset {dataset_name} do disco: {e}")
                    # Tenta baixar novamente
                    return self._download_dataset(dataset_name) and self.load_dataset(dataset_name)
        
        # Retorna do cache de memória
        with self.load_lock:
            return self.loaded_datasets.get(dataset_name)
    
    def load_all_datasets(self) -> Dict[str, Union[Dataset, DatasetDict]]:
        """Carrega todos os datasets configurados."""
        results = {}
        failed = []
        
        for dataset_name in self.datasets_to_use:
            self.logger.info(f"Carregando dataset: {dataset_name}")
            dataset = self.load_dataset(dataset_name)
            
            if dataset:
                results[dataset_name] = dataset
            else:
                failed.append(dataset_name)
        
        if failed:
            self.logger.warning(f"Falha ao carregar os seguintes datasets: {', '.join(failed)}")
        
        return results
    
    def filter_by_languages(self, dataset, column_name: str = "language") -> Dataset:
        """Filtra um dataset mantendo apenas as linguagens configuradas."""
        if not isinstance(dataset, (Dataset, DatasetDict)):
            self.logger.error("Dataset inválido para filtragem por idioma")
            return dataset
        
        try:
            if isinstance(dataset, DatasetDict):
                filtered = {}
                for split_name, split_dataset in dataset.items():
                    if column_name in split_dataset.column_names:
                        filtered[split_name] = split_dataset.filter(
                            lambda x: x[column_name] in self.languages,
                            num_proc=self.num_proc
                        )
                    else:
                        filtered[split_name] = split_dataset
                return DatasetDict(filtered)
            elif column_name in dataset.column_names:
                return dataset.filter(
                    lambda x: x[column_name] in self.languages,
                    num_proc=self.num_proc
                )
            return dataset
        except Exception as e:
            self.logger.error(f"Erro ao filtrar por idiomas: {e}")
            return dataset
    
    def process_dataset_for_training(self, dataset, tokenizer, max_length: int = None) -> Dataset:
        """Processa um dataset para treinamento."""
        if max_length is None:
            max_length = self.max_tokens
            
        if not isinstance(dataset, (Dataset, DatasetDict)):
            self.logger.error("Dataset inválido para processamento")
            return None
        
        try:
            def tokenize_function(examples):
                return tokenizer(
                    examples["content"] if "content" in examples else examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_special_tokens_mask=True
                )
            
            column_names = dataset.column_names if isinstance(dataset, Dataset) else dataset["train"].column_names
            text_column = "content" if "content" in column_names else "text"
            
            if isinstance(dataset, DatasetDict):
                tokenized_datasets = {}
                for split_name, split_dataset in dataset.items():
                    if text_column not in split_dataset.column_names:
                        self.logger.warning(f"Coluna de texto '{text_column}' não encontrada no split {split_name}")
                        continue
                        
                    tokenized = split_dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=self.num_proc,
                        remove_columns=[col for col in split_dataset.column_names if col != text_column]
                    )
                    
                    if self.shuffle:
                        tokenized = tokenized.shuffle(seed=42)
                        
                    tokenized_datasets[split_name] = tokenized
                
                return DatasetDict(tokenized_datasets)
            else:
                if text_column not in dataset.column_names:
                    self.logger.warning(f"Coluna de texto '{text_column}' não encontrada no dataset")
                    return None
                    
                tokenized = dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=self.num_proc,
                    remove_columns=[col for col in dataset.column_names if col != text_column]
                )
                
                if self.shuffle:
                    tokenized = tokenized.shuffle(seed=42)
                    
                return tokenized
                
        except Exception as e:
            self.logger.error(f"Erro ao processar dataset para treinamento: {e}", exc_info=True)
            return None
    
    def merge_datasets(self, datasets: List[Union[Dataset, DatasetDict]]) -> Union[Dataset, DatasetDict]:
        """Combina múltiplos datasets em um só."""
        if not datasets:
            return None
            
        if not HF_AVAILABLE:
            self.logger.error("Biblioteca 'datasets' não disponível para mesclar datasets")
            return None
        
        try:
            if all(isinstance(ds, Dataset) for ds in datasets):
                return concatenate_datasets(datasets)
            elif all(isinstance(ds, DatasetDict) for ds in datasets):
                # Combina por split
                result = {}
                all_splits = set()
                for ds in datasets:
                    all_splits.update(ds.keys())
                
                for split in all_splits:
                    split_datasets = [ds[split] for ds in datasets if split in ds]
                    if split_datasets:
                        result[split] = concatenate_datasets(split_datasets)
                
                return DatasetDict(result)
            else:
                self.logger.error("Não é possível mesclar tipos diferentes de datasets")
                return None
        except Exception as e:
            self.logger.error(f"Erro ao mesclar datasets: {e}", exc_info=True)
            return None
    
    def get_dataset_stats(self) -> Dict:
        """Retorna estatísticas sobre os datasets carregados."""
        stats = {
            "total_datasets": len(self.loaded_datasets),
            "total_size_gb": sum(self.dataset_sizes.values()),
            "datasets": []
        }
        
        for name, dataset in self.loaded_datasets.items():
            info = self.dataset_info.get(name, {})
            
            dataset_stats = {
                "name": name,
                "size_gb": self.dataset_sizes.get(name, 0),
                "num_examples": info.get("num_examples", 0),
                "splits": info.get("splits", []),
                "download_date": info.get("download_date")
            }
            
            if isinstance(dataset, DatasetDict):
                dataset_stats["samples_per_split"] = {
                    split: ds.num_rows for split, ds in dataset.items()
                }
            elif isinstance(dataset, Dataset):
                dataset_stats["samples"] = dataset.num_rows
            
            stats["datasets"].append(dataset_stats)
        
        return stats
    
    def prepare_shards(self, dataset_name: str, shard_size: int = None) -> List[Path]:
        """Prepara shards do dataset para treinamento distribuído."""
        if shard_size is None:
            shard_size = self.proc_config.get("shard_size", 1000)
            
        dataset = self.load_dataset(dataset_name)
        if not dataset:
            return []
            
        dataset_path = self._get_dataset_path(dataset_name)
        shards_dir = dataset_path / "shards"
        shards_dir.mkdir(exist_ok=True)
        
        # Limpa shards antigos
        for old_shard in shards_dir.glob("*.json"):
            old_shard.unlink()
        
        shard_paths = []
        
        try:
            if isinstance(dataset, DatasetDict):
                # Prepara shards para cada split
                for split_name, split_dataset in dataset.items():
                    num_samples = split_dataset.num_rows
                    num_shards = max(1, (num_samples + shard_size - 1) // shard_size)
                    
                    for i in range(num_shards):
                        start_idx = i * shard_size
                        end_idx = min((i + 1) * shard_size, num_samples)
                        
                        shard = split_dataset.select(range(start_idx, end_idx))
                        shard_path = shards_dir / f"{dataset_name.replace('/', '_')}_{split_name}_shard_{i:05d}.json"
                        
                        shard.to_json(shard_path)
                        shard_paths.append(shard_path)
            else:
                # Dataset único
                num_samples = dataset.num_rows
                num_shards = max(1, (num_samples + shard_size - 1) // shard_size)
                
                for i in range(num_shards):
                    start_idx = i * shard_size
                    end_idx = min((i + 1) * shard_size, num_samples)
                    
                    shard = dataset.select(range(start_idx, end_idx))
                    shard_path = shards_dir / f"{dataset_name.replace('/', '_')}_shard_{i:05d}.json"
                    
                    shard.to_json(shard_path)
                    shard_paths.append(shard_path)
            
            self.logger.info(f"Preparados {len(shard_paths)} shards para {dataset_name}")
            return shard_paths
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar shards para {dataset_name}: {e}", exc_info=True)
            return []
            
    def list_available_datasets(self) -> List[str]:
        """Lista todos os datasets disponíveis localmente."""
        available = []
        
        for item in self.data_dir.iterdir():
            if item.is_dir():
                info_file = item / "dataset_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                            available.append(info.get("name", item.name))
                    except Exception:
                        pass  # Ignora erros de leitura
        
        return available
    
    def create_dataset_from_files(self, name: str, files: List[Path], text_column: str = "text") -> Optional[Dataset]:
        """Cria um dataset a partir de arquivos existentes."""
        if not HF_AVAILABLE:
            self.logger.error("Biblioteca 'datasets' não disponível para criar dataset")
            return None
            
        dataset_path = self._get_dataset_path(name)
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Cria dataset com base no tipo de arquivo
            extensions = set(file.suffix.lower() for file in files)
            
            if all(ext in ['.json', '.jsonl'] for ext in extensions):
                dataset = load_dataset('json', data_files=[str(f) for f in files], cache_dir=str(dataset_path))
            elif all(ext == '.csv' for ext in extensions):
                dataset = load_dataset('csv', data_files=[str(f) for f in files], cache_dir=str(dataset_path))
            elif all(ext == '.txt' for ext in extensions):
                # Para arquivos de texto, cria um dataset com a coluna 'text'
                raw_data = []
                for file in files:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    raw_data.append({text_column: content})
                
                dataset = Dataset.from_list(raw_data)
            elif all(ext == '.parquet' for ext in extensions):
                dataset = load_dataset('parquet', data_files=[str(f) for f in files], cache_dir=str(dataset_path))
            else:
                self.logger.info(f"Tentando processar tipos de arquivo não padrão: {extensions}")
                
                # Classificando arquivos por tipo
                image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']]
                audio_files = [f for f in files if f.suffix.lower() in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']]
                zip_files = [f for f in files if f.suffix.lower() in ['.zip']]
                
                # Listas para resultados
                datasets = []
                
                # Processando os arquivos por tipo
                if image_files:
                    self.logger.info(f"Processando {len(image_files)} arquivos de imagem")
                    img_dataset = self._process_image_files(image_files, dataset_path)
                    if img_dataset:
                        datasets.append(img_dataset)
                
                if audio_files:
                    self.logger.info(f"Processando {len(audio_files)} arquivos de áudio")
                    audio_dataset = self._process_audio_files(audio_files, dataset_path)
                    if audio_dataset:
                        datasets.append(audio_dataset)
                
                if zip_files:
                    self.logger.info(f"Processando {len(zip_files)} arquivos zip")
                    zip_dataset = self._process_zip_files(zip_files, dataset_path)
                    if zip_dataset:
                        datasets.append(zip_dataset)
                
                # Arquivos restantes são tratados como binários genéricos
                other_files = [f for f in files if f not in image_files and f not in audio_files and f not in zip_files]
                if other_files:
                    self.logger.info(f"Processando {len(other_files)} arquivos binários genéricos")
                    bin_dataset = self._process_binary_files(other_files, dataset_path)
                    if bin_dataset:
                        datasets.append(bin_dataset)
                
                if not datasets:
                    self.logger.error("Nenhum arquivo pôde ser processado com sucesso")
                    return None
                
                # Se tivermos apenas um tipo de dataset
                if len(datasets) == 1:
                    return datasets[0]
                
                # Se tivermos múltiplos tipos, criamos um dataset misto
                try:
                    # Adicionamos uma coluna para identificar o tipo
                    for i, ds in enumerate(datasets):
                        ds = ds.add_column("content_type", ["image" if i==0 else "audio" if i==1 else "binary"] * ds.num_rows)
                        datasets[i] = ds
                    
                    # Combinamos os datasets
                    combined = self.merge_datasets(datasets)
                    self.logger.info(f"Dataset misto criado com sucesso: {combined.num_rows} entradas")
                    return combined
                except Exception as e:
                    self.logger.error(f"Falha ao combinar datasets de diferentes tipos: {str(e)}")
                    # Retornar apenas o primeiro dataset como fallback
                    return datasets[0] if datasets else None
            
            # Salva informações
            actual_size = self._get_disk_usage(dataset_path)
            info = {
                "name": name,
                "creation_date": time.time(),
                "size_gb": actual_size,
                "splits": list(dataset.keys()) if isinstance(dataset, dict) else ["train"],
                "num_examples": sum(split.num_rows for split in dataset.values()) if isinstance(dataset, dict) else dataset.num_rows,
                "source": "local_files",
                "files": [str(f) for f in files]
            }
            
            self._save_dataset_info(name, info)
            
            # Armazena no cache de memória
            with self.load_lock:
                self.loaded_datasets[name] = dataset
                self.dataset_sizes[name] = actual_size
                self.dataset_info[name] = info
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Erro ao criar dataset a partir de arquivos: {e}", exc_info=True)
            return None
    
    def _process_image_files(self, files: List[Path], dataset_path: Path) -> Optional[Dataset]:
        """Processa arquivos de imagem para um Dataset."""
        try:
            from PIL import Image
            import io
            import base64
            
            image_data = []
            for file in files:
                try:
                    # Abre a imagem e converte para base64
                    with open(file, 'rb') as img_file:
                        img_bytes = img_file.read()
                        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Obtém metadados da imagem
                    with Image.open(file) as img:
                        width, height = img.size
                        format_type = img.format
                    
                    # Cria entrada no dataset
                    image_data.append({
                        "file_name": file.name,
                        "file_path": str(file),
                        "image_b64": img_b64,
                        "width": width,
                        "height": height,
                        "format": format_type,
                        "size_bytes": len(img_bytes)
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Erro ao processar imagem {file}: {str(e)}")
            
            # Cria dataset a partir dos dados processados
            if image_data:
                return Dataset.from_list(image_data)
            return None
        
        except ImportError:
            self.logger.error("Bibliotecas necessárias não disponíveis para processar imagens (PIL)")
            return None

    def _process_audio_files(self, files: List[Path], dataset_path: Path) -> Optional[Dataset]:
        """Processa arquivos de áudio para um Dataset."""
        try:
            import base64
            
            audio_data = []
            for file in files:
                try:
                    # Lê o arquivo de áudio em bytes e converte para base64
                    with open(file, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    # Obtém metadados básicos do arquivo
                    size_bytes = len(audio_bytes)
                    
                    # Tenta obter mais metadados se librosa estiver disponível
                    duration = None
                    sample_rate = None
                    try:
                        import librosa
                        y, sr = librosa.load(file, sr=None)
                        duration = librosa.get_duration(y=y, sr=sr)
                        sample_rate = sr
                    except ImportError:
                        pass
                    
                    # Cria entrada no dataset
                    audio_data.append({
                        "file_name": file.name,
                        "file_path": str(file),
                        "audio_b64": audio_b64,
                        "format": file.suffix.lower()[1:],
                        "size_bytes": size_bytes,
                        "duration": duration,
                        "sample_rate": sample_rate
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Erro ao processar áudio {file}: {str(e)}")
            
            # Cria dataset a partir dos dados processados
            if audio_data:
                return Dataset.from_list(audio_data)
            return None
        
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivos de áudio: {str(e)}")
            return None

    def _process_binary_files(self, files: List[Path], dataset_path: Path) -> Optional[Dataset]:
        """Processa arquivos binários genéricos para um Dataset."""
        try:
            import base64
            import mimetypes
            
            binary_data = []
            for file in files:
                try:
                    # Lê o arquivo em bytes e converte para base64
                    with open(file, 'rb') as bin_file:
                        binary_bytes = bin_file.read()
                        binary_b64 = base64.b64encode(binary_bytes).decode('utf-8')
                    
                    # Tenta identificar o tipo MIME
                    mime_type, _ = mimetypes.guess_type(file)
                    
                    # Cria entrada no dataset
                    binary_data.append({
                        "file_name": file.name,
                        "file_path": str(file),
                        "extension": file.suffix.lower(),
                        "mime_type": mime_type or "application/octet-stream",
                        "content_b64": binary_b64,
                        "size_bytes": len(binary_bytes)
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Erro ao processar arquivo {file}: {str(e)}")
            
            # Cria dataset a partir dos dados processados
            if binary_data:
                return Dataset.from_list(binary_data)
            return None
        
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivos binários: {str(e)}")
            return None

    def _process_zip_files(self, files: List[Path], dataset_path: Path) -> Optional[Dataset]:
        """Processa arquivos zip para um Dataset."""
        try:
            import zipfile
            import tempfile
            import os
            import shutil
            
            # Criamos um diretório temporário para extrair os arquivos
            temp_dir = Path(tempfile.mkdtemp(dir=dataset_path))
            
            extracted_files = []
            zip_info = []
            
            try:
                # Processamos cada arquivo zip
                for zip_file in files:
                    try:
                        # Informações básicas do arquivo zip
                        zip_info_entry = {
                            "file_name": zip_file.name,
                            "file_path": str(zip_file),
                            "size_bytes": zip_file.stat().st_size,
                            "contents": []
                        }
                        
                        # Criamos um subdiretório para este zip
                        extract_dir = temp_dir / zip_file.stem
                        extract_dir.mkdir(exist_ok=True)
                        
                        # Extraímos o conteúdo
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            # Listamos os conteúdos
                            for item in zip_ref.infolist():
                                zip_info_entry["contents"].append({
                                    "name": item.filename,
                                    "size": item.file_size,
                                    "compressed_size": item.compress_size,
                                    "is_dir": item.is_dir()
                                })
                            
                            # Extraímos tudo
                            zip_ref.extractall(extract_dir)
                        
                        # Adicionamos este zip ao dataset
                        zip_info.append(zip_info_entry)
                        
                        # Encontramos todos os arquivos extraídos
                        for root, _, filenames in os.walk(extract_dir):
                            for filename in filenames:
                                extracted_files.append(Path(os.path.join(root, filename)))
                                
                    except Exception as e:
                        self.logger.warning(f"Erro ao processar arquivo zip {zip_file}: {str(e)}")
                
                if not extracted_files:
                    return Dataset.from_list(zip_info)  # Retornamos apenas as informações do zip
                    
                # Processamos os arquivos extraídos com base em seus tipos
                result_dataset = self.create_dataset_from_files(
                    f"{dataset_path.name}_zip_contents", 
                    extracted_files
                )
                
                # Adicionamos informações do zip ao dataset resultante
                if result_dataset:
                    # Adicionamos uma coluna com a informação de origem (arquivo zip)
                    zip_sources = [next(z["file_name"] for z in zip_info if Path(z["file_path"]).stem in f.parents) 
                                  if any(Path(z["file_path"]).stem in str(f.parents) for z in zip_info) 
                                  else "unknown" 
                                  for f in extracted_files]
                    
                    result_dataset = result_dataset.add_column("zip_source", zip_sources)
                    
                    # Também adicionamos os metadados dos zips
                    result_dataset = result_dataset.add_column("zip_metadata", [zip_info] * len(result_dataset))
                    
                    return result_dataset
                else:
                    # Se falhar, pelo menos retornamos as informações dos arquivos zip
                    return Dataset.from_list(zip_info)
                
            finally:
                # Limpamos os arquivos temporários
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Erro ao limpar diretório temporário: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivos zip: {str(e)}")
            return None 