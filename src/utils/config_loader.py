#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Carregador de configurações para o sistema de treinamento.
Responsável por carregar, validar e mesclar configurações.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """Carregador de configurações do sistema."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.default_config = {
            "project": {
                "name": "Relex.AI",
                "description": "Modelo de IA multimodal para programação",
                "version": "0.1.0"
            },
            "infrastructure": {
                "master_ip": "0.0.0.0",
                "master_port": 12345,
                "worker_port": 12346,
                "communication_protocol": "grpc",
                "max_workers": 10,
                "master_storage_path": "./storage/model",
                "connection_timeout": 60,
                "heartbeat_interval": 10
            },
            "training": {
                "model_type": "transformer",
                "model_size": "medium",
                "batch_size": 4,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "epochs": 3,
                "save_steps": 10000,
                "max_seq_length": 1024,
                "fp16": False,
                "distributed": True
            },
            "data": {
                "datasets": ["bigcode/the-stack-v2"],
                "storage_limit_gb": 300,
                "model_storage_limit_gb": 500,
                "data_dir": "./storage/data",
                "processing": {
                    "num_proc": 4,
                    "max_tokens": 1024,
                    "shuffle": True,
                    "shard_size": 1000
                },
                "languages": ["python", "cpp"]
            },
            "logging": {
                "level": "INFO",
                "save_to_file": True,
                "log_dir": "./logs",
                "log_progress_interval": 60
            }
        }
    
    def load(self) -> Dict[str, Any]:
        """Carrega a configuração a partir do arquivo."""
        try:
            config_path = Path(self.config_path)
            
            if not config_path.exists():
                self.logger.warning(f"Arquivo de configuração não encontrado: {self.config_path}")
                self.logger.info("Usando configuração padrão")
                return self.default_config
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Mescla com a configuração padrão
            merged_config = self._merge_configs(self.default_config, config)
            
            # Valida a configuração
            self._validate_config(merged_config)
            
            self.logger.info(f"Configuração carregada de {self.config_path}")
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {str(e)}")
            self.logger.info("Usando configuração padrão")
            return self.default_config
    
    def _merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mescla configurações, mantendo valores padrão para chaves ausentes."""
        result = default_config.copy()
        
        for key, value in user_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Valida as configurações e corrige valores inválidos."""
        # Garante que os tamanhos de model_storage_limit_gb e storage_limit_gb sejam válidos
        if config.get("data", {}).get("model_storage_limit_gb", 0) <= 0:
            config["data"]["model_storage_limit_gb"] = self.default_config["data"]["model_storage_limit_gb"]
            self.logger.warning("model_storage_limit_gb inválido, usando valor padrão")
        
        if config.get("data", {}).get("storage_limit_gb", 0) <= 0:
            config["data"]["storage_limit_gb"] = self.default_config["data"]["storage_limit_gb"]
            self.logger.warning("storage_limit_gb inválido, usando valor padrão")
        
        # Garante que o número de processos seja válido
        num_proc = config.get("data", {}).get("processing", {}).get("num_proc", 0)
        if num_proc <= 0:
            config["data"]["processing"]["num_proc"] = self.default_config["data"]["processing"]["num_proc"]
            self.logger.warning("num_proc inválido, usando valor padrão")
    
    def save(self, config: Dict[str, Any], output_path: Optional[str] = None) -> bool:
        """Salva a configuração em um arquivo."""
        try:
            save_path = output_path or self.config_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Configuração salva em {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar configuração: {str(e)}")
            return False 