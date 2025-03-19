#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de configuração de logging para o sistema de treinamento.
Responsável por configurar e gerenciar os logs do sistema.
"""

import os
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(log_level="INFO", log_file=None, log_to_console=True, name=None):
    """
    Configura o logger global do sistema.
    
    Args:
        log_level (str): Nível de log (DEBUG, INFO, WARNING, ERROR)
        log_file (str): Caminho para o arquivo de log
        log_to_console (bool): Se True, logs também são direcionados para o console
        name (str): Nome do logger (None para logger raiz)
        
    Returns:
        logging.Logger: Logger configurado
    """
    # Converte nível de log para constante do logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Obtém ou cria o logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Limpa handlers existentes
    logger.handlers = []
    
    # Define o formato do log
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Adiciona handler de console se solicitado
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Adiciona handler de arquivo se fornecido
    if log_file:
        # Garante que o diretório de log existe
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configura handler de arquivo com rotação
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Evita propagar logs para o handler raiz
    logger.propagate = False
    
    return logger

class TrainingLogger:
    """
    Logger especializado para treinamento de modelos.
    Registra métricas de treinamento e progresso.
    """
    
    def __init__(self, config, model_name, output_dir=None):
        """
        Inicializa o logger de treinamento.
        
        Args:
            config (dict): Configuração do sistema
            model_name (str): Nome do modelo sendo treinado
            output_dir (str): Diretório de saída para logs (opcional)
        """
        self.config = config
        self.model_name = model_name
        
        # Configura diretório de saída
        log_config = config.get("logging", {})
        self.log_dir = output_dir or log_config.get("log_dir", "./logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Cria arquivos de log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log_file = os.path.join(self.log_dir, f"{model_name}_{timestamp}.log")
        self.metrics_file = os.path.join(self.log_dir, f"{model_name}_{timestamp}_metrics.csv")
        
        # Configura logger principal
        self.logger = setup_logger(
            log_level=log_config.get("level", "INFO"),
            log_file=self.main_log_file,
            name=f"train.{model_name}"
        )
        
        # Inicializa arquivo de métricas
        self._init_metrics_file()
        
        # Estatísticas de treinamento
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.start_time = datetime.now()
    
    def _init_metrics_file(self):
        """Inicializa o arquivo de métricas CSV."""
        header = "epoch,step,loss,learning_rate,time_elapsed_sec"
        with open(self.metrics_file, 'w') as f:
            f.write(f"{header}\n")
    
    def log_metrics(self, metrics):
        """
        Registra métricas de treinamento.
        
        Args:
            metrics (dict): Dicionário com métricas a serem registradas
        """
        # Atualiza contadores
        self.global_step = metrics.get("step", self.global_step)
        self.epoch = metrics.get("epoch", self.epoch)
        
        # Calcula tempo decorrido
        time_elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Registra métricas no log
        metrics_str = ", ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"[E{self.epoch:3d}|S{self.global_step:8d}] {metrics_str}")
        
        # Adiciona ao arquivo CSV
        with open(self.metrics_file, 'a') as f:
            values = [
                str(self.epoch),
                str(self.global_step),
                f"{metrics.get('loss', 0):.6f}",
                f"{metrics.get('learning_rate', 0):.8f}",
                f"{time_elapsed:.2f}"
            ]
            f.write(",".join(values) + "\n")
    
    def log_checkpoint(self, checkpoint_path, metrics=None):
        """
        Registra informações sobre um checkpoint salvo.
        
        Args:
            checkpoint_path (str): Caminho para o checkpoint
            metrics (dict): Métricas associadas ao checkpoint
        """
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            self.logger.info(f"Checkpoint salvo em {checkpoint_path} | {metrics_str}")
        else:
            self.logger.info(f"Checkpoint salvo em {checkpoint_path}")
    
    def log_system_info(self, system_info):
        """
        Registra informações do sistema.
        
        Args:
            system_info (dict): Informações do sistema
        """
        self.logger.info("Informações do sistema:")
        for key, value in system_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_dataset_info(self, dataset_info):
        """
        Registra informações sobre o dataset.
        
        Args:
            dataset_info (dict): Informações do dataset
        """
        self.logger.info("Informações do dataset:")
        for key, value in dataset_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_hyperparams(self, hyperparams):
        """
        Registra hiperparâmetros do treinamento.
        
        Args:
            hyperparams (dict): Hiperparâmetros
        """
        self.logger.info("Hiperparâmetros:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_error(self, message, exc_info=True):
        """
        Registra um erro.
        
        Args:
            message (str): Mensagem de erro
            exc_info (bool): Se True, inclui informações da exceção
        """
        self.logger.error(message, exc_info=exc_info)
    
    def log_warning(self, message):
        """
        Registra um aviso.
        
        Args:
            message (str): Mensagem de aviso
        """
        self.logger.warning(message)
    
    def log_info(self, message):
        """
        Registra uma informação.
        
        Args:
            message (str): Mensagem informativa
        """
        self.logger.info(message)
    
    def log_debug(self, message):
        """
        Registra uma mensagem de debug.
        
        Args:
            message (str): Mensagem de debug
        """
        self.logger.debug(message) 