#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema de Treinamento Distribuído para Modelo de IA focado em Programação
Desenvolvido para executar em múltiplas máquinas e plataformas (Linux/Windows)
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Adiciona os diretórios ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa os módulos do sistema
from src.core.coordinator import TrainingCoordinator
from src.infrastructure.node_manager import NodeManager
from src.data.dataset_handler import DatasetHandler
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sistema de Treinamento Distribuído de IA')
    
    parser.add_argument('--config', type=str, default='config/default_config.json',
                        help='Caminho para o arquivo de configuração')
    parser.add_argument('--mode', type=str, choices=['master', 'worker', 'standalone'],
                        default='standalone', help='Modo de execução')
    parser.add_argument('--master-ip', type=str, help='IP do nó master para modo worker')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Nível de log')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['bigcode/the-stack-v2', 'stanford-oval/ccnews'],
                        help='Datasets a serem utilizados')
    
    return parser.parse_args()

def main():
    """Função principal do sistema."""
    args = parse_arguments()
    
    # Configura logger
    log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(args.log_level, log_file)
    
    logger.info("Iniciando sistema de treinamento distribuído...")
    logger.info(f"Modo: {args.mode}")
    
    try:
        # Carrega configurações
        config = ConfigLoader(args.config)
        config_data = config.load()
        
        # Ajusta configurações com argumentos da linha de comando
        if args.master_ip:
            config_data['infrastructure']['master_ip'] = args.master_ip
        if args.datasets:
            config_data['data']['datasets'] = args.datasets
        
        # Inicializa o gerenciador de nós
        node_manager = NodeManager(config_data['infrastructure'], mode=args.mode)
        
        # Inicializa o manipulador de datasets
        dataset_handler = DatasetHandler(config_data['data'])
        
        # Inicializa o coordenador de treinamento
        coordinator = TrainingCoordinator(
            config=config_data,
            node_manager=node_manager,
            dataset_handler=dataset_handler
        )
        
        # Inicia o processo de treinamento
        if args.mode == 'master' or args.mode == 'standalone':
            # Modo master ou standalone: coordena o treinamento
            coordinator.start()
        elif args.mode == 'worker':
            # Modo worker: conecta ao master e executa trabalhos
            node_manager.connect_to_master()
            node_manager.start_worker()
        
        logger.info("Sistema encerrado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 