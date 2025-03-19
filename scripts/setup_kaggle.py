#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para configurar um ambiente Kaggle como nó de trabalho para o sistema
de treinamento distribuído Relex.AI.
"""

import os
import sys
import subprocess
import argparse
import logging
import time
import json
import socket
import requests
from pathlib import Path

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Processa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Configurar ambiente Kaggle como nó de trabalho')
    parser.add_argument('--master-ip', type=str, required=True, help='IP do servidor mestre')
    parser.add_argument('--master-port', type=int, default=5000, help='Porta do servidor mestre')
    parser.add_argument('--worker-port', type=int, default=5001, help='Porta do nó de trabalho')
    parser.add_argument('--working-dir', type=str, default='/kaggle/working/relex',
                        help='Diretório de trabalho para arquivos do projeto')
    parser.add_argument('--token', type=str, required=True, help='Token de autenticação')
    parser.add_argument('--storage-limit', type=float, default=0.8,
                        help='Limite de uso de armazenamento (0.0-1.0)')
    parser.add_argument('--ngrok-token', type=str, help='Token API do ngrok para criar um túnel')
    parser.add_argument('--node-name', type=str, default='',
                        help='Nome do nó (padrão: nome do host)')
    
    return parser.parse_args()

def check_gpu():
    """Verifica se uma GPU está disponível no ambiente Kaggle."""
    try:
        gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
        if 'failed' in gpu_info.lower():
            logger.warning("GPU não encontrada ou não acessível")
            return False
        
        # Extrair o nome da GPU do output
        import re
        gpu_name = re.search(r'(?:NVIDIA)?\s*(.+?)\s*\|', gpu_info)
        if gpu_name:
            logger.info(f"GPU disponível: {gpu_name.group(1).strip()}")
        else:
            logger.info("GPU disponível, mas não foi possível identificar o modelo")
            
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("GPU não disponível")
        return False

def install_dependencies():
    """Instala dependências necessárias para o projeto."""
    logger.info("Instalando dependências...")
    
    # Lista de pacotes necessários
    packages = [
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "numpy",
        "tqdm",
        "psutil",
        "py-cpuinfo",
        "ray",
        "pyzmq",
        "grpcio",
        "protobuf",
        "pyngrok"
    ]
    
    # Verifica se estamos no Kaggle
    in_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
    
    # Instalar pacotes
    for package in packages:
        try:
            if package == "torch" and in_kaggle:
                logger.info("Pulando instalação do PyTorch pois já deve estar presente no Kaggle")
                continue
                
            logger.info(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.SubprocessError as e:
            logger.error(f"Erro ao instalar {package}: {e}")
            if package in ["torch", "transformers"]:
                logger.error(f"Falha ao instalar pacote essencial {package}. Abortando.")
                sys.exit(1)
    
    logger.info("Dependências instaladas com sucesso")

def setup_ngrok(port, ngrok_token):
    """Configura ngrok para criar um túnel para o worker."""
    try:
        from pyngrok import ngrok, conf
        
        if ngrok_token:
            conf.get_default().auth_token = ngrok_token
            logger.info("Token ngrok configurado")
        else:
            logger.warning("Token ngrok não fornecido. Usando sessão não autenticada (limitada)")
        
        # Iniciar túnel ngrok
        public_url = ngrok.connect(port).public_url
        logger.info(f"Túnel ngrok estabelecido: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Erro ao configurar ngrok: {e}")
        return None

def get_system_info():
    """Coleta informações do sistema."""
    import psutil
    import platform
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_total": psutil.disk_usage('/').total,
        "disk_free": psutil.disk_usage('/').free,
    }
    
    # Tente obter informações da CPU
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        system_info["cpu_model"] = cpu_info.get('brand_raw', 'Unknown')
    except:
        system_info["cpu_model"] = "Unknown"
    
    # Adiciona informações da GPU, se disponível
    system_info["has_gpu"] = check_gpu()
    if system_info["has_gpu"]:
        try:
            gpu_info = subprocess.check_output('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader', 
                                              shell=True).decode('utf-8').strip()
            system_info["gpu_info"] = gpu_info
        except:
            system_info["gpu_info"] = "GPU disponível, mas não foi possível obter detalhes"
    
    logger.info(f"Informações do sistema coletadas: CPU: {system_info['cpu_model']}, "
                f"Memória: {system_info['memory_total'] / (1024**3):.2f} GB, "
                f"GPU: {'Sim' if system_info['has_gpu'] else 'Não'}")
    
    return system_info

def register_with_master(master_ip, master_port, worker_url, system_info, token, node_name):
    """Registra o nó de trabalho com o servidor mestre."""
    registration_url = f"http://{master_ip}:{master_port}/register_node"
    
    # Prepara dados para registro
    data = {
        "node_url": worker_url,
        "system_info": system_info,
        "token": token,
        "node_name": node_name or socket.gethostname(),
    }
    
    try:
        logger.info(f"Registrando com o servidor mestre em {registration_url}...")
        response = requests.post(registration_url, json=data, timeout=30)
        if response.status_code == 200:
            logger.info("Registro com o servidor mestre bem-sucedido")
            return response.json()
        else:
            logger.error(f"Falha no registro: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"Erro ao conectar ao servidor mestre: {e}")
        return None

def setup_worker_directory(working_dir):
    """Configura o diretório de trabalho para o worker."""
    work_dir = Path(working_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar diretórios necessários
    data_dir = work_dir / "data"
    logs_dir = work_dir / "logs"
    models_dir = work_dir / "models"
    
    for directory in [data_dir, logs_dir, models_dir]:
        directory.mkdir(exist_ok=True)
        
    logger.info(f"Diretórios de trabalho configurados em {work_dir}")
    return str(work_dir)

def clone_repository(working_dir, repo_url="https://github.com/relex-ai/distributed-training.git"):
    """Clona ou atualiza o repositório do projeto."""
    repo_dir = Path(working_dir) / "repo"
    
    if repo_dir.exists():
        # Atualizar repositório existente
        logger.info(f"Atualizando repositório em {repo_dir}")
        try:
            subprocess.check_call(["git", "pull"], cwd=str(repo_dir))
        except subprocess.SubprocessError as e:
            logger.error(f"Erro ao atualizar repositório: {e}")
            return False
    else:
        # Clonar novo repositório
        logger.info(f"Clonando repositório de {repo_url} para {repo_dir}")
        try:
            subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
        except subprocess.SubprocessError as e:
            logger.error(f"Erro ao clonar repositório: {e}")
            return False
    
    logger.info("Repositório atualizado com sucesso")
    return True

def create_worker_config(working_dir, master_ip, master_port, worker_port, node_name):
    """Cria arquivo de configuração para o worker."""
    config = {
        "master_address": f"{master_ip}:{master_port}",
        "worker_port": worker_port,
        "working_directory": working_dir,
        "node_name": node_name or socket.gethostname(),
        "log_level": "INFO",
    }
    
    config_path = Path(working_dir) / "worker_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Arquivo de configuração do worker criado em {config_path}")
    return str(config_path)

def main():
    """Função principal para orquestrar o processo de configuração."""
    args = parse_args()
    
    logger.info("Iniciando configuração do ambiente Kaggle para o sistema Relex.AI")
    
    # Verifica ambiente Kaggle
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is None:
        logger.warning("Este script foi projetado para execução em ambientes Kaggle.")
        response = input("Continuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            logger.info("Configuração cancelada pelo usuário")
            return
    
    # Instalar dependências
    install_dependencies()
    
    # Configurar diretório de trabalho
    working_dir = setup_worker_directory(args.working_dir)
    
    # Coletar informações do sistema
    system_info = get_system_info()
    
    # Configurar ngrok
    worker_url = None
    if args.ngrok_token:
        worker_url = setup_ngrok(args.worker_port, args.ngrok_token)
    
    if not worker_url:
        worker_ip = socket.gethostbyname(socket.gethostname())
        worker_url = f"http://{worker_ip}:{args.worker_port}"
        logger.warning(f"Usando URL local para o worker: {worker_url}")
        logger.warning("Este endereço pode não ser acessível pelo servidor mestre!")
    
    # Registrar com o servidor mestre
    registration_result = register_with_master(
        args.master_ip, args.master_port, worker_url, 
        system_info, args.token, args.node_name
    )
    
    if not registration_result:
        logger.error("Falha ao registrar com o servidor mestre. Verifique a conexão e tente novamente.")
        return
    
    # Clonar repositório do projeto
    clone_repository(working_dir)
    
    # Criar configuração do worker
    config_path = create_worker_config(
        working_dir, args.master_ip, args.master_port, 
        args.worker_port, args.node_name
    )
    
    logger.info("Configuração concluída. Iniciando o worker...")
    
    # Executar o worker
    worker_script = Path(working_dir) / "repo" / "worker" / "run_worker.py"
    if worker_script.exists():
        subprocess.Popen([sys.executable, str(worker_script), "--config", config_path])
        logger.info(f"Worker iniciado com configuração {config_path}")
    else:
        logger.error(f"Script do worker não encontrado em {worker_script}")
        logger.info("Verifique o repositório e tente novamente.")

if __name__ == "__main__":
    main() 