#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para configurar o ambiente do Google Colab como nó worker
para o sistema distribuído de treinamento Relex.AI.
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
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Processa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Configuração do Colab para treinamento distribuído")
    parser.add_argument("--master-ip", type=str, required=True, help="IP do servidor master")
    parser.add_argument("--master-port", type=int, default=12345, help="Porta do servidor master")
    parser.add_argument("--worker-port", type=int, default=12346, help="Porta para o nó worker")
    parser.add_argument("--working-dir", type=str, default="/content/relex-ai", help="Diretório de trabalho")
    parser.add_argument("--token", type=str, help="Token de autenticação para conectar ao master")
    parser.add_argument("--storage-limit", type=int, default=20, help="Limite de armazenamento em GB")
    parser.add_argument("--ngrok-token", type=str, help="Token do ngrok para criar túnel")
    parser.add_argument("--node-name", type=str, help="Nome personalizado para o nó")
    
    return parser.parse_args()

def check_gpu():
    """Verifica se GPU está disponível no Colab."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU disponível: {gpu_name}")
            return True, gpu_name
        else:
            logger.warning("GPU não disponível. O treinamento será mais lento.")
            return False, None
    except ImportError:
        logger.warning("PyTorch não instalado. Não foi possível verificar GPU.")
        return False, None

def install_dependencies():
    """Instala as dependências necessárias no Colab."""
    dependencies = [
        "torch>=1.9.0",
        "transformers>=4.15.0",
        "datasets>=1.18.0",
        "tokenizers>=0.10.3",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "psutil>=5.9.0",
        "py-cpuinfo>=8.0.0",
        "ray>=1.9.0",
        "pyzmq>=22.3.0",
        "grpcio>=1.42.0",
        "protobuf>=3.19.0",
        "pyngrok>=5.1.0"
    ]
    
    logger.info("Instalando dependências...")
    
    for dep in dependencies:
        try:
            logger.info(f"Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro ao instalar {dep}: {e}")
    
    logger.info("Dependências instaladas com sucesso.")

def setup_ngrok(ngrok_token, port):
    """Configura o ngrok para criar um túnel para o worker."""
    if not ngrok_token:
        logger.warning("Token do ngrok não fornecido. Pulando configuração de túnel.")
        return None
    
    try:
        from pyngrok import ngrok, conf
        
        # Configura autenticação do ngrok
        conf.get_default().auth_token = ngrok_token
        
        # Encerra túneis existentes
        ngrok.kill()
        
        # Cria novo túnel
        logger.info(f"Criando túnel ngrok para a porta {port}...")
        tunnel = ngrok.connect(port, "tcp")
        public_url = tunnel.public_url
        
        # Extrai o endereço público
        # Formato: tcp://X.tcp.ngrok.io:YYYYY
        parts = public_url.replace("tcp://", "").split(":")
        host = parts[0]
        port = int(parts[1])
        
        logger.info(f"Túnel ngrok criado: {host}:{port}")
        return (host, port)
    
    except Exception as e:
        logger.error(f"Erro ao configurar ngrok: {e}")
        return None

def get_system_info():
    """Coleta informações do sistema."""
    import psutil
    import platform
    
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info.get('brand_raw', 'Desconhecido')
    except:
        cpu_name = platform.processor() or 'Desconhecido'
    
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": cpu_name,
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
    }
    
    # Adiciona informações de GPU se disponível
    has_gpu, gpu_name = check_gpu()
    if has_gpu:
        info["gpu"] = gpu_name
        
        # Adiciona informações adicionais de GPU se o torch estiver disponível
        try:
            import torch
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory"] = []
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu_memory"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2)
                })
        except:
            pass
    
    return info

def register_with_master(master_ip, master_port, node_info, token=None):
    """Registra o nó worker com o servidor master."""
    url = f"http://{master_ip}:{master_port}/register_node"
    headers = {"Content-Type": "application/json"}
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        logger.info(f"Tentando registrar com o master em {url}...")
        response = requests.post(url, json=node_info, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Nó registrado com sucesso. ID atribuído: {result.get('node_id')}")
            return result
        else:
            logger.error(f"Falha ao registrar nó. Código: {response.status_code}, Erro: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Erro ao comunicar com o master: {e}")
        return None

def setup_worker_directory(working_dir):
    """Configura o diretório de trabalho do worker."""
    # Cria diretório de trabalho
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Cria subdiretórios
    (working_dir / "data").mkdir(exist_ok=True)
    (working_dir / "models").mkdir(exist_ok=True)
    (working_dir / "logs").mkdir(exist_ok=True)
    (working_dir / "scripts").mkdir(exist_ok=True)
    
    return working_dir

def clone_repository(working_dir, repo_url="https://github.com/user/relex-ai.git", branch="main"):
    """Clona o repositório do projeto."""
    try:
        if not (working_dir / ".git").exists():
            logger.info(f"Clonando repositório {repo_url}, branch {branch}...")
            subprocess.run(["git", "clone", "-b", branch, repo_url, str(working_dir)], check=True)
        else:
            logger.info("Atualizando repositório...")
            subprocess.run(["git", "-C", str(working_dir), "pull"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao clonar repositório: {e}")
        return False

def create_worker_config(working_dir, args, node_info, node_id):
    """Cria arquivo de configuração do worker."""
    config = {
        "node_id": node_id,
        "worker": {
            "master_ip": args.master_ip,
            "master_port": args.master_port,
            "worker_port": args.worker_port,
            "working_dir": str(working_dir),
            "storage_limit_gb": args.storage_limit
        },
        "system": node_info
    }
    
    config_file = working_dir / "worker_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuração do worker salva em {config_file}")
    return config_file

def main():
    """Função principal."""
    args = parse_args()
    
    logger.info("Iniciando configuração do Colab como nó worker...")
    
    # Instala dependências
    install_dependencies()
    
    # Configura diretório de trabalho
    working_dir = setup_worker_directory(args.working_dir)
    logger.info(f"Diretório de trabalho configurado: {working_dir}")
    
    # Coleta informações do sistema
    system_info = get_system_info()
    logger.info("Informações do sistema coletadas")
    
    # Configura túnel ngrok se token fornecido
    tunnel_info = None
    if args.ngrok_token:
        tunnel_info = setup_ngrok(args.ngrok_token, args.worker_port)
    
    # Prepara informações do nó
    node_info = {
        "name": args.node_name or f"colab-{socket.gethostname()}",
        "host": tunnel_info[0] if tunnel_info else socket.gethostname(),
        "port": tunnel_info[1] if tunnel_info else args.worker_port,
        "system_info": system_info,
        "storage_limit_gb": args.storage_limit,
        "platform": "colab"
    }
    
    # Registra com o master
    registration = register_with_master(args.master_ip, args.master_port, node_info, args.token)
    
    if registration:
        node_id = registration.get("node_id")
        
        # Cria configuração do worker
        config_file = create_worker_config(working_dir, args, system_info, node_id)
        
        # Inicia o worker
        logger.info("Iniciando o worker...")
        worker_script = working_dir / "src" / "main.py"
        
        if worker_script.exists():
            cmd = [
                sys.executable, str(worker_script),
                "--mode", "worker",
                "--master-ip", args.master_ip,
                "--config", str(config_file)
            ]
            
            try:
                # Executa o worker em modo bloqueante
                logger.info(f"Executando: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except KeyboardInterrupt:
                logger.info("Worker interrompido pelo usuário")
            except subprocess.CalledProcessError as e:
                logger.error(f"Erro ao executar o worker: {e}")
        else:
            logger.error(f"Script do worker não encontrado: {worker_script}")
    else:
        logger.error("Falha ao registrar com o master. Encerrando.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Configuração interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro durante a configuração: {e}", exc_info=True) 