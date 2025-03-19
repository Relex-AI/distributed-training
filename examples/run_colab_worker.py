#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de exemplo para configurar um ambiente Google Colab como nó de trabalho.
Este script pode ser copiado e executado em um notebook Colab.
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    """Processa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Iniciar nó de trabalho no Google Colab')
    parser.add_argument('--master-ip', type=str, default='', help='IP do servidor mestre')
    parser.add_argument('--master-port', type=int, default=5000, help='Porta do servidor mestre')
    parser.add_argument('--token', type=str, default='', help='Token de autenticação')
    parser.add_argument('--ngrok-token', type=str, default='', help='Token API do ngrok para criar um túnel')
    parser.add_argument('--node-name', type=str, default='colab-worker', help='Nome do nó')
    
    return parser.parse_args()

def check_environment():
    """Verifica o ambiente Google Colab e exibe informações sobre recursos disponíveis."""
    print("Verificando ambiente Google Colab...")
    
    # Verificar se estamos no Colab
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False
    
    if not is_colab:
        print("AVISO: Este script não parece estar sendo executado no Google Colab.")
    
    # Verificar GPU
    try:
        gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
        print("Informações da GPU:")
        print(gpu_info)
    except:
        print("GPU não disponível ou erro ao executar nvidia-smi")
    
    # Verificar CPU
    try:
        cpu_info = subprocess.check_output('cat /proc/cpuinfo | grep "model name" | head -1', shell=True).decode('utf-8')
        print("Informações da CPU:")
        print(cpu_info)
    except:
        print("Erro ao acessar informações da CPU")
    
    # Verificar memória
    try:
        memory_info = subprocess.check_output('free -h', shell=True).decode('utf-8')
        print("Informações de memória:")
        print(memory_info)
    except:
        print("Erro ao acessar informações de memória")
    
    # Verificar espaço em disco
    try:
        disk_info = subprocess.check_output('df -h /content', shell=True).decode('utf-8')
        print("Informações de armazenamento:")
        print(disk_info)
    except:
        print("Erro ao acessar informações de armazenamento")

def download_setup_script():
    """Baixa o script de configuração para o ambiente Colab."""
    print("Baixando script de configuração...")
    
    # Criar diretório scripts se não existir
    os.makedirs('scripts', exist_ok=True)
    
    # URL para o script de configuração
    script_url = "https://raw.githubusercontent.com/relex-ai/distributed-training/main/scripts/setup_colab.py"
    
    # Baixar script
    try:
        subprocess.check_call(['wget', '-O', 'scripts/setup_colab.py', script_url])
        print("Script de configuração baixado com sucesso!")
    except:
        print("Erro ao baixar script de configuração.")
        print("Tentando usar curl como alternativa...")
        try:
            subprocess.check_call(['curl', '-o', 'scripts/setup_colab.py', script_url])
            print("Script de configuração baixado com sucesso usando curl!")
        except:
            print("Falha ao baixar script de configuração. Verifique sua conexão.")
            return False
    
    # Verificar se o download foi bem-sucedido
    if os.path.exists('scripts/setup_colab.py'):
        print(f"Script salvo em: {os.path.abspath('scripts/setup_colab.py')}")
        return True
    else:
        print("Erro: O script não foi baixado corretamente.")
        return False

def run_setup_script(args):
    """Executa o script de configuração com os parâmetros fornecidos."""
    if not os.path.exists('scripts/setup_colab.py'):
        print("Erro: Script de configuração não encontrado.")
        return False
    
    # Construir comando para executar o script
    cmd = [
        sys.executable, 'scripts/setup_colab.py',
        '--master-ip', args.master_ip,
        '--master-port', str(args.master_port),
        '--node-name', args.node_name
    ]
    
    # Adicionar token se fornecido
    if args.token:
        cmd.extend(['--token', args.token])
    else:
        print("AVISO: Token de autenticação não fornecido. O registro com o servidor mestre irá falhar.")
        print("Forneça um token usando o argumento --token")
    
    # Adicionar token ngrok se fornecido
    if args.ngrok_token:
        cmd.extend(['--ngrok-token', args.ngrok_token])
    else:
        print("AVISO: Token ngrok não fornecido. A comunicação externa pode ser limitada.")
        print("Para obter um token gratuito, visite: https://dashboard.ngrok.com/get-started/your-authtoken")
    
    # Executar script
    print("Iniciando configuração do nó de trabalho...")
    print(f"Comando: {' '.join(cmd)}")
    
    try:
        subprocess.Popen(cmd)
        print("Script de configuração iniciado com sucesso!")
        return True
    except Exception as e:
        print(f"Erro ao executar script de configuração: {e}")
        return False

def setup_keep_alive():
    """Configura um script para manter o Colab ativo."""
    print("Configurando script para manter o Colab ativo...")
    
    keep_alive_code = """
import time
import IPython.display
from IPython.display import HTML, display

def keep_alive():
    display(HTML('''
    <script>
        function ClickConnect(){
            console.log("Mantendo conexão ativa");
            document.querySelector("colab-connect-button").click()
        }
        setInterval(ClickConnect, 60000)
    </script>
    ''')
    )
    
    while True:
        time.sleep(60)
        IPython.display.clear_output(wait=True)
        print(f"Sessão Colab mantida ativa - Último ping: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Para manter o Colab ativo, execute:
# keep_alive()
    """
    
    with open('keep_alive.py', 'w') as f:
        f.write(keep_alive_code)
    
    print("Script keep_alive.py criado. Para usá-lo no notebook Colab, execute:")
    print("from keep_alive import keep_alive")
    print("keep_alive()")

def main():
    """Função principal."""
    print("=== Inicializador de Nó de Trabalho Google Colab para Relex.AI ===")
    
    # Verificar ambiente
    check_environment()
    
    # Baixar script
    if not download_setup_script():
        print("Erro ao baixar script de configuração. Abortando.")
        return
    
    # Processar argumentos
    args = parse_args()
    
    # Solicitar argumentos faltantes interativamente
    if not args.master_ip:
        args.master_ip = input("Digite o IP do servidor mestre: ")
    
    if not args.token:
        args.token = input("Digite o token de autenticação: ")
    
    if not args.ngrok_token:
        args.ngrok_token = input("Digite o token ngrok (opcional, pressione Enter para pular): ")
    
    # Executar script de configuração
    success = run_setup_script(args)
    
    if success:
        # Configurar script para manter o Colab ativo
        setup_keep_alive()
        
        print("\nSe o script de configuração foi iniciado com sucesso, este ambiente Colab")
        print("agora deve estar conectado ao sistema de treinamento distribuído Relex.AI como um nó de trabalho.")
        print("\nPara verificar o status, você pode executar:")
        print("  !ps aux | grep setup_colab")
        print("  !tail -f /content/relex/logs/worker.log")
        print("\nIMPORTANTE: Para evitar que o Colab desconecte por inatividade, execute o script keep_alive:")
        print("  from keep_alive import keep_alive")
        print("  keep_alive()")
        print("\nMantenha este notebook em execução enquanto desejar que este ambiente")
        print("participe do treinamento distribuído.")

if __name__ == "__main__":
    main() 