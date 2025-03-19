# Scripts de Configuração para Treinamento Distribuído Relex.AI

Este repositório contém scripts para configurar ambientes externos como nós de trabalho no sistema de treinamento distribuído Relex.AI.

## Visão Geral

O Relex.AI utiliza uma arquitetura de treinamento distribuído que consiste em um servidor mestre e vários nós de trabalho. Estes scripts facilitam a configuração de plataformas como Google Colab e Kaggle como nós de trabalho, permitindo o aproveitamento dos recursos computacionais disponíveis nessas plataformas.

## Scripts Disponíveis

### 1. Setup Google Colab (`scripts/setup_colab.py`)

Configura um ambiente Google Colab como nó de trabalho para o treinamento distribuído.

### 2. Setup Kaggle (`scripts/setup_kaggle.py`)

Configura um ambiente Kaggle como nó de trabalho para o treinamento distribuído.

## Requisitos

- Python 3.6 ou superior
- Conexão com a internet
- Conta no Google Colab ou Kaggle

## Uso no Google Colab

1. Abra um novo notebook no Google Colab
2. Faça upload do script `setup_colab.py` ou clone este repositório
3. Execute o script com os parâmetros necessários:

```python
!python setup_colab.py --master-ip <IP_DO_SERVIDOR_MESTRE> --master-port <PORTA_DO_MESTRE> --token <SEU_TOKEN> --ngrok-token <TOKEN_NGROK>
```

## Uso no Kaggle

1. Crie um novo notebook no Kaggle
2. Faça upload do script `setup_kaggle.py` ou clone este repositório
3. Execute o script com os parâmetros necessários:

```python
!python setup_kaggle.py --master-ip <IP_DO_SERVIDOR_MESTRE> --master-port <PORTA_DO_MESTRE> --token <SEU_TOKEN> --ngrok-token <TOKEN_NGROK>
```

## Parâmetros

Os scripts aceitam os seguintes parâmetros:

- `--master-ip`: IP do servidor mestre (obrigatório)
- `--master-port`: Porta do servidor mestre (padrão: 5000)
- `--worker-port`: Porta local do nó de trabalho (padrão: 5001)
- `--working-dir`: Diretório de trabalho para arquivos do projeto
- `--token`: Token de autenticação para registro com o servidor mestre (obrigatório)
- `--storage-limit`: Limite de uso de armazenamento (0.0-1.0, padrão: 0.8)
- `--ngrok-token`: Token de API do ngrok para criar um túnel
- `--node-name`: Nome personalizado para o nó (opcional)

## Detalhes de Funcionamento

Os scripts realizam as seguintes operações:

1. Verificam a disponibilidade de GPU
2. Instalam as dependências necessárias
3. Configuram um túnel ngrok para comunicação externa
4. Coletam informações do sistema
5. Registram o nó com o servidor mestre
6. Clonam o repositório do projeto
7. Configuram o ambiente de trabalho
8. Iniciam o processo de worker

## Solução de Problemas

- **Erro de conexão com o servidor mestre**: Verifique se o IP e a porta estão corretos
- **Problemas com ngrok**: Certifique-se de que o token ngrok é válido
- **Falha na instalação de dependências**: Verifique a conexão com a internet

## Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue para discutir mudanças propostas ou envie um pull request.

## Licença

Este projeto está sob a licença MIT. 