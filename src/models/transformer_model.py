#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelo Transformer para treinamento em código e dados multimodais.
Este módulo define a arquitetura do modelo e funções relacionadas ao treinamento.
"""

import os
import logging
import math
from typing import Dict, List, Optional, Union, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        PreTrainedModel, 
        PretrainedConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class RelexConfig(PretrainedConfig):
    """Configuração para o modelo Relex.AI."""
    
    model_type = "relex"
    
    def __init__(
        self,
        vocab_size=50000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id, 
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id, 
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

class RelexSelfAttention(nn.Module):
    """Implementação de Self-Attention para o modelo Relex."""
    
    def __init__(self, config):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"O tamanho oculto ({config.hidden_size}) não é múltiplo do número de "
                f"cabeças de atenção ({config.num_attention_heads})"
            )
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Camadas de projeção
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def transpose_for_scores(self, x):
        """Transpõe a dimensão para o formato esperado pelo cálculo de atenção."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calcula os scores de atenção
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Aplica a máscara de atenção se fornecida
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Normaliza os scores para probabilidades
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Multiplica os pesos de atenção pelos valores
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Restaura o formato original
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Projeção final
        output = self.output(context_layer)
        output = self.proj_dropout(output)
        
        return output, attention_probs

class RelexBlock(nn.Module):
    """Bloco de camada do Transformer para o modelo Relex."""
    
    def __init__(self, config):
        super().__init__()
        
        # Self-Attention
        self.attention = RelexSelfAttention(config)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention
        attention_output, attention_weights = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_layer_norm(hidden_states + attention_output)
        
        # Feed-forward
        intermediate_output = self.intermediate(attention_output)
        output = self.output_layer_norm(attention_output + intermediate_output)
        
        return output, attention_weights

class RelexModel(nn.Module):
    """Modelo de linguagem Transformer para o Relex.AI."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(config.vocab_size, config.hidden_size),
            "position_embeddings": nn.Embedding(config.max_position_embeddings, config.hidden_size),
            "token_type_embeddings": nn.Embedding(config.type_vocab_size, config.hidden_size)
        })
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Camadas do Transformer
        self.layers = nn.ModuleList([RelexBlock(config) for _ in range(config.num_hidden_layers)])
        
        # Registra buffer para posições
        position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.register_buffer("position_ids", position_ids)
        
        # Inicialização dos pesos
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Inicializa os pesos do modelo."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None,
        output_attentions=None
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        
        # Prepara as máscaras de atenção
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            
        # Formata a máscara para o formato esperado
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Prepara IDs de tipo de token
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            
        # Prepara IDs de posição
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        # Embeddings
        word_embeddings = self.embeddings["word_embeddings"](input_ids)
        position_embeddings = self.embeddings["position_embeddings"](position_ids)
        token_type_embeddings = self.embeddings["token_type_embeddings"](token_type_ids)
        
        # Soma todos os embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Processamento nas camadas do Transformer
        hidden_states = embeddings
        all_attentions = [] if output_attentions else None
        
        for layer in self.layers:
            hidden_states, attention_weights = layer(hidden_states, extended_attention_mask)
            if output_attentions:
                all_attentions.append(attention_weights)
        
        return {
            "last_hidden_state": hidden_states,
            "attentions": all_attentions
        }

class RelexForCausalLM(nn.Module):
    """Modelo para geração de linguagem autoregressiva."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.transformer = RelexModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Inicialização dos pesos
        self.apply(self._init_weights)
        
        # Compartilha pesos entre o embedding e a camada de saída
        self.tie_weights()
        
    def _init_weights(self, module):
        """Inicializa os pesos do modelo."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def tie_weights(self):
        """Compartilha pesos entre embedding e camada de saída."""
        self.lm_head.weight = self.transformer.embeddings["word_embeddings"].weight
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        output_attentions=None
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions
        )
        
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Predição de tokens
        lm_logits = self.lm_head(hidden_states)
        
        # Cálculo da perda se labels fornecidos
        loss = None
        if labels is not None:
            # Desloca os labels para corresponder à predição autoregressiva
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calcula a perda
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
            "attentions": transformer_outputs["attentions"]
        }

class RelexTokenizer:
    """Wrapper para tokenizadores do HuggingFace."""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        
    def from_pretrained(self, tokenizer_name_or_path):
        """Carrega um tokenizador pré-treinado."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers não está instalado. Instale com pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        return self
    
    def train_new_tokenizer(self, texts, vocab_size=None, min_frequency=2):
        """Treina um novo tokenizador a partir de textos."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers não está instalado. Instale com pip install transformers")
            
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
        
        # Configuração do tokenizador
        if vocab_size is None:
            vocab_size = self.config.get("vocab_size", 50000)
            
        # Cria um tokenizador BPE
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Configura o treinador
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
        )
        
        # Treina o tokenizador
        tokenizer.train_from_iterator(texts, trainer)
        
        # Adiciona post-processor
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # Converte para o formato HuggingFace
        from transformers import PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>"
        )
        
        return self
    
    def save_pretrained(self, directory):
        """Salva o tokenizador em um diretório."""
        if self.tokenizer is None:
            raise ValueError("Tokenizador não inicializado")
            
        os.makedirs(directory, exist_ok=True)
        self.tokenizer.save_pretrained(directory)
    
    def encode(self, text, **kwargs):
        """Codifica o texto em IDs de tokens."""
        if self.tokenizer is None:
            raise ValueError("Tokenizador não inicializado")
            
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids, **kwargs):
        """Decodifica IDs de tokens para texto."""
        if self.tokenizer is None:
            raise ValueError("Tokenizador não inicializado")
            
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def __call__(self, texts, **kwargs):
        """Tokeniza textos."""
        if self.tokenizer is None:
            raise ValueError("Tokenizador não inicializado")
            
        return self.tokenizer(texts, **kwargs)

def create_model_from_config(config_data):
    """Cria um modelo a partir de um dicionário de configuração."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch não está instalado. Instale com pip install torch")
        
    # Extrai parâmetros do modelo
    model_params = config_data.get("model", {})
    
    # Cria configuração
    config = RelexConfig(
        vocab_size=model_params.get("vocab_size", 50000),
        hidden_size=model_params.get("hidden_size", 768),
        num_hidden_layers=model_params.get("num_hidden_layers", 12),
        num_attention_heads=model_params.get("num_attention_heads", 12),
        intermediate_size=model_params.get("intermediate_size", 3072),
        hidden_dropout_prob=model_params.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=model_params.get("attention_probs_dropout_prob", 0.1),
        max_position_embeddings=model_params.get("max_seq_length", 1024),
        initializer_range=model_params.get("initializer_range", 0.02),
        layer_norm_eps=model_params.get("layer_norm_eps", 1e-12)
    )
    
    # Cria o modelo
    model = RelexForCausalLM(config)
    
    # Cria tokenizador
    tokenizer = RelexTokenizer(config_data.get("tokenizer", {}))
    
    return model, tokenizer

def save_model(model, tokenizer, output_dir):
    """Salva o modelo e o tokenizador em um diretório."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva o modelo
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # Salva a configuração
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        f.write(model.config.to_json_string())
    
    # Salva o tokenizador
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Modelo salvo em {output_dir}")
    
def load_model(model_dir):
    """Carrega um modelo salvo."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch não está instalado. Instale com pip install torch")
        
    # Carrega a configuração
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = RelexConfig.from_json_file(f.read())
    
    # Cria o modelo
    model = RelexForCausalLM(config)
    
    # Carrega pesos
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Carrega tokenizador
    tokenizer = RelexTokenizer(config)
    tokenizer.from_pretrained(model_dir)
    
    return model, tokenizer

class CodeDataset(Dataset):
    """Dataset para treinamento com dados de código."""
    
    def __init__(self, tokenized_data, block_size=1024):
        self.examples = tokenized_data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Processa os dados para treinamento autoregressivo
        input_ids = example["input_ids"]
        
        # Trunca ou faz padding para o tamanho do bloco
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
        else:
            input_ids = input_ids + [0] * (self.block_size - len(input_ids))
        
        attention_mask = example.get("attention_mask", [1] * len(input_ids))
        
        # Converter para tensores
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Para treinamento autoregressivo, os labels são os próprios input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        } 