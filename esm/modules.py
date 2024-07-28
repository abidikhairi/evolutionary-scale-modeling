import math
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import LayerNorm as Esm1bLayerNorm

from esm.multihead_attention import MultiHeadAttention



def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x
    

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, pad_token_id: int) -> None:
        super().__init__()
        
        self.scale_factor = math.sqrt(hidden_size)
        
        self.embed_tokens = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = x * self.scale_factor
        
        return x

# class SinusoidalPositionalEmbedding(nn.Module):
#     def __init__(self, hidden_size: int, padding_idx: int, learned: bool = False):
#         super().__init__()
#         self.embed_dim = hidden_size
#         self.padding_idx = padding_idx
#         self.register_buffer("_float_tensor", torch.FloatTensor(1))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         bsz, seq_len = x.shape
#         max_pos = self.padding_idx + 1 + seq_len
#         if self.weights is None or max_pos > self.weights.size(0):
#             self.weights = self.get_embedding(max_pos)
#         self.weights = self.weights.type_as(self._float_tensor)

#         positions = self.make_positions(x)
#         return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

#     def make_positions(self, x):
#         mask = x.ne(self.padding_idx)
#         range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
#         positions = range_buf.expand_as(x)
#         return positions * mask.long() + self.padding_idx * (1 - mask.long())

#     def get_embedding(self, num_positions):
#         half_dim = self.embed_dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
#         emb = torch.arange(num_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_positions, -1)
#         if self.embed_dim % 2 == 1:
#             # zero pad
#             emb = torch.cat([emb, torch.zeros(num_positions, 1)], dim=1)
#         if self.padding_idx is not None:
#             emb[self.padding_idx, :] = 0
#         return emb

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, hidden_size: int) -> None:
        super().__init__()
        
        pe = torch.zeros(num_positions, hidden_size)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [1, max_len, hidden_size]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, batch_size: int, seq_len: int) -> torch.Tensor:        
        # [1, seq_len, hidden_size]
        pe = self.pe[:, :seq_len, :]
        
        
        # [batch_size, seq_len, hidden_size]
        pe = pe.repeat(batch_size, 1, 1)

        return pe
    
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        feedforward_dropout_probs: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = intermediate_size

        self.activation_fn = gelu

        self.activation_dropout_module = nn.Dropout(
            feedforward_dropout_probs,
        )

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
       
        return x
    

class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        add_bias_qkv: bool = True,
        attention_dropout_porbs: bool = 0.2,
        feedforward_dropout_porbs: bool = 0.2,
        use_rotary_embeddings: bool = True,
    ):
        super().__init__()
    
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        
    
        self.self_attn = MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            bias=add_bias_qkv,
            use_rotary_embeddings=self.use_rotary_embeddings,
            attention_dropout_probs=attention_dropout_porbs
        )
        self.self_attn_layer_norm = ESM1LayerNorm(self.hidden_size)

    
        self.feedforward = FeedForwardNetwork(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            feedforward_dropout_probs=feedforward_dropout_porbs
        )

        self.final_layer_norm = ESM1LayerNorm(self.hidden_size)
        

    def forward(
        self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.feedforward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



class LanguageModelHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(x)
