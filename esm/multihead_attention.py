import math
from typing import Optional
import torch
from torch import nn

from esm.rotary_embedding import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True, use_rotary_embeddings: bool = False, attention_dropout_probs: float = 0.2) -> None:
        super().__init__()

        assert hidden_size % num_heads == 0, f"hidden_size must be divisible by num_heads, got hidden_size={hidden_size} and num_heads={num_heads}"
        
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.scaling = math.sqrt(hidden_size)
        self.attention_dropout_probs = attention_dropout_probs
        
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=bias)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=bias)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=bias)
        
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_embed = None
        
        if use_rotary_embeddings:
            self.rotary_embed = RotaryEmbedding(self.hidden_size)
        
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, seq_len, _ = hidden_states.shape
        
        query_states = hidden_states.view(bs, seq_len, self.num_heads, self.head_dim)
        key_states = hidden_states.view(bs, seq_len, self.num_heads, self.head_dim)
        value_states = hidden_states.view(bs, seq_len, self.num_heads, self.head_dim)
        
        query_states = self.query(query_states)
        key_states = self.key(key_states)
        value_states = self.value(value_states)
        
        query_states = query_states.view(bs, seq_len, -1)
        key_states = key_states.view(bs, seq_len, -1)
        value_states = value_states.view(bs, seq_len, -1)
        
        if self.rotary_embed is not None:
            query_states, key_states = self.rotary_embed(query_states, key_states)
        
        
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / self.scaling
        scores = torch.softmax(scores, dim=-1)

        if attention_mask is not None:         
            scores = scores.masked_fill(attention_mask, 1e-22)
        
        scores = torch.dropout(scores, p=self.attention_dropout_probs, train=self.training)
                
        hidden_states = torch.matmul(scores, value_states)
        
        return self.output(hidden_states)
