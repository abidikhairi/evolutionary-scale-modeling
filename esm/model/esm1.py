from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

from esm.model.modeling_outputs import Esm1Outputs
from esm.modules import (
    LanguageModelHead,
    TransformerLayer,
    SinusoidalPositionalEmbedding as PositionEmbedding,
    Embedding as TokenEmbedding
)

from esm.mask_utils import (
    create_3d_attention_mask,
    create_3d_attention_mask_from_scratch
)


class Esm1Transformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int,
                 intermediate_size: int,
                 num_heads: int,
                 num_layers: int,
                 pad_token_id: int,
                 add_bias_qkv: bool= True,
                 attention_dropout_porbs: bool = 0.2, 
                 feedforward_dropout_porbs: bool = 0.2,
                 use_rotary_embeddings: bool = True,
                 max_positions: int = 1024,
                 **kwargs
                ) -> None:
        super().__init__()
        
        self.embed_tokens = TokenEmbedding(
            vocab_size=vocab_size, hidden_size=hidden_size, pad_token_id=pad_token_id
        )
        
        self.embed_positions = PositionEmbedding(
            num_positions=max_positions, hidden_size=hidden_size
        )
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size=hidden_size,
                             intermediate_size=intermediate_size,
                             num_heads=num_heads,
                             add_bias_qkv=add_bias_qkv,
                             attention_dropout_porbs=attention_dropout_porbs,
                             feedforward_dropout_porbs=feedforward_dropout_porbs,
                             use_rotary_embeddings=use_rotary_embeddings
                            )
            for _ in range(num_layers)
        ])
        
        self.lm_head = LanguageModelHead(hidden_size=hidden_size, vocab_size=vocab_size)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs
        ) -> Esm1Outputs:

        bs, seq_len = input_ids.shape
        
        hidden_states = self.embed_tokens(input_ids)
        
        hidden_states = hidden_states + self.embed_positions(bs, seq_len)
        
        all_hidden_states = []
        
        if attention_mask is not None:
            assert attention_mask.ndim == 2, f"attention_mask should have shape (bs, seq_len) got {attention_mask.shape}"
            attention_mask = create_3d_attention_mask(attention_mask)
        else:
            attention_mask = create_3d_attention_mask_from_scratch(batch_size=bs, seq_len=seq_len)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)
            
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(bs * seq_len, -1), labels.view(-1))
    
        return Esm1Outputs(
            loss=loss,
            hidden_states=all_hidden_states,
            logits=logits
        )
 
        