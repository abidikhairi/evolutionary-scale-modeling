from typing import Any, Optional
from enum import Enum
import torch
from torch import optim
import pytorch_lightning as pl

from esm.model import Esm1Transformer
from esm.schedulers import InverseRootSquareScheduler



class ModelNames(Enum):
    esm1 = Esm1Transformer


class TransformerTrainer(pl.LightningModule):
    def __init__(self, model_name: ModelNames,
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
                max_positions: int = 256
    ) -> None:
        super().__init__()
        
        clazz = model_name.value
        self.model: torch.nn.Module = clazz(vocab_size=vocab_size,
                           hidden_size=hidden_size,
                           intermediate_size=intermediate_size,
                           num_heads=num_heads,
                           num_layers=num_layers,
                           pad_token_id=pad_token_id,
                           add_bias_qkv=add_bias_qkv,
                           attention_dropout_porbs=attention_dropout_porbs,
                           feedforward_dropout_porbs=feedforward_dropout_porbs,
                           use_rotary_embeddings=use_rotary_embeddings,
                           max_positions=max_positions
                        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        lr_scheduler = InverseRootSquareScheduler(optimizer=optimizer, last_epoch=-1, warmup_steps=2000)
        
        return [optimizer], [lr_scheduler] 
    
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                ) -> Any:
        
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)        
        
        self.log("train/loss", outputs.loss)
        
        return outputs.loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        
        self.log("valid/loss", outputs.loss)
        
        return {
            'valid/loss': outputs.loss 
        }
    
    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        
        self.log("test/loss", outputs.loss)
        
        return {
            'test/loss': outputs.loss 
        }
