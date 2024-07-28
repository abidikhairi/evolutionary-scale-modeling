from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from esm.tokenization import EsmTokenizer


ESM1_DEBUG = {
    "hidden_size": 32,
    "intermediate_size": 128,
    "num_heads": 4,
    "num_layers": 4,
    "add_bias_qkv": False,
    "use_rotary_embeddings": True,
    "max_positions": 256
}

ESM1_SMALL = { 
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_heads": 8,
    "num_layers": 6,
    "add_bias_qkv": True,
    "use_rotary_embeddings": True,
    "max_positions": 256
}

ESM1_MEDIUM = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_heads": 8,
    "num_layers": 12,
    "add_bias_qkv": True,
    "use_rotary_embeddings": True,
    "max_positions": 256
}


@dataclass
class ModelsConfig(Enum):
    esm1_debug = ESM1_DEBUG
    esm1_sm = ESM1_SMALL
    esm1_medium = ESM1_MEDIUM
    
    
def load_model_config(variant: ModelsConfig, tokenizer: EsmTokenizer) -> Dict[str, Any]:
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    
    config_dict = variant.value 
    
    config_dict['vocab_size'] = vocab_size
    config_dict['pad_token_id'] = pad_token_id
    
    return config_dict
