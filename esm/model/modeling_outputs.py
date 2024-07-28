from dataclasses import dataclass
from typing import List, OrderedDict

import torch


@dataclass
class Esm1Outputs(OrderedDict):
    loss: float = None
    hidden_states: List[torch.Tensor] = None
    logits: torch.Tensor = None
