from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class InverseRootSquareScheduler(_LRScheduler):
    """
    The models follow a warm-up period of 16000 updates, during which the learning rate increases linearly.
    Afterwards, the learning rate follows an inverse square root decay schedule.
    """
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, warmup_steps: int = 16000) -> None:
        self.warmup_steps = warmup_steps
     
        super().__init__(optimizer, last_epoch)
        
    
    def get_lr(self) -> float:
        step = max(1, self.last_epoch)
        
        if step <= self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Inverse square root decay
            return [base_lr * (self.warmup_steps ** 0.5) / (step ** 0.5) for base_lr in self.base_lrs]
