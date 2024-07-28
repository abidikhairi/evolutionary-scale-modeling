import torch


def create_3d_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask has shape (bs, seq_len)
    _, seq_len = attention_mask.shape
    attention_mask = attention_mask.float()

    extended_attention_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1)
    extended_attention_mask = (extended_attention_mask == 0).bool()
    
    return extended_attention_mask
    # extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min
    
    # return extended_attention_mask
    

def create_3d_attention_mask_from_scratch(batch_size: int, seq_len: int) -> torch.Tensor:
    attention_mask = torch.zeros((seq_len, seq_len), dtype=torch.float)
    
    attention_mask = attention_mask.repeat(batch_size, 1, 1)
    import pdb; pdb.set_trace()
    
    return attention_mask
    