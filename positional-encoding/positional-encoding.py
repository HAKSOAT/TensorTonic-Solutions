import math
import torch

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    enc = torch.arange(seq_len, dtype=torch.float64).repeat(d_model, 1)
    row_idx = torch.arange((d_model//2)+1, dtype=torch.float64).unsqueeze(-1)\
        .repeat_interleave(2, dim=0)\
        .repeat(1, seq_len)
    row_idx = row_idx[:d_model]
    v = enc / (base ** (2 * row_idx / d_model))
    v[0::2] = torch.sin(v[0::2])
    v[1::2] = torch.cos(v[1::2])
    
    return v.T.numpy()