import torch

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """
    q # (b, qh, s, dim)
    k # (b, kh, s, dim)
    g = q.shape[1] // k.shape[1]
    query = q.unflatten(dim=1, sizes=(-1, g)) # (b, qh, g, s, dim)
    att = query @ k.unsqueeze(2).transpose(-2, -1) # (b, qh, g, s, s)
    seq_len = att.shape[-1]
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = torch.where(mask.bool(), 0, float("-inf")).to(q.dtype)
    att = torch.softmax((att / (k.shape[-1] ** 0.5)) + mask, dim=-1)
    res = (att @ v.unsqueeze(2)).flatten(start_dim=1, end_dim=2)
    return res
