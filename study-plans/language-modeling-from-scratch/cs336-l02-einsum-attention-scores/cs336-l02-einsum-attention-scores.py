import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    q # (b, sq, d)
    k # (b, sk, d)
    query = q.unflatten(2, sizes=(num_heads, -1)).transpose(1, 2) # (b, h, sq, d)
    key = k.unflatten(2, sizes=(num_heads, -1)).transpose(1, 2) # (b, h, sk, d)
    return query @ key.transpose(-1, -2) / query.shape[-1] ** 0.5
