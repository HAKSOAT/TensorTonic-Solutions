import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    query # (B, H_q, D)
    key_cache, value_cache # (B, H_kv, S, D)
    new_key, new_value # (B, H_kv, D)
    
    key_cache = torch.cat([key_cache, new_key.unsqueeze(2)], dim=2)
    value_cache = torch.cat([value_cache, new_value.unsqueeze(2)], dim=2)
    dim = query.shape[-1]
    assert query.shape[1] % key_cache.shape[1] == 0
    group = query.shape[1] // key_cache.shape[1]
    # Group the query heads to match available key heads and value heads
    query = query.unflatten(1, (-1, group))
    # Calculate attention weights using the grouped queries and corresponding key
    att_weights = query @ key_cache.transpose(2, 3) / (dim ** 0.5)
    softmaxed_weights = torch.softmax(att_weights, dim=-1) # (B, H_q, group, S)
    # Project back into required shape using the attention weights and the corresponding value
    output = (softmaxed_weights @ value_cache) # (B, H_q, group, S)

    return {
        "output": output.flatten(1, 2),
        "new_key_cache": key_cache,
        "new_value_cache": value_cache
    }