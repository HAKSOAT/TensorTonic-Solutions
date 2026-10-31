import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    scores = np.array(scores)
    mask = np.tril(np.ones_like(scores, dtype=bool))
    result = np.where(mask, scores, mask_value)
    return result