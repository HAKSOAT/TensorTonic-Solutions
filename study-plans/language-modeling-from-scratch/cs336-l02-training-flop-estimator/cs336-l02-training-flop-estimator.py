def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """
    forward = sum([2 * d[0] * d[1] * d[2] for d in matmuls]) + attention_flops
    return {
        "forward_flops":forward,"backward_flops":2*forward,"total_flops":2*forward+forward
    }
