def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    derivative = lambda a, b, x: 2*a*x + b
    # Write code here
    while steps:
        x0 = x0 - lr * derivative(a, b, x0)
        steps -= 1

    return x0

