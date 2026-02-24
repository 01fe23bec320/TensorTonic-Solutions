def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Perform vanilla gradient descent on f(x) = ax^2 + bx + c
    and return final x after 'steps' iterations.
    """
    
    x = x0
    
    for _ in range(steps):
        grad = 2 * a * x + b   # derivative
        x = x - lr * grad      # update rule
    
    return x