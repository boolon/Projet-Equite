def gradient_descent(X, gradient,N = 20, lr = 1, alpha = 0.8):
    """
    This is the simplest gradient descent we could hope for. They are better
    methods out there, especially if we know about convexity, but it is enough
    to get an idea about performance
    """
    for _ in range(N):
        X = X - lr * gradient(X)
        lr *= alpha
    return X
