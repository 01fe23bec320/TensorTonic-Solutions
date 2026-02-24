import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    
    predictions : array-like (K,) - predicted probabilities (sum to 1)
    target      : int - correct class index
    epsilon     : float - smoothing factor (0 <= epsilon < 1)
    
    Returns: float
    """

    # Convert to numpy array
    predictions = np.asarray(predictions, dtype=np.float64)

    # Number of classes
    K = predictions.shape[0]

    # Validate target
    if target < 0 or target >= K:
        raise ValueError("Target index out of range")

    # Numerical stability
    eps = 1e-12
    predictions = np.clip(predictions, eps, 1.0)

    # Normalize predictions (if not already normalized)
    predictions = predictions / np.sum(predictions)

    # Build smoothed target distribution
    q = np.full(K, epsilon / K)
    q[target] = (1 - epsilon) + (epsilon / K)

    # Compute cross-entropy loss
    loss = -np.sum(q * np.log(predictions))

    return float(loss)