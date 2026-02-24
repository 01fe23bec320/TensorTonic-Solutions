import numpy as np

def binary_focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    """
    Compute mean Binary Focal Loss.
    
    predictions : array-like (N,) - predicted probabilities (0 < p < 1)
    targets     : array-like (N,) - binary labels {0,1}
    alpha       : balancing factor
    gamma       : focusing parameter
    
    Returns: float (mean focal loss)
    """

    # Convert to numpy arrays
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # Check same shape
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have same shape")

    # Numerical stability (avoid log(0))
    eps = 1e-12
    predictions = np.clip(predictions, eps, 1 - eps)

    # Compute p_t
    p_t = np.where(targets == 1, predictions, 1 - predictions)

    # Compute focal loss for each sample
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)

    # Return mean loss
    return np.mean(loss)