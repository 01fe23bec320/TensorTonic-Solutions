import numpy as np

def kl_divergence(P, Q, normalize=True):
    """
    Compute KL Divergence D_KL(P || Q)

    Parameters:
    -----------
    P : array-like
        True probability distribution
    Q : array-like
        Approximated / predicted probability distribution
    normalize : bool
        If True, automatically normalize P and Q to sum to 1

    Returns:
    --------
    float
        KL divergence value
    """

    # Convert to numpy arrays
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Check same shape
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")

    # Normalize if required
    if normalize:
        P = P / np.sum(P)
        Q = Q / np.sum(Q)

    # Numerical stability (avoid log(0))
    eps = 1e-12
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)

    # Compute KL divergence
    kl = np.sum(P * np.log(P / Q))

    return kl