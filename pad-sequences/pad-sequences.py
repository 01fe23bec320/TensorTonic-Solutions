import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if not seqs:
        return np.array([])

    # Determine max length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    # Create padded array filled with pad_value
    padded = np.full((len(seqs), max_len), pad_value, dtype=int)

    # Copy sequences (right padding)
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]

    return padded