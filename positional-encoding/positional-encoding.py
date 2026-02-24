import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return positional encoding matrix of shape (seq_len, d_model)
    using sinusoidal formulation.
    
    Odd d_model -> last column is sin.
    """
    
    # Position indices (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]
    
    # Dimension indices (1, d_model)
    dims = np.arange(d_model)[np.newaxis, :]
    
    # Compute the angle rates
    angle_rates = 1 / (base ** (2 * (dims // 2) / d_model))
    
    # Compute angle matrix
    angle_rads = positions * angle_rates
    
    # Initialize encoding matrix
    PE = np.zeros((seq_len, d_model))
    
    # Apply sin to even indices (0,2,4,...)
    PE[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices (1,3,5,...)
    PE[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return PE