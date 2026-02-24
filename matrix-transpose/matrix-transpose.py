import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    
    rows, cols = A.shape
    
    # Create empty matrix with swapped shape
    transpose = np.zeros((cols, rows), dtype=A.dtype)
    
    # Swap indices
    for i in range(rows):
        for j in range(cols):
            transpose[j, i] = A[i, j]
    
    return transpose
