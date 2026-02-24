import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    Works for scalar, list, or numpy array.
    """
    x = np.array(x)  # Convert input to numpy array
    return 1 / (1 + np.exp(-x))


# -------------------------
# Example Usage
# -------------------------

# Scalar
print("Scalar:", sigmoid(2))

# List
print("List:", sigmoid([1, 2, 3]))

# NumPy array
arr = np.array([-1, 0, 1])
print("Array:", sigmoid(arr))

# Matrix
matrix = np.array([[1, -1], [2, -2]])
print("Matrix:\n", sigmoid(matrix))