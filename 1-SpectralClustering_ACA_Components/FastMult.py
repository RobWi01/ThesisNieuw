import numpy as np

def fastMult(W_T, deltas, v):
    """
    Compute the product of G_k and vector v efficiently.

    Args:
    - W_T: Transpose of matrix W (ACA output for similarity matrix).
    - deltas: List of scalars (delta_i).
    - v: The vector to be multiplied with G_k.

    Returns:
    - result: The result of G_k * v.
    """
    result = np.zeros(v.shape) 
    for i in range(len(deltas)):
        W_T_i = W_T[:, i].reshape(-1, 1)
        result += (np.dot(W_T_i.T, v) * W_T_i) / deltas[i]
        # print(f'Iteration {i} adds {(np.dot(W_T_i.T, v) * W_T_i) / deltas[i]}')
    return result
