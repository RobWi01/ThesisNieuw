import sys
import numpy as np
# from FastMult import fastMult

## Using the C++ wrapper
sys.path.append('C:\\Users\\robwi\\Documents\\ThesisFinal\\1-SpectralClustering_ACA_Components\\build\\Debug')
import fastMultModule

def lanczosFastMult(w_mat, deltas, k):
    """
    Performs the Lanczos algorithm with full reorthogonalization using a custom matrix-vector product.

    Args:
    - w_mat: The matrix for which we compute the tridiagonal matrix T.
    - deltas: List of scalars for the fastMult function.
    - k: Number of Lanczos iterations to perform.

    Returns:
    - T: The k x k symmetric tridiagonal matrix.
    - Q: The N x k matrix used for the similarity transformation, if requested.
    """
    n = w_mat.shape[1]
    v = np.ones((n, 1))

    # Degree matrix calculation
    degree = fastMultModule.fastMult(w_mat.T, np.ones(len(deltas))/deltas, np.ones((n, 1)))
    inv_degree = 1 / np.sqrt(degree)

    # Initialize variables
    Q = np.nan * np.ones((n, k))
    q = v / np.linalg.norm(v)
    Q[:, 0] = q.flatten()
    d = np.nan * np.ones((k,1))
    od = np.nan * np.ones((k - 1, 1))
    # Perform Lanczos iterations
    for i in range(k):

        # Unnormalized Laplacian calculation
        # z = np.multiply(degree, q) - fastMultModule.fastMult(w_mat.T, np.ones(len(deltas))/deltas, q)
        # z = fastMult(w_mat.T, deltas, q)
        # print(f'z_{i}:{z}')

        # Normalized Laplacian (symmetric) calculation
        temp1 = inv_degree * q
        temp2 = (degree * temp1) - fastMultModule.fastMult(w_mat.T, np.ones(len(deltas))/deltas, temp1)
        z = inv_degree * temp2

        d[i] = q.T.dot(z)
        # print(d)  


        # Full re-orthogonalization (x2)
        z = z - Q[:, :i+1] @ (Q[:, :i+1].T @ z)
        z = z - Q[:, :i+1] @ (Q[:, :i+1].T @ z)

        if i != k - 1:
            od[i] = np.linalg.norm(z)
            q = z / od[i]
            Q[:, i + 1] = q.flatten()

    # Construction T (tridiagonal matrix)
            
    # Ensure d and od are 1D arrays
    d = d.flatten()  # Flatten d to 1D
    od = od.flatten()  # Flatten od to 1D

    # Initialize T as a k x k matrix of zeros
    T = np.zeros((k, k))

    # Fill the diagonal of T with d
    np.fill_diagonal(T, d)

    # Fill the sub-diagonal and super-diagonal with od
    np.fill_diagonal(T[1:], od)  # Sub-diagonal
    np.fill_diagonal(T[:, 1:], od)  # Super-diagonal


    return T, Q
