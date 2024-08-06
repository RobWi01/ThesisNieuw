import os 

import numpy as np
from scipy.sparse import csgraph

import matplotlib.pyplot as plt


def symmetric_normalized_laplacian(S):
    degree_matrix = np.diag(np.sum(S, axis=1))
    laplacian_matrix = degree_matrix - S
    with np.errstate(divide='ignore'):
        inv_sqrt_degree_matrix = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    inv_sqrt_degree_matrix[np.isinf(inv_sqrt_degree_matrix)] = 0
    normalized_laplacian_matrix = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix
    return normalized_laplacian_matrix

def compare_eigen_svd(matrix):
    # Calculate Eigenvalues
    eigenvalues, _ = np.linalg.eigh(matrix)

    # Calculate Singular Values
    _, singular_values, _ = np.linalg.svd(matrix)

    # Sort eigenvalues and singular values in descending order
    eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
    singular_values_sorted = np.sort(singular_values)[::-1]

    # Print and compare
    print("Eigenvalues:", eigenvalues_sorted)
    print("Singular Values:", singular_values_sorted)

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(eigenvalues_sorted, label='Eigenvalues', marker='o')
    plt.plot(singular_values_sorted, label='Singular Values', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison of Eigenvalues and Singular Values')
    plt.legend()
    plt.grid(True)
    plt.show()

name = "CBF"

# Directly specify the absolute path for clarity and reliability
base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
file_path = os.path.join(base_dir, "Matrices", "Similarity_matrices", f"{name}_dtw.npy")

try:
    similarity_matrix = np.load(file_path)
except IOError:
    raise Exception(f"Unable to load the file. Please check the path: {file_path}")

laplacian_matrix_approx = symmetric_normalized_laplacian(similarity_matrix)

compare_eigen_svd(laplacian_matrix_approx)
