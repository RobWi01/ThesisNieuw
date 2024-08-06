import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import warnings
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/1-SpectralClustering_ACA_Components')
from LanczosFastMult import lanczosFastMult
from Low_rank_timeseries.util import create_cluster_problem
from Low_rank_timeseries.Low_rank_approx.ACA_diag import ACA_diag_pivots
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix
from Low_rank_timeseries.util import load_labels

def symmetric_normalized_laplacian(S):
    degree_matrix = np.diag(np.sum(S, axis=1))
    laplacian_matrix = degree_matrix - S
    with np.errstate(divide='ignore'):
        inv_sqrt_degree_matrix = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    inv_sqrt_degree_matrix[np.isinf(inv_sqrt_degree_matrix)] = 0
    normalized_laplacian_matrix = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix
    return normalized_laplacian_matrix

def get_amount_of_classes(labels):
    return len(np.unique(labels))

def reorthogonalize(Q):
    for i in range(Q.shape[1]):
        for j in range(i):
            Q[:, i] -= np.dot(Q[:, i], Q[:, j]) * Q[:, j]
        Q[:, i] /= np.linalg.norm(Q[:, i])
    return Q

if __name__ == "__main__":

    name = 'CBF'
    labels = load_labels(name)
    cp = create_cluster_problem(name, "dtw", Distance=False, include_series=False)

    tau = 0.001
    kmax = int(0.10 * len(labels))
    Deltak = 200

    W, deltas, kbest, gamma = ACA_diag_pivots(cp, tau, kmax, Deltak)
    num_classes = 1
    size = W.shape[1]

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"

    file_path = os.path.join(base_dir, "Matrices", "Similarity_matrices", f"{name}_dtw.npy")

    S = np.load(file_path)

    S_approx = np.zeros((size, size))
    reconstruct_matrix(S_approx, W, deltas, Distance=False, do_corrections=False)
    laplacian_matrix_approx = symmetric_normalized_laplacian(S_approx)
    laplacian_matrix_exact = symmetric_normalized_laplacian(S)

    w, v = eigsh(laplacian_matrix_approx, k=num_classes + 1, which='SM')
    w = np.real(w)
    v = np.real(v)
    idx = w.argsort()
    w = w[idx]
    v = v[:, idx]

    # Set the first eigenvalue to zero if it falls below a certain threshold
    zero_threshold = 1e-10
    if w[0] < zero_threshold:
        w[0] = 0

    true_eigenvectors = v[:, :num_classes]  # First num_classes eigenvectors
    eigs_approx = np.zeros((size, num_classes, 28))

    for l in range(3, 31):
        T, Q = lanczosFastMult(W, deltas, l)
        D, V = np.linalg.eig(T)
        D = np.real(D)
        V = np.real(V)
        idx = D.argsort()
        D = D[idx]
        V = V[:, idx]

        print(f"Shape of Q: {Q.shape}")
        print(f"Shape of V: {V.shape}")
        print(f"Shape of approx holder: {eigs_approx.shape}")

        if D[0] < zero_threshold:
            D[0] = 0

        if len(D) >= num_classes:
            Q = Q[:, :num_classes]  # Keep only the first num_classes Lanczos vectors
            V = V[:num_classes, :num_classes]  # Ensure V has the correct number of columns
            print(f"Shape of Q: {Q.shape}")
            print(f"Shape of V: {V.shape}")
            test = Q @ V
            print(f"Shape of QV: {test.shape}")
            eigs_approx[:, :, l - 3] = Q @ V

    rel_diffs = np.zeros((num_classes, 28))

    for i in range(num_classes):
        true_vec = true_eigenvectors[:, i]
        for j in range(28):
            approx_vec = eigs_approx[:, :, j][:, i]
            rel_diffs[i, j] = np.linalg.norm(true_vec - approx_vec) / np.linalg.norm(true_vec)

    convergence_threshold = 1e-6
    converged_iters = np.argmax(rel_diffs < convergence_threshold, axis=1)

    for i in range(num_classes):
        print(f"Convergence iteration for eigenvector {i}: {converged_iters[i] + 3}")

    iterations = np.arange(3, 3 + rel_diffs.shape[1])

    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        print(f'Relative errors for eigenvector {i}: {rel_diffs[i]}')
        plt.plot(iterations, rel_diffs[i], label=f'Eigenvector {i}')

    plt.xlabel('Lanczos Iterations')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.ylim([10 ** -18, 1])
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.minorticks_on()
    plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='all'))

    plt.title('Relative Errors for Corresponding Eigenvectors')
    plt.show()
