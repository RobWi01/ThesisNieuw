# Weer gebaseerd op Matlab (die code werk nu helemaal)

# Hier krijg ook de verwachte conventionele scores, waarom heeft mijn methode een betere score dat snap ik nog niet helemaal,
# I don't to better than the reconstruction that is already very nice to see.
# Sometimes I do better then the reconstruction and not fully aware of why that is the case.


# Input that code here that I can use that as a benchmark


import numpy as np
import time
from scipy.linalg import pinv, sqrtm
from sklearn.cluster import KMeans

def nystrom(num_samples, num_clusters, S):
    """
    NYSTROM Spectral clustering using the Nystrom method.

    :param data: N-by-D data matrix, where N is the number of data, D is the number of dimensions
    :param num_samples: number of random samples
    :param sigma: sigma value used in computing similarity
    :param num_clusters: number of clusters
    :param S: similarity matrix
    :return: cluster_labels, evd_time, kmeans_time, total_time
    """
    # Randomly select samples
    print('Randomly selecting samples...')
    num_rows = S.shape[0]
    inds = np.random.choice(num_rows, num_samples, replace=False)
    # Sort the indices
    inds = np.sort(inds)
    # inds = np.array([l for l in range(num_samples)])
    inv_inds = np.array([a for a in range(num_rows) if a not in inds])
    start_time = time.time()

    # permed_index = np.random.permutation(num_rows)
    # sampled_indices = permed_index[:num_samples]
    # remaining_indices = permed_index[num_samples:]
    
    # Calculate the euclidean distance between samples themselves
    A = S[inds, :].astype(np.float32)
    A = A[:, inds]

    print(A)
    # A = S[:num_samples, :num_samples]
    print(f'Shape of A: {A.shape}')
    
    # Calculate the euclidean distance between samples and other points
    B = S[inds, :].astype(np.float32)
    B = B[:, inv_inds]
    # B = S[:num_samples, num_samples:]
    print(f'Shape of B: {B.shape}')
    
    # Normalize A and B for Laplacian
    print('Normalizing A and B for Laplacian...')
    B_T = B.T
    d1 = np.sum(A, axis=1) + np.sum(B, axis=1)
    d2 = np.sum(B_T, axis=1) + B_T @ (pinv(A) @ np.sum(B, axis=1))
    dhat = np.sqrt(1.0 / np.concatenate((d1, d2)))
    A = A * np.outer(dhat[:num_samples], dhat[:num_samples])
    B = B * np.outer(dhat[:num_samples], dhat[num_samples:])

    # Try that other code here and also just look at the eigenvectors and how well these are approximated
    
    time1 = time.time()
    
    # Orthogonalization and eigendecomposition
    print('Orthogonalizing and eigendecomposition...')
    Asi = np.real(sqrtm(pinv(A)))
    BBT = B @ B_T
    W = np.zeros((A.shape[0] + B_T.shape[0], A.shape[1]), dtype=np.float32)
    W[:A.shape[0], :] = A
    W[A.shape[0]:, :] = B_T

    # print('W:', W)

    # Calculate R = A + A^-1/2*B*B'*A^-1/2
    R = A + Asi @ BBT @ Asi
    R = (R + R.T) / 2  # Make sure R is symmetric
    U, L, _ = np.linalg.svd(R)
    sorted_indices = np.argsort(-L)
    U = U[:, sorted_indices]
    L = np.diag(L[sorted_indices])
    
    W = W @ Asi
    V = W @ U[:, :num_clusters+1] @ pinv(np.sqrt(L[:num_clusters+1, :num_clusters+1]))
    
    # print('V:', V)
    time2 = time.time()
    
    # Perform k-means
    print('Performing kmeans...')
    # # Normalize each row to be of unit length
    # sq_sum = np.sqrt(np.sum(V**2, axis=1)) 
    # U_norm = V / sq_sum[:, np.newaxis]
    # kmeans = KMeans(n_clusters=num_clusters).fit(U_norm)
    # cluster_labels = kmeans.labels_

    corr_eig = V[: , 1:num_clusters+1] # This leaves in the zero eigenvalue one
    norm = np.sqrt(np.sum(corr_eig**2, axis=1, keepdims=True))
    norm_mat = corr_eig / norm
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    kmeans.fit(norm_mat)
    cluster_labels = kmeans.predict(norm_mat)
    cluster_labels[np.hstack((inds,inv_inds))] = cluster_labels
    
    total_time = time.time()
    
    # Calculate and show time statistics
    evd_time = time2 - time1
    kmeans_time = total_time - time2
    total_time = total_time - start_time
    
    print('Finished!')
    
    return cluster_labels, evd_time, kmeans_time, total_time    