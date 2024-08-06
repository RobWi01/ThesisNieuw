# Do here a first run with crop
import os
import sys
import time
import numpy as np

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


sys.path.append('C:/Users/robwi/Documents/ThesisFinal/1-SpectralClustering_ACA_Components') # Change this when uploading to gitlab, working with relative imports?
from LanczosFastMult import lanczosFastMult

from Low_rank_timeseries.util import create_cluster_problem, load_matrix, load_timeseries, load_svd
from Low_rank_timeseries.Low_rank_approx.ACA_diag_rook import ACA_diag_pivots
from Low_rank_timeseries.Data_paths import matrix_data_folder

from scipy.sparse import csgraph

from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem

from Low_rank_timeseries.util import load_labels

from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix


def symmetric_normalized_laplacian(S):
    # Step 1: Calculate the Degree Matrix D
    degree_matrix = np.diag(np.sum(S, axis=1))
    
    # Step 2: Calculate the Laplacian Matrix L
    laplacian_matrix = degree_matrix - S
    
    # Step 3: Calculate D^(-1/2)
    with np.errstate(divide='ignore'):
        inv_sqrt_degree_matrix = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    inv_sqrt_degree_matrix[np.isinf(inv_sqrt_degree_matrix)] = 0
    
    # Step 4: Calculate the Symmetric Normalized Laplacian L_norm
    normalized_laplacian_matrix = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix
    
    return normalized_laplacian_matrix



def get_amount_of_classes(labels):
    return len(np.unique(labels))


def myAlgo(lanczos_it, W, deltas, labels):

    num_cluster = get_amount_of_classes(labels)

    # print(W.shape)
    # print('deltas:', deltas)


    T, Q = lanczosFastMult(W, deltas, lanczos_it)
    T = np.array(T)  # Convert the 2D list to a 2D NumPy array
    Q = np.array(Q)  # Convert the 2D list to a 2D NumPy array
    end = time.time()

    D, V =  np.linalg.eigh(T)

    idx = D.argsort()
    D = D[idx]
    V = V[:, idx]

    # Eigenvectors of S are Q*V
    Q = np.dot(Q, V)
    T = np.dot(np.dot(np.transpose(V),T), V)  # Similarity transform for T

    # corr_eig = Q[: , 1:num_cluster+1] # This leave out the zero eigenvalue one
    corr_eig = Q[: , 1:num_cluster+1] # This leaves in the zero eigenvalue one
    # print(corr_eig)
    norm = np.sqrt(np.sum(corr_eig**2, axis=1, keepdims=True))
    norm_mat = corr_eig / norm

    # Watch out with true labels here, sometimes different order!

    print(norm_mat.shape)
    kmeans = KMeans(n_clusters=num_cluster, init='k-means++')
    kmeans.fit(norm_mat)
    predicted_labels = kmeans.predict(norm_mat)

    ARIscore = adjusted_rand_score(predicted_labels, labels)
    print('ARI score:', ARIscore)
    return corr_eig, ARIscore



if __name__ == "__main__":

    print('Updated')
    data_name = "FacesUCR   "
    compare_name = "dtw"
    scores = []
    scores_recon = []
    labels = load_labels(data_name)

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    file_path = os.path.join(base_dir, "Matrices", "Similarity_matrices", f"{data_name}_dtw.npy")

    S = np.load(file_path)


    num_clusters = get_amount_of_classes(labels)

    for i in range(10):

        cp = create_cluster_problem(data_name, compare_name,  no_solved_matrix=False, Distance=False, include_series=False)

        tau = 0.001
        kmax = 300   
        Deltak = 20  
        W, deltas, kbest, gamma = ACA_diag_pivots(cp, tau, kmax, Deltak)
        print(gamma)

        S_approx = np.zeros((len(labels), len(labels))) 
        reconstruct_matrix(S_approx, W, deltas, Distance = False, do_corrections = True)
        laplacian_matrix_approx = symmetric_normalized_laplacian(S_approx)
        laplacian_matrix_org = symmetric_normalized_laplacian(S)
        w,v = np.linalg.eig(laplacian_matrix_org)   
        # print(v[:,1])

        # # Initialize the SpectralClustering model
        # spectral_model = SpectralClustering(n_clusters=num_clusters,
        #                                     affinity='precomputed',
        #                                     assign_labels='kmeans')

        # Fit the model and predict the cluster labels
        # predicted_labels = spectral_model.fit_predict(S_approx)
        # score = adjusted_rand_score(predicted_labels, labels)
        corr_eig = v[: , 1:num_clusters+1]
        norm = np.sqrt(np.sum(corr_eig**2, axis=1, keepdims=True))
        norm_mat = corr_eig / norm

        # Watch out with true labels here, sometimes different order!

        print(norm_mat.shape)   
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
        kmeans.fit(norm_mat)
        predicted_labels = kmeans.predict(norm_mat)

        score = adjusted_rand_score(predicted_labels, labels)
        scores_recon.append(score)
        print('Reconstruction score:', score)

        eig, score = myAlgo(40, W, deltas, labels)
        scores.append(score)  # Append the returned ARIscore to the scores list

        result1 = v[:,1]
        result2 = eig[:, 0]
        print("error first eigv:", np.linalg.norm(result1 - result2, 2) / np.linalg.norm(result1, 2))
        
    # Convert the scores list to a NumPy array
    scores_array_recon = np.array(scores_recon)
    scores_array = np.array(scores)

    # Calculate the average, mean and standard deviation
    average_score_recon = np.average(scores_array_recon)
    mean_score_recon = np.mean(scores_array_recon)
    std_deviation_recon = np.std(scores_array_recon)
    Q3 = np.percentile(scores_array_recon, 75)
    Q1 = np.percentile(scores_array_recon, 25)
    IQR_recon = Q3 - Q1
        
    # Calculate the average, mean and standard deviation
    average_score = np.average(scores_array)
    mean_score = np.mean(scores_array)
    std_deviation = np.std(scores_array)
    Q3 = np.percentile(scores_array, 75)
    Q1 = np.percentile(scores_array, 25)
    IQR = Q3 - Q1

    print(data_name)
    print('Reconstruction')
    print(f"Average: {average_score_recon}, Mean: {mean_score_recon}, Standard Deviation: {std_deviation_recon}, IQR: {IQR_recon}")
    print('My method')
    print(f"Average: {average_score}, Mean: {mean_score}, Standard Deviation: {std_deviation}, IQR: {IQR}")
