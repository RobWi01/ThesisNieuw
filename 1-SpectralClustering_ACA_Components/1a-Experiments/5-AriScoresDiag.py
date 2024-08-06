import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/1-SpectralClustering_ACA_Components')
from LanczosFastMult import lanczosFastMult
from Low_rank_timeseries.util import create_cluster_problem, load_matrix, load_timeseries, load_svd, load_timeseries_and_labels
from Low_rank_timeseries.Low_rank_approx.ACA_diag import ACA_diag_pivots
from Low_rank_timeseries.Data_paths import matrix_data_folder
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.util import load_labels
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix

def symmetric_normalized_laplacian(S):
    degree_matrix = np.diag(np.sum(S, axis=1))
    laplacian_matrix = degree_matrix - S
    inv_sqrt_degree_matrix = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    normalized_laplacian_matrix = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix
    return normalized_laplacian_matrix

def get_amount_of_classes(labels):
    return len(np.unique(labels))

def myAlgo(lanczos_it, W, deltas, labels):
    num_cluster = get_amount_of_classes(labels)
    T, Q = lanczosFastMult(W, deltas, lanczos_it)
    T = np.array(T)
    Q = np.array(Q)
    D, V =  np.linalg.eigh(T)
    idx = D.argsort()
    D = D[idx]
    V = V[:, idx]
    Q = np.dot(Q, V)
    # Q = reorthogonalize(Q)
    corr_eig = Q[: , 1:num_cluster+1]
    norm = np.sqrt(np.sum(corr_eig**2, axis=1, keepdims=True))
    norm_mat = corr_eig / norm
    kmeans = KMeans(n_clusters=num_cluster, init='k-means++')
    kmeans.fit(norm_mat)
    predicted_labels = kmeans.predict(norm_mat)
    ARIscore = adjusted_rand_score(predicted_labels, labels)
    print('My method, ARI score:', ARIscore)
    return corr_eig, ARIscore

def reorthogonalize(Q):
    """Reorthogonalize the columns of Q using the Gram-Schmidt process."""
    for i in range(Q.shape[1]):
        for j in range(i):
            Q[:, i] -= np.dot(Q[:, i], Q[:, j]) * Q[:, j]
        Q[:, i] /= np.linalg.norm(Q[:, i])
    return Q


if __name__ == "__main__":
    dataset_names = labels = [
    "CBF", 
    # "Crop",
    "ItalyPowerDemand",
    "Mallat", 
    # "StarLightCurves", 
    "Symbols", 
    # "TwoPatterns"
]

    # Still filter on the really small datasets --> don

    # bigger_dataset_names = ["Wafer"]

    # "Mallat" 
    # "StarLightCurves",
    # "TwoPatterns", "Symbols"

    # dataset_names = ['Symbols']

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    sampling_percentages = [0.01]
    
    # np.arange(0.02, 0.04, 0.01)
    
    results = []

    for data_name in dataset_names:
        print(f'Processing dataset: {data_name}')
        compare_name = "dtw"
        labels = load_labels(data_name)
        file_path = os.path.join(base_dir, "Matrices", "Similarity_matrices", f"{data_name}_dtw.npy")
        S = np.load(file_path)

        file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{data_name}_dtw.npy")
        A = np.load(file_path)

        labels, series = load_timeseries_and_labels(data_name)
        num_clusters = get_amount_of_classes(labels)


        for sp in sampling_percentages:
            print(f'Processing percentage: {sp}')
            scores = []
            scores_recon = []

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                for i in range(10):
                    cp = ClusterProblem(labels, series, "dtw", solved_matrix=A, similarity=False)
                    tau = 0.001
                    Deltak = 20
                    kmax = int(sp*len(labels))
                    W, deltas, kbest, gamma = ACA_diag_pivots(cp, tau, kmax, Deltak)
                    S_approx = np.zeros((len(labels), len(labels))) 
                    reconstruct_matrix(S_approx, W, deltas, Distance=False, do_corrections=False)
                    laplacian_matrix_approx = symmetric_normalized_laplacian(S_approx)
                    w,v = np.linalg.eig(laplacian_matrix_approx)
                    corr_eig = np.real(v[: , 1:num_clusters+1])
                    norm = np.sqrt(np.sum(corr_eig**2, axis=1, keepdims=True))
                    norm_mat = corr_eig / norm
                    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
                    kmeans.fit(norm_mat)
                    predicted_labels = kmeans.predict(norm_mat)
                    score = adjusted_rand_score(predicted_labels, labels)
                    scores_recon.append(score)
                    print('Reconstruction score:', score)
                    eigv, score = myAlgo(19, W, deltas, labels)
                    scores.append(score)

                    result1 = v[:,1]
                    result2 = eigv[:, 0]
                    print("error first eigv:", np.linalg.norm(result1 - result2, 2) / np.linalg.norm(result1, 2))

            scores_array_recon = np.array(scores_recon)
            scores_array = np.array(scores)
            average_score_recon = np.average(scores_array_recon)
            mean_score_recon = np.mean(scores_array_recon)
            std_deviation_recon = np.std(scores_array_recon)
            Q3_recon = np.percentile(scores_array_recon, 75)
            Q1_recon = np.percentile(scores_array_recon, 25)
            IQR_recon = Q3_recon - Q1_recon
            average_score = np.average(scores_array)
            mean_score = np.mean(scores_array)
            std_deviation = np.std(scores_array)
            Q3 = np.percentile(scores_array, 75)
            Q1 = np.percentile(scores_array, 25)
            IQR = Q3 - Q1
            results.append({
                'Dataset': data_name,
                'Sampling Percentage': sp,
                'kbest': kbest,
                'Average Score Recon': average_score_recon,
                'Mean Score Recon': mean_score_recon,
                'Std Deviation Recon': std_deviation_recon,
                'IQR Recon': IQR_recon,
                'Average Score': average_score,
                'Mean Score': mean_score,
                'Std Deviation': std_deviation,
                'IQR': IQR
            })

    df = pd.DataFrame(results)
    output_path = os.path.join(base_dir, "clustering_results_diag_test_final_001.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")
