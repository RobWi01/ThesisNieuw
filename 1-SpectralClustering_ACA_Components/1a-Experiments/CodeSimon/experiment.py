import copy
import os
import sys
import os

path = os.getcwd()

print(path)
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from cluster_problem import ClusterProblem
from data_loader import load_timeseries_from_tsv
from extendable_aca import ACA
import random as rn
import numpy as np

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

from Low_rank_timeseries.util import create_cluster_problem
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix




sys.path.append('C:/Users/robwi/Documents/ThesisFinal/1-SpectralClustering_ACA_Components')
from LanczosFastMult import lanczosFastMult

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



def distance_to_similarity(distance, gamma=None, method="Gaussian"):
    """
    Convert the given distance values to similarity values.
    Supported methods are 'Gaussian'
    :param distance: array with distance values, or 1 distance value
    :param gamma: parameter for the similarity method
    :param method: String that indicated which similarity method to use. The default is Gaussian.
    :return: Array with similarity values
    """
    if method == "Gaussian":
        if gamma is None:
            raise Exception('gamma should not be None if the method is Gaussian')
        return np.exp(-1*np.square(np.divide(distance, gamma)))



def symmetric_normalized_laplacian(S):
    degree_matrix = np.diag(np.sum(S, axis=1))
    laplacian_matrix = degree_matrix - S
    inv_sqrt_degree_matrix = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    normalized_laplacian_matrix = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix
    return normalized_laplacian_matrix


def calculateClusters(approx, index, labels, a_spectral, k, name, approx2):
    rows = distance_to_similarity(approx2.rows, gamma=a_spectral)
    deltas = distance_to_similarity(np.ones(len(approx2.deltas))/approx2.deltas, gamma=a_spectral)
    print(f"rows shape: {rows.shape}, deltas shape: {deltas.shape}")

    

    # S_approx = np.zeros((rows.shape[1], rows.shape[1])) 
    # reconstruct_matrix(S_approx, approx2.rows,  np.ones(len(approx2.deltas))/approx2.deltas, Distance=True, do_corrections=True)
    # print(S_approx)
    # S_approx = distance_to_similarity(S_approx, gamma=a_spectral)
    # print(S_approx)
    # laplacian_matrix_approx = symmetric_normalized_laplacian(S_approx)


    # Make reconstruction here to check

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=0)
        # pred_spectral = spectral.fit_predict( laplacian_matrix_approx)
        # ARI_Approx_spectral2 = adjusted_rand_score(labels[0:index], pred_spectral)

        agglomerative = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete')
        pred_agglo = agglomerative.fit_predict(approx)

        labels_subset = labels[:index]  # Adjust labels to the correct length
        eigv, ARI_Approx_spectral = myAlgo(20, rows, deltas, labels_subset)
        
        if len(pred_agglo) != len(labels_subset):
            raise ValueError(f"Length mismatch: pred_agglo length {len(pred_agglo)}, labels length {len(labels_subset)}")
        
        ARI_Approx_agglomerative = adjusted_rand_score(labels_subset, pred_agglo)

    print(f"Spectral ARI: {ARI_Approx_spectral}, Agglomerative ARI: {ARI_Approx_agglomerative}, Reconstruction ARI: {0}")
    
    return ARI_Approx_spectral, ARI_Approx_agglomerative

def add_series_to_dm(true, next, dm):
    next = next - 1
    all_dtw = np.transpose(dm[next, range(next + 1)])
    true = np.append(true, [np.zeros(len(true[1, :]))], 0)
    true = np.append(true, np.transpose([np.zeros(len(true[1, :]) + 1)]), 1)

    true[:, next] = all_dtw
    true[next, :] = all_dtw
    return true


def extend_approximations(approximations, methods, new_series, solved_matrix=None):
    for approximation, method in zip(approximations, methods):
        approximation.extend(new_series, method=method, solved_matrix=solved_matrix)


def print_result(new_result):
    print("Spectral", "Iteration", new_result[0], "Approx ARI:", new_result[3])


def update_results(approximations, results, labels, true_dm, a_spectral, k, index, start_index, skip, name):
    for approx, result in zip(approximations, results):
        ARI_score_spec_approx, ARI_score_agglo_approx = calculateClusters(approx.getApproximation(), index, labels,
                                                                          a_spectral, k, name, approx)
        ARI_score_spec_exact, ARI_score_agglo_exact = calculateClusters(true_dm, index, labels, a_spectral, k, name, approx)
        amount_of_skeletons = len(approx.rows) + len(approx.full_dtw_rows)
        amount_of_dtws = approx.get_DTW_calculations()
        new_result = [index, amount_of_skeletons, ARI_score_spec_exact, ARI_score_agglo_exact,
                      ARI_score_spec_approx, ARI_score_agglo_approx, amount_of_dtws]
        print_result(new_result)
        result[len(result) - 1, :, int((index - start_index) / skip)] = np.array(new_result)


def read_all_results(file_names, size, start_index, skip):
    results = []
    for file_name in file_names:
        try:
            result = np.load(file_name + ".npy")
            n_skips = int((size - start_index) / skip)
            results.append(np.append(result, [np.zeros((7, n_skips))], 0))
        except:
            results.append(np.zeros((1, 7, int((size - start_index) / skip))))
    return results


def do_full_experiment(series, labels, dm, start_index, skip, methods, cluster_algo, a_spectral, name, rank=15,
                       iterations=100, random_file=True):
    func_name = "dtw"
    args = {"window": len(series) - 1}
    k = len(set(labels))
    file_names = []
    seed_file_name = rn.randint(0, 9999999999)
    for method in methods:
        if random_file:
            file_names.append("results/part2/" + name + "/" + str(seed_file_name) + "_" + method)
        else:
            file_names.append("results/part2/" + name + "/" + name + "_" + method)
    results = read_all_results(file_names, len(series), start_index, skip)
    while len(results[0]) <= iterations:
        if dm is not None:
            active_dm = dm[range(start_index), :]
            active_dm = active_dm[:, range(start_index)]
        else:
            active_dm = None
        cp = ClusterProblem(series[0:start_index], func_name, compare_args=args, solved_matrix=active_dm)
        results = read_all_results(file_names, len(series), start_index, skip)
        start_index_approx = rn.randint(0, start_index - 1)
        seed = rn.randint(0, 99999999)
        print(name + ":" + " STARTING NEW APPROX: it =", len(results[0]), "start index approx =", start_index_approx,
              "seed =", seed, "skip =", skip)
        approximations = [ACA(cp, tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed)]
        index = start_index
        update_results(approximations, results, labels, active_dm, a_spectral, k, index, start_index, skip, name)
        new_series = []
        while index < len(series) - 1:
            index += 1
            new_series.append(series[index])
            if index % skip == 0:
                if dm is not None:
                    active_dm = dm[range(index), :]
                    active_dm = active_dm[:, range(index)]
                else:
                    active_dm = None
                extend_approximations(approximations, methods, new_series, solved_matrix=active_dm)
                update_results(approximations, results, labels, active_dm, a_spectral, k, index, start_index, skip,
                               name)
                new_series = []

        for file_name, result in zip(file_names, results):
            np.save(file_name, result)


def load_data(name):
    data_dir = "C:/Users/robwi/Documents/ThesisFinal/Data/"

    path_train = data_dir + name + "/" + name + "_TRAIN.tsv"
    labels_train, series_train = load_timeseries_from_tsv(path_train)

    path_test = data_dir + name + "/" + name + "_TEST.tsv"
    labels_test, series_test = load_timeseries_from_tsv(path_test)

    labels = np.concatenate((labels_train, labels_test), axis=0)
    series = np.concatenate((series_train, series_test), axis=0)
    return series, labels


# already_found_names = ["CBF", "ChlorineConcentration", "CinCECGTorso"]
# names = [x[1] for x in [y[0].split("\\") for y in os.walk("Data")][1:-1] if x[1] not in already_found_names]
names = ["CBF"]

cp3 = create_cluster_problem('CBF', 'dtw', no_solved_matrix=False, Distance=False, include_series=False)

gamma = cp3.give_95_percentile()
print(gamma)
variables_spectral = [gamma[0]]
for name, a_spectral in zip(names, variables_spectral):
    series, labels = load_data(name)
    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{name}_dtw.npy")


    true_dm = np.load(file_path)
    methods = ["method3"]
    start = int(len(series) / 2)
    skip = int((len(series) - start) / 5)
    print("start: ", start, "Skip: ", skip)

    do_full_experiment(series, labels, true_dm, start, skip, methods, "spectral", a_spectral, name, rank=9000,
                       iterations=1000, random_file=False)