import copy
import os
import sys
import os

path = os.getcwd()

print(path)
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from cluster_problem import ClusterProblem
from data_loader import load_timeseries_from_tsv
from extendable_aca import ACA
import random as rn
from collections import Counter
import numpy as np

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import csgraph

from Low_rank_timeseries.util import create_cluster_problem
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/1-SpectralClustering_ACA_Components')
from LanczosFastMult import lanczosFastMult

def get_amount_of_classes(labels):
    return len(np.unique(labels))

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

def compute_row(W, inv_deltas, t_new, indices):
    """
    Compute the row A_k(n + b, :) as given in the text.
    """
    k = W.shape[0]  # Number of skeletons
    n = W.shape[1]  # Number of time series
    row = np.zeros(n)
    for i in range(k):
        row += (W[i, -1] * W[i, :]) / inv_deltas[i]
    return row

def calculateKNNClassification(approx, start_index, labels, k, name, approx2, series_test, dm):
    train_indices = np.arange(start_index)
    size = len(train_indices)
    test_indices = np.arange(start_index, len(labels))

    correct_predictions = 0
    for index2 in range(len(series_test)):
        T_new = series_test[index2]

        active_dm = dm[range(start_index+index2+1), :]
        active_dm = active_dm[:, range(start_index+index2+1)]

        approx2.extend([T_new], method='method3', solved_matrix=active_dm)

        W = np.array(approx2.rows)
        inv_deltas = np.ones(len(approx2.deltas)) / approx2.deltas

        rows = compute_row(W, inv_deltas, T_new, 0)

        approx_distances = rows[:size]

        sorted_indices = np.argsort(approx_distances)
        top_indices = sorted_indices[:4]
        extracted_labels = [labels[idx] for idx in top_indices]
        label_counts = Counter(extracted_labels)
        most_common_label = label_counts.most_common(1)[0][0]

        if most_common_label == labels[start_index + index2]:
            correct_predictions += 1

    accuracy = correct_predictions / len(series_test)
    print(f"KNN Classification Accuracy: {accuracy}")

    return accuracy

def add_series_to_dm(true, next, dm):
    print(next)
    all_dtw = np.transpose(dm[next, range(next)])
    print(all_dtw.shape)
    true = np.append(true, [np.zeros(len(true[1, :]))], 0)
    print(true.shape)
    true = np.append(true, np.transpose([np.zeros(len(true[1, :]) + 1)]), 1)

    true[:, next] = all_dtw
    true[next, :] = all_dtw
    return true

def extend_approximations(approximations, methods, new_series, solved_matrix=None):
    for approximation, method in zip(approximations, methods):
        approximation.extend(new_series, method=method, solved_matrix=solved_matrix)

def print_result(new_result):
    print("KNN", "Iteration", new_result[0], "Accuracy:", new_result[1])

def update_results(approximations, results, labels, true_dm, k, index, start_index, skip, name, testseries, dm):
    for approx, result in zip(approximations, results):
        knn_accuracy_approx = calculateKNNClassification(approx.getApproximation(), index, labels, k, name, approx, testseries, dm)
        knn_accuracy_exact = calculateKNNClassification(true_dm, index, labels, k, name, approx, testseries, dm)
        
        new_result = [index, knn_accuracy_exact, knn_accuracy_approx]
        print_result(new_result)
        result[len(result) - 1, :, int((index - start_index) / skip)] = np.array(new_result)

def read_all_results(file_names, size, start_index, skip):
    results = []
    for file_name in file_names:
        try:
            result = np.load(file_name + ".npy")
            n_skips = int((size - start_index) / skip)
            results.append(np.append(result, [np.zeros((3, n_skips))], 0))
        except:
            results.append(np.zeros((1, 3, int((size - start_index) / skip))))
    return results

def do_full_experiment(series, labels, dm, start_index, skip, methods, cluster_algo, name, k=5, rank=15, iterations=100, random_file=True):
    func_name = "dtw"
    args = {"window": len(series) - 1}
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
        print("test:", start_index)
        cp = ClusterProblem(series[0:start_index], func_name, compare_args=args, solved_matrix=active_dm)
        results = read_all_results(file_names, len(series), start_index, skip)
        start_index_approx = rn.randint(0, start_index - 1)
        seed = rn.randint(0, 99999999)
        print(name + ":" + " STARTING NEW APPROX: it =", len(results[0]), "start index approx =", start_index_approx,
              "seed =", seed, "skip =", skip)
        approximations = [ACA(cp, tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed)]
        index = start_index
        update_results(approximations, results, labels, active_dm, k, start_index, start_index, skip, name, series[start_index:], dm)
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
                update_results(approximations, results, labels, active_dm, k, index, start_index, skip, name, series[start_index:], dm)
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

    do_full_experiment(series, labels, true_dm, start, skip, methods, "knn", name, k=5, rank=9000,
                       iterations=1000, random_file=False)
