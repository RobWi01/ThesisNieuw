import os
import time
import numpy as np

from dtaidistance import dtw

from collections import Counter

from Low_rank_timeseries.util import load_timeseries_and_labels
from Low_rank_timeseries.util import create_cluster_problem
from Low_rank_timeseries.util import create_cluster_problem
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.Low_rank_approx.ACA_diag import ACA_diag_pivots
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix


###################################### Helper Functions ######################################

def load_timeseries_and_labels_from_tsv(path):
    """
    Loads Time Series from TSV file. The Format is expected to be the Class number as first element of the row,
    followed by the elements of the time series.
    :param path: The path where the TSV file is located
    :return: the labels and the timeseries. The i'th timeserie is series[i] or series[i,:].
    """

    if os.path.isfile(path):
        data = np.genfromtxt(path, delimiter='\t')
        labels, series = data[:, 0], data[:, 1:]
        labels = np.array(labels, np.single)
        series = np.array(series, np.single)
    else:
        labels = None
        series = None

    return labels, series

###################################### Main Code ######################################

def main():
    data_name = 'MixedShapesSmallTrain'
    compare_name = 'dtw'

    cp = create_cluster_problem(data_name, compare_name, no_solved_matrix=False, Distance=False, include_series=False)

    tau = 0.001
    kmax = 200
    Deltak = 20
    W, deltas, kbest, gamma = ACA_diag_pivots(cp, tau, kmax, Deltak)

    A_approx = np.zeros((len(labels), len(labels))) 



    k_nb = 7

    correct_predictions = 0
    for i in range(nb_samples):
        T_new = series_test_sampled[i]
        sim_measures = []
        for ts in series_train:
            di = dtw.distance(ts, T_new, use_pruning=True)  # Added use_pruning for efficiency
            sim = distance_to_similarity(di, gamma=gamma, method="Gaussian")
            sim_measures.append(sim[0]) # Werken met numpy arrays hier! (vaste grootte van array)

        sim_measures = np.array(sim_measures)
        sorted_indices = np.argsort(-sim_measures)

        labels_for_this_row2 = [labels_train[idx] for idx in sorted_indices[:k_nb]]

        most_common_label = Counter(labels_for_this_row2).most_common(1)[0][0]
        # print(most_common_label)
        # print(labels_test_sampled[i])

        if most_common_label == labels_test_sampled[i]:
            correct_predictions += 1

        else:
            print(most_common_label)
            print(labels_test_sampled[i])

    bruteforce_method_accuracy = correct_predictions / nb_samples
    print(f'Brute Force Accuracy: {bruteforce_method_accuracy}')


    correct_predictions = 0
    for i in range(nb_samples):  # Sample 10 new time series
        T_new = series_test_sampled[i]
        new_W = update_skeletons(W, inv_deltas, pivot_indices, series_train, T_new, size, 1, gamma)
        S_approx = np.zeros((size+1, size+1))
        reconstruct_matrix(S_approx, new_W, inv_deltas, Distance=False, do_corrections=False)
        similarities = S_approx[size, :size]
        sorted_indices = np.argsort(-similarities)
        top_indices = sorted_indices[:k_nb]
        extracted_labels = [labels_train[idx] for idx in top_indices]
        label_counts = Counter(extracted_labels)
        most_common_label = label_counts.most_common(1)[0][0]

        print(most_common_label)

        if most_common_label == labels_test_sampled[i]:
            correct_predictions += 1


    bruteforce_method_accuracy = correct_predictions / nb_samples
    print(f'ACA with Update: {bruteforce_method_accuracy}')


if __name__ == "__main__":
    main()