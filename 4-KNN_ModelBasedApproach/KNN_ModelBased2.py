import os
import numpy as np
from dtaidistance import dtw
from concurrent.futures import ProcessPoolExecutor

from Low_rank_timeseries.util import load_timeseries_and_labels, create_cluster_problem, get_amount_of_classes, load_labels
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.Low_rank_approx.ACA_diag_pivots import ACA_diag_pivots
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix
from Low_rank_timeseries.Distance_functions.util import distance_to_similarity

###################################### Helper Functions ######################################

def load_timeseries_and_labels_from_tsv(path):
    if os.path.isfile(path):
        data = np.genfromtxt(path, delimiter='\t')
        labels, series = data[:, 0], data[:, 1:]
        labels = np.array(labels, dtype=np.float32)
        series = np.array(series, dtype=np.float32)
    else:
        labels = None
        series = None
    return labels, series

def get_orginal_rows(cp:ClusterProblem, kbest, pivot_indices):

    rows = np.ndarray((kbest[0], cp.cp_size()), np.single)

    for i in range(len(pivot_indices)):
        cp.sample_row(pivot_indices[i], rows[i,:])

    return rows

def apply_threshold(matrix, threshold):
    filtered_matrix = np.where(matrix >= threshold, matrix, 0)
    return filtered_matrix

def worker(args):
    train_instance, test_instance, gamma = args
    return dtw.distance(train_instance, test_instance, use_pruning=True), gamma

def calculate_distances(train_set, test_instance, gamma):
    with ProcessPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your CPU
        futures = [executor.submit(dtw.distance, train, test_instance) for train in train_set]
        distances = [(future.result(), gamma) for future in futures]
    return distances


def calculate_distances_precomputed(Dis_Mat, indices, i, train_size):
    distances = []

    for index in indices:
        di = Dis_Mat[index][train_size + i]
        distances.append(index)
    return distances

# def distance_to_similarity(distances):
#     """
#     Convert distances to similarities using a Gaussian kernel.
#     """
#     if not distances:
#         return np.array([])
    
#     dist, gammas = zip(*distances)  # Unzip into separate lists
#     dist = np.array(dist)
#     gammas = np.array(gammas).flatten()  # Ensure gamma is a flat array for broadcasting

#     # Broadcast gamma across distances if necessary
#     similarities = np.exp(-1 * np.square(dist / gammas))

    return similarities

def knn_aca_model_based(name, K_nb, sim_threshold_within_skeletons, sim_threshold_between_skeletons_tnew):
    train_path = os.path.join("C:/Users/robwi/Documents/ThesisClean/Data", name, f"{name}_TRAIN.tsv")
    test_path = os.path.join("C:/Users/robwi/Documents/ThesisClean/Data", name, f"{name}_TEST.tsv")

    labels_train, series_train = load_timeseries_and_labels_from_tsv(train_path)
    _, series_test = load_timeseries_and_labels_from_tsv(test_path)

    cp = ClusterProblem(labels_train, series_train, "dtw", similarity=False)

    tau = 0.001
    kmax = int(0.25*len(series_train)) # max is 25% of the rang of the training matrix
    Deltak = 200
    _, _, kbest, gamma, pivot_indices = ACA_diag_pivots(cp, tau, kmax, Deltak) # Source code modified to also return the chosen pivot indices

    rows = np.array(get_orginal_rows(cp, kbest, pivot_indices))

    #TO-DO: Check sample rate after I get these original rows as an experiment to run

    filtered_rows = apply_threshold(rows, sim_threshold_within_skeletons)

    # Find the non-zero indices for each row
    non_zero_indices_per_row = [np.nonzero(row)[0] for row in filtered_rows]

    true_labels = load_labels(name)

    # Directly specify the absolute path for clarity and reliability
    base_dir = "C:/Users/robwi/Documents/ThesisFinal/"
    file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{name}_dtw.npy")

    size = len(series_train)
    A = np.load(file_path)

    # Convert indices to labels for each row
    labels_per_row = []
    j = 0
    for non_zero_indices in non_zero_indices_per_row:
        labels_for_this_row = [(rows[j][idx], true_labels[idx]) for idx in non_zero_indices]
        labels_per_row.append(labels_for_this_row)
        j += 1

    results = []
    # Loop over each sublist in the nested list
    for sublist in labels_per_row:  
        # Dictionary to accumulate the similarities for each label in the sublist
        similarity_per_label = {}
        
        # Process each tuple in the sublist
        for similarity, label in sublist:
            if label in similarity_per_label:
                similarity_per_label[label] += similarity
            else:
                similarity_per_label[label] = similarity
        
        # Append the dictionary to the results list
        results.append(similarity_per_label)

    predictions = []
    for i in range(len(series_test)):
        distances = calculate_distances_precomputed(A, pivot_indices, i, size)

        weights = distance_to_similarity(distances, gamma=gamma, method="Gaussian")
        weights = [weight if weight >= sim_threshold_between_skeletons_tnew else 0 for weight in weights]



        if len(weights) != len(results):
            print("Warning: The number of weights does not match the number of sublists.")
        else:
            weighted_results = []

            # Loop over each sublist's results and corresponding weight
            for weight, result_dict in zip(weights, results):
                # Dictionary to store the weighted similarities for each label
                weighted_dict = {label: similarity * weight for label, similarity in result_dict.items()}
                weighted_results.append(weighted_dict)

            # Dictionary to accumulate the total weighted similarities for each label
            total_scores = {}

            # Loop over all weighted results
            for weighted_result in weighted_results:
                for label, similarity in weighted_result.items():
                    if label in total_scores:
                        total_scores[label] += similarity
                    else:
                        total_scores[label] = similarity

            max_label = max(total_scores, key=total_scores.get)
                
        predictions.append(max_label)
    return predictions


