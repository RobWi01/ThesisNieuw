import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtaidistance import dtw

from collections import Counter

from Low_rank_timeseries.util import load_timeseries_and_labels
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.Low_rank_approx.ACA_diag_pivots1 import ACA_diag_pivots

from Low_rank_timeseries.util import load_labels

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

def get_orginal_rows(cp:ClusterProblem, kbest, pivot_indices):

    rows = np.ndarray((kbest[0], cp.cp_size()), np.single)

    for i in range(len(pivot_indices)):
        cp.sample_row(pivot_indices[i], rows[i,:])

    return rows

def apply_threshold_dis(matrix, threshold):
    filtered_matrix = np.where(matrix <= threshold, matrix, -1)
    return filtered_matrix

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
        return np.exp(-1 * np.square(np.divide(distance, gamma)))

def similarity_to_distance(similarity, gamma=None, method="Gaussian"):
    """
    Convert the given similarity values to distance values.
    Supported methods are 'Gaussian'
    :param similarity: array with similarity values, or 1 similarity value
    :param gamma: parameter for the similarity method
    :param method: String that indicated which similarity method to use. The default is Gaussian.
    :return: Array with distance values
    """
    if method is "Gaussian":
        if gamma is None:
            raise Exception('gamma should not be None if the method is Gaussian')
        return gamma * np.sqrt(-1 * np.log(similarity))

###################################### Main Code ######################################

def main():
    dataset_names = [
        "DistalPhalanxOutlineAgeGroup",
        "DistalPhalanxOutlineCorrect",
        "DistalPhalanxTW",
        "Earthquakes",
        "ECG5000",
        "FaceAll",
        "FordA",
        "LargeKitchenAppliances",
        "MedicalImages",
        "MiddlePhalanxOutlineCorrect",
        "PhalangesOutlinesCorrect",
        "ProximalPhalanxOutlineAgeGroup",
        "ProximalPhalanxOutlineCorrect",
        "ProximalPhalanxTW",
        "ShapesAll",
        "SmallKitchenAppliances",
        "StarLightCurves",
        "Strawberry",
        "SwedishLeaf",
        "SyntheticControl",
        "TwoPatterns",
        "Wafer",
        "Yoga"
    ]

    best_K_values = {
        "DistalPhalanxOutlineAgeGroup": 1, 
        "DistalPhalanxOutlineCorrect": 4,
        "DistalPhalanxTW": 7, 
        "Earthquakes": 5, 
        "ECG5000": 3, 
        "FaceAll": 4, 
        "FordA": 7, 
        "LargeKitchenAppliances": 4,
        "MedicalImages": 1, 
        "MiddlePhalanxOutlineCorrect": 6, 
        "PhalangesOutlinesCorrect": 6, 
        "ProximalPhalanxOutlineAgeGroup": 6, 
        "ProximalPhalanxOutlineCorrect": 3, 
        "ProximalPhalanxTW": 6,
        "ShapesAll": 1, 
        "SmallKitchenAppliances": 5,
        "StarLightCurves": 4,
        "Strawberry": 1, 
        "SwedishLeaf": 1, 
        "SyntheticControl": 1, 
        "TwoPatterns": 1, 
        "Wafer": 1, 
        "Yoga": 1
    }

    results = []

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"

    for name in dataset_names:
        print(f'Working on {name} ...')

        train_path = "C:/Users/robwi/Documents/ThesisClean/Data/" + name + "/" + name + "_TRAIN.tsv"
        labels_org, series_org = load_timeseries_and_labels_from_tsv(train_path)
        size = len(labels_org)

        file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{name}_dtw.npy")
        A = np.load(file_path)
        A_train = A[:size,:size]

        test_path = "C:/Users/robwi/Documents/ThesisClean/Data/" + name + "/" + name + "_TEST.tsv"
        labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

        best_scores = []

        for percentage_sampled in np.arange(0.01, 0.1, 0.02):
            cp = ClusterProblem(labels_org, series_org, "dtw", solved_matrix=A_train, similarity=False)
            tau = 0.001
            Deltak = 20
            kmax = int(percentage_sampled * size)
            _, _, kbest, gamma, pivot_indices = ACA_diag_pivots(cp, tau, kmax, Deltak)

            rows = get_orginal_rows(cp, kbest, pivot_indices)
            rowsdis = np.array([similarity_to_distance(row, gamma) for row in rows])

            best_score = 0
            best_sim_threshold = 0
            best_dis_threshold = 0

            # Retrieve the best K value for the current dataset
            k_nb = best_K_values.get(name, 1)

            for sim_threshold in np.arange(0.4, 1, 0.02):
                dis_threshold = similarity_to_distance(sim_threshold, gamma)[0]

                filtered_rows = apply_threshold_dis(rowsdis, dis_threshold)

                non_zero_indices_per_row = [np.where(row != -1)[0] for row in filtered_rows]

                labels_count_per_row = []
                for non_zero_indices in non_zero_indices_per_row:
                    labels_for_this_row = [labels_org[idx] for idx in non_zero_indices]
                    label_counts = Counter(labels_for_this_row)
                    labels_count_per_row.append(label_counts)

                correct_predictions = 0
                error_occurred = False
                for i in range(len(series_test)):
                    T_new = series_test[i]
                    di_measures = []
                    for idx in pivot_indices:
                        di = A[idx, size + i]
                        di_measures.append(di)

                    di_measures = np.array(di_measures)
                    sorted_indices = np.argsort(di_measures)

                    j = 0
                    new_dict = Counter()
                    total_count = 0
                    error_occurred = False
                    while k_nb > total_count:
                        try:
                            index = sorted_indices[j]
                            count = labels_count_per_row[index]
                            new_dict += count
                            total_count = sum(new_dict.values())
                            j += 1
                        except IndexError:
                            print(f'IndexError occurred for k_nb: {k_nb}, sim_threshold: {sim_threshold}. Skipping this iteration.')
                            error_occurred = True
                            break

                    if error_occurred:
                        correct_predictions = 0
                        break

                    most_common_label = new_dict.most_common(1)[0][0]
                    if most_common_label == labels_test[i]:
                        correct_predictions += 1

                custom_method_accuracy = correct_predictions / len(series_test)
                print(f'Percentage Sampled: {percentage_sampled}, k_nb: {k_nb}, Similarity Threshold: {sim_threshold}, Distance Threshold: {dis_threshold}, Custom Method Accuracy: {custom_method_accuracy}')

                if custom_method_accuracy > best_score:
                    best_score = custom_method_accuracy
                    best_sim_threshold = sim_threshold
                    best_dis_threshold = dis_threshold

            best_scores.append((percentage_sampled, best_score, k_nb, best_sim_threshold, best_dis_threshold))

        results.append((name, best_scores))

    # Save results to Excel
    excel_data = []
    for result in results:
        name, best_scores = result
        for score_data in best_scores:
            percentage_sampled, best_score, best_k, best_sim_threshold, best_dis_threshold = score_data
            excel_data.append([name, percentage_sampled, best_score, best_k, best_sim_threshold, best_dis_threshold])

    df_results = pd.DataFrame(excel_data, columns=['Dataset Name', 'Percentage Sampled', 'Best Score', 'Best K', 'Best Sim Threshold', 'Best Dis Threshold'])
    output_file = os.path.join(base_dir, "Best_Scores_Datasets_ModelBased_Final.xlsx")
    df_results.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
