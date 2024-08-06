####################################################### Imports #######################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from Low_rank_timeseries.util import load_labels

from warnings import simplefilter

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/2-KNN_ACA_Naive') # Change this when uploading to gitlab, working with relative imports?
from KNN_Exact import Get_KNN_Using_Dis_Matrix

from dtaidistance import dtw
from collections import Counter

from Low_rank_timeseries.util import create_cluster_problem, load_matrix, load_timeseries, load_svd
from Low_rank_timeseries.Low_rank_approx.ACA_diag_pivots import ACA_diag_pivots
from Low_rank_timeseries.Data_paths import matrix_data_folder
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.util import load_labels
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix
from Low_rank_timeseries.util import load_labels

####################################################### Helper Functions #######################################################

def get_amount_of_classes(labels):
    return len(np.unique(labels))

def calculate_accuracy(true_labels, predicted_labels):
    correct_predictions = (true_labels == predicted_labels)
    accuracy = np.mean(correct_predictions)  # Calculate the proportion of correct predictions
    return accuracy

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


def update_skeletons(W, inv_deltas, indices, T, Tnew, n, b, gamma, Precomputed = False, Dis_Mat = None, train_size = 0, index = 0):
    k = W.shape[0]  # Number of skeletons
    size = W.shape[1]
    new_W = np.zeros((k, size+1))
    
    for i in range(k):
        wi = W[i]
        wi_new = np.zeros(n + b)  # Create a new array for the updated skeleton
        wi_new[:n] = wi  # Copy existing values into the new array

        # Calculate each new value of the skeleton.
        t = n + 1
        approx = 0

        # Add the approximation of the previous skeletons.
        for j in range(i):
            approx += new_W[j][indices[i]] * new_W[j][t-1] * inv_deltas[j]  # inv_deltas needs to be defined or passed as an argument

        # Double check this math
        if Precomputed:  
            di = Dis_Mat[indices[i]][train_size + index]
            sim = distance_to_similarity(di, gamma=gamma, method="Gaussian")
            # print("sim:", sim)
            # print("approx:", approx)
            # print("test:", (sim[0] - approx))
            # print("wi_new:", wi_new[i])
            wi_new[t-1] = sim[0] - approx
            new_W[i] = wi_new  # Update the skeleton in the list
        else:
            # How to decide which time series to use here
            di = dtw.distance(T[indices[i]], Tnew)
            wi_new[t-1] = di - approx
            new_W[i] = wi_new  # Update the skeleton in the list

    return new_W

def similarity_to_distance(similarity, gamma=None, method="Gaussian"):
    """
    Convert the given similarity values to distance values.
    Supported methods are 'Gaussian'
    :param similarity: array with similarity values, or 1 similarity value
    :param gamma: parameter for the similarity method
    :param method: String that indicated which similarity method to use. The default is Gaussian.
    :return: Array with distance values
    """
    if method == "Gaussian":
        if gamma is None:
            raise Exception('gamma should not be None if the method is Gaussian')
        return gamma * np.sqrt(-1 * np.log(similarity))

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

###################################### Main Code ######################################

if __name__ == "__main__":
    dataset_names = [
         "OliveOil", "PhalangesOutlinesCorrect", "Plane",
        "PowerCons", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", 
         "ShapesAll", "SmallKitchenAppliances",
        "SmoothSubspace", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
        "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "Wafer", "Yoga"
    ]

    best_K_values = {
        "BeetleFly": 4, "BirdChicken": 1, "BME": 1, "Car": 1, "CBF": 4, "Chinatown": 1, "Coffee": 1, "CricketX": 1, "CricketZ": 5,
        "CinCECGTorso": 1, "DiatomSizeReduction": 1, "DistalPhalanxOutlineAgeGroup": 1, "DistalPhalanxOutlineCorrect": 4,
        "DistalPhalanxTW": 7, "Computers": 4, "Earthquakes": 5, "ECG200": 3, "ECG5000": 3, "FaceAll": 4, "FacesUCR": 1, "Fish": 1,
        "FordA": 7, "Fungi": 1, "FreezerRegularTrain": 1, "FreezerSmallTrain": 6, "GunPoint": 1, "GunPointMaleVersusFemale": 1,
        "GunPointOldVersusYoung": 1, "GunPointAgeSpan": 3, "HandOutlines": 7, "HouseTwenty": 7, "InsectEPGSmallTrain": 1,
        "InsectEPGRegularTrain": 1, "ItalyPowerDemand": 6, "LargeKitchenAppliances": 4, "Lightning2": 1, "Mallat": 1, "Meat": 1,
        "MedicalImages": 1, "MiddlePhalanxOutlineCorrect": 6, "MiddlePhalanxTW": 7, "MixedShapesRegularTrain": 1, "MixedShapesSmallTrain": 1,
        "MoteStrain": 1, "NonInvasiveFetalECGThorax1": 4, "NonInvasiveFetalECGThorax2": 4, "OliveOil": 1, "PhalangesOutlinesCorrect": 6,
        "Plane": 1, "PowerCons": 1, "ProximalPhalanxOutlineAgeGroup": 6, "ProximalPhalanxOutlineCorrect": 3, "ProximalPhalanxTW": 6,
        "SemgHandGenderCh2": 6, "SemgHandMovementCh2": 4, "SemgHandSubjectCh2": 1, "ShapesAll": 1, "SmallKitchenAppliances": 5,
        "SmoothSubspace": 7, "SonyAIBORobotSurface2": 1, "StarLightCurves": 4, "Strawberry": 1, "SwedishLeaf": 1, "Symbols": 1,
        "SyntheticControl": 1, "ToeSegmentation1": 1, "ToeSegmentation2": 7, "Trace": 1, "TwoLeadECG": 1, "TwoPatterns": 1, "Wafer": 1,
        "Yoga": 1, 
    }


    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    compare_name = "dtw"

    results = []

    # run block of code and catch warnings
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        for data_name in dataset_names:
            print(f'Working on {data_name} ...')

            # Directly specify the absolute path for clarity and reliability
            data_dir = "C:/Users/robwi/Documents/ThesisFinal/Data/"
            file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{data_name}_dtw.npy")

            train_path = data_dir + data_name + "/" + data_name + "_TRAIN.tsv"
            test_path = data_dir + data_name + "/" + data_name + "_TEST.tsv"

            labels_train, series_train = load_timeseries_and_labels_from_tsv(train_path)
            labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

            A = np.load(file_path)

            labels = load_labels(data_name)

            size = len(series_train)

            best_accuracy = 0
            best_K = best_K_values[data_name]

            cp = ClusterProblem(labels_train, series_train, "dtw", similarity=False)  # dtw satisfies triangle inequality

            tau = 0.001
            kmax = int(0.10 * size)
            Deltak = 20
            W, inv_deltas, kbest, gamma, pivot_indices = ACA_diag_pivots(cp, tau, kmax, Deltak)


            correct_predictions = 0
            total_error = 0

            for index in range(0, len(series_test)):
                T_new = series_test[index]
                new_W = update_skeletons(W, inv_deltas, pivot_indices, series_train, T_new, size, 1, gamma, Precomputed=True, Dis_Mat=A, train_size=size, index=index)
                # A_approx = np.zeros((size + 1, size + 1))
                # reconstruct_matrix(A_approx, new_W, inv_deltas, Distance=False, do_corrections=True)
                # A_approx = np.array([similarity_to_distance(w, gamma) for w in A_approx])

                # Compute the row as explained in the text
                row = compute_row(new_W, inv_deltas, T_new, pivot_indices)
                # print(row)
                row = similarity_to_distance(row, gamma)

                # neighbors = Get_KNN_Using_Dis_Matrix(A_approx, size, 0, best_K)
                true_distances = A[size, :size]
                approx_distances = row[:size]


                error = np.linalg.norm(true_distances - approx_distances, 2) / np.linalg.norm(true_distances, 2)
                total_error += error

                sorted_indices = np.argsort(approx_distances)
                top_indices = sorted_indices[:best_K]
                extracted_labels = [labels_train[idx] for idx in top_indices]
                label_counts = Counter(extracted_labels)
                most_common_label = label_counts.most_common(1)[0][0]

                if most_common_label == labels[size + index]:
                    correct_predictions += 1

            accuracy = correct_predictions / len(labels_test)
            avg_error = total_error / len(series_test)

            print(f'Best K for {data_name}: {best_K}, Accuracy: {accuracy}, Avg Error: {avg_error}')
            if accuracy >= 0:
                results.append((data_name, accuracy, best_K, avg_error))

    if results:
        df_results = pd.DataFrame(results, columns=['Dataset Name', 'Score', 'Best K', 'Avg Error'])
        output_file = os.path.join(base_dir, "High_Score_Datasets_with_K_Updates3.xlsx")
        df_results.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No dataset had a score greater than 0.50")
