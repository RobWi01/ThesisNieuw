####################################################### Imports #######################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error

from collections import Counter
from warnings import simplefilter

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/2-KNN_ACA_Naive') # Change this when uploading to gitlab, working with relative imports?
from KNN_Exact import Get_KNN_Using_Dis_Matrix

from Low_rank_timeseries.util import create_cluster_problem, load_timeseries, load_labels
from Low_rank_timeseries.Low_rank_approx.ACA_diag import ACA_diag_pivots
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix

####################################################### Helper Functions #######################################################

def get_amount_of_classes(labels):
    return len(np.unique(labels))

def calculate_accuracy(true_labels, predicted_labels):
    correct_predictions = (true_labels == predicted_labels)
    accuracy = np.mean(correct_predictions)  # Calculate the proportion of correct predictions
    return accuracy

def load_timeseries_and_labels_from_tsv(path):
    if os.path.isfile(path):
        data = np.genfromtxt(path, delimiter='\t')
        labels, series = data[:, 0], data[:, 1:]
        labels = np.array(labels, np.single)
        series = np.array(series, np.single)
    else:
        labels = None
        series = None

    return labels, series

def calculate_mape(true_values, approx_values):
    return np.mean(np.abs((true_values - approx_values) / true_values)) * 100

def calculate_smape(true_values, approx_values):
    return 100 * np.mean(2 * np.abs(true_values - approx_values) / (np.abs(true_values) + np.abs(approx_values)))

# def analyze_error_impact(true_distances, approx_distances, labels, k):
#     sorted_indices_true = np.argsort(true_distances)
#     sorted_indices_approx = np.argsort(approx_distances)

#     neighbors_true = sorted_indices_true[:k]
#     neighbors_approx = sorted_indices_approx[:k]

#     overlap = len(set(neighbors_true).intersection(set(neighbors_approx)))
#     accuracy_impact = overlap / k

#     mae = mean_absolute_error(true_distances, approx_distances)
#     mse = mean_squared_error(true_distances, approx_distances)
#     rmse = np.sqrt(mse)
#     mape = calculate_mape(true_distances, approx_values)
#     smape = calculate_smape(true_distances, approx_values)

#     return accuracy_impact, neighbors_true, neighbors_approx, mae, mse, rmse, mape, smape

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
    
if __name__ == "__main__":
    dataset_names = [
        "BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown", "Coffee", "CricketX", "CricketZ", "CinCECGTorso",
        "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
        "Computers", "Earthquakes", "ECG200", "ECG5000", "FaceAll", "FacesUCR", "Fish", "FordA", "Fungi", 
        "FreezerRegularTrain", "FreezerSmallTrain", "GunPoint", "GunPointMaleVersusFemale", "GunPointOldVersusYoung",
        "GunPointAgeSpan", "HandOutlines", "HouseTwenty", "InsectEPGSmallTrain", "InsectEPGRegularTrain",
        "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Mallat", "Meat", "MedicalImages", 
        "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain", "MixedShapesSmallTrain", "MoteStrain",
        "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil", "PhalangesOutlinesCorrect", "Plane",
        "PowerCons", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", 
        "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2", "ShapesAll", "SmallKitchenAppliances",
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
    error_metrics = []
    error_details = []
    absolute_errors = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for data_name in dataset_names:
            print(f'Working on {data_name} ...')

            data_dir = "C:/Users/robwi/Documents/ThesisFinal/Data/"
            file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{data_name}_dtw.npy")

            train_path = data_dir + data_name + "/" + data_name + "_TRAIN.tsv"
            test_path = data_dir + data_name + "/" + data_name + "_TEST.tsv"

            _, series_train = load_timeseries_and_labels_from_tsv(train_path)
            labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

            A = np.load(file_path)

            labels = load_labels(data_name)

            size = len(series_train)

            best_accuracy = 0
            best_K = best_K_values[data_name]

            cp = create_cluster_problem(data_name, compare_name, no_solved_matrix=False, Distance=False, include_series=False)

            tau = 0.001
            kmax = int(0.02 * len(labels))
            Deltak = 20
            W, deltas, kbest, gamma = ACA_diag_pivots(cp, tau, kmax, Deltak)

            A_approx = np.zeros((len(labels), len(labels)))
            reconstruct_matrix(A_approx, W, deltas, Distance=False, do_corrections=True)
            A_approx = np.array([similarity_to_distance(w, gamma) for w in A_approx])

            correct_predictions = 0
            total_mae = 0
            total_mse = 0
            total_rmse = 0
            total_mape = 0
            total_smape = 0
            total_error = 0
            accuracy_impact_list = []

            for index in range(0, len(series_test)):
                neighbors_true = Get_KNN_Using_Dis_Matrix(A, size, index, best_K)
                neighbors_approx = Get_KNN_Using_Dis_Matrix(A_approx, size, index, best_K)

                true_distances = A[size + index, :size]
                approx_distances = A_approx[size + index, :size]

                # accuracy_impact, neighbors_true, neighbors_approx, mae, mse, rmse, mape, smape = analyze_error_impact(true_distances, approx_distances, labels, best_K)

                # total_mae += mae
                # total_mse += mse
                # total_rmse += rmse
                # total_mape += mape
                # total_smape += smape

                error = np.linalg.norm(true_distances - approx_distances, 2) / np.linalg.norm(true_distances, 2)
                total_error += error

                # accuracy_impact_list.append(accuracy_impact)

                # Collecting absolute errors for plotting
                absolute_errors.append(np.abs(true_distances - approx_distances))

                # error_details.append({
                #     'dataset': data_name,
                #     'index': index,
                #     'mae': mae,
                #     'mse': mse,
                #     'rmse': rmse,
                #     'mape': mape,
                #     'smape': smape,
                #     'accuracy_impact': accuracy_impact,
                #     'neighbors_true': neighbors_true,
                #     'neighbors_approx': neighbors_approx
                # })

                neighbor_labels = labels[neighbors_approx]
                label_counts = Counter(neighbor_labels)
                majority_label = label_counts.most_common(1)[0][0]

                if majority_label == labels[size + index]:
                    correct_predictions += 1

            accuracy = correct_predictions / len(labels_test)
            avg_mae = total_mae / len(series_test)
            avg_mse = total_mse / len(series_test)
            avg_rmse = total_rmse / len(series_test)
            avg_mape = total_mape / len(series_test)
            avg_smape = total_smape / len(series_test)
            avg_error = total_error / len(series_test)

            print(accuracy)

            error_metrics.append((data_name, avg_mae, avg_mse, avg_rmse, avg_mape, avg_smape))

            if accuracy >= 0:
                results.append((data_name, accuracy, best_K, avg_error))

    # Save results to a CSV file
    df_results = pd.DataFrame(results, columns=['Dataset Name', 'Accuracy', 'Best_k_nb', 'Avg_Error'])
    output_file = os.path.join(base_dir, "results_KNN_good.csv")
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
