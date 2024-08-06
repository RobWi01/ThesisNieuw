###################################### Imports ######################################
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dtaidistance import dtw
from collections import Counter

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/2-KNN_ACA_Naive') # Change this when uploading to gitlab, working with relative imports?
from KNN_Exact import Get_KNN_Using_Dis_Matrix


from Low_rank_timeseries.util import load_timeseries_and_labels
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.Low_rank_approx.ACA_diag_pivots import ACA_diag_pivots
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix
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
            # di2 = dtw.distance(T[indices[i]], Tnew)
            # print(di)
            # print(di2)
            wi_new[t-1] = di - approx
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

###################################### Main Code ######################################

def main():
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

    results = []

    for name in dataset_names:
        try:
            train_path = f"C:/Users/robwi/Documents/ThesisClean/Data/{name}/{name}_TRAIN.tsv"
            labels_train, series_train = load_timeseries_and_labels_from_tsv(train_path)
            cp = ClusterProblem(labels_train, series_train, "dtw", similarity=False)  # dtw satisfies triangle inequality

            print(f"Processing dataset: {name}")

            tau = 0.001
            kmax = int(0.05 * len(labels_train))
            Deltak = 200
            W, inv_deltas, _, gamma, pivot_indices = ACA_diag_pivots(cp, tau, kmax, Deltak)
            size = W.shape[1]

            test_path = f"C:/Users/robwi/Documents/ThesisClean/Data/{name}/{name}_TEST.tsv"
            labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

            # Directly specify the absolute path for clarity and reliability
            base_dir = "C:/Users/robwi/Documents/ThesisFinal/"
            file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{name}_dtw.npy")
            
            labels = load_labels(name)

            size = len(series_train)
            A = np.load(file_path)

            best_accuracy = 0
            best_k_nb = best_K_values[name]

            correct_predictions = 0
            total_error = 0

            absolute_errors = []

            for i in range(len(series_test)):  # Sample new time series
                T_new = series_test[i]
                new_W = update_skeletons(W, inv_deltas, pivot_indices, series_train, T_new, size, 1, gamma, Precomputed=True, Dis_Mat=A, train_size=size, index=i)
                A_approx = np.zeros((size + 1, size + 1))
                reconstruct_matrix(A_approx, new_W, inv_deltas, Distance=False, do_corrections=True)
                A_approx = np.array([similarity_to_distance(w, gamma) for w in A_approx])

                true_distances = A[size + 0, :size]
                approx_distances = A_approx[size + 0, :size]

                error = np.linalg.norm(true_distances - approx_distances, 2) / np.linalg.norm(true_distances, 2)
                total_error += error

                absolute_errors.append(np.abs(true_distances - approx_distances))  # Collecting absolute errors for plotting

                neighbors = Get_KNN_Using_Dis_Matrix(A_approx, size, 0, best_k_nb)

                # Retrieve the labels for the nearest neighbors
                neighbor_labels = labels[neighbors]

                # Use Counter to count the occurrences of each label
                label_counts = Counter(neighbor_labels)
                # Determine the majority label
                majority_label = label_counts.most_common(1)[0][0]

                if majority_label == labels_test[i]:
                    correct_predictions += 1

            accuracy = correct_predictions / len(series_test)
            avg_error = total_error / len(series_test)

            print(f'Best k_nb for {name}: {best_k_nb} with accuracy: {accuracy}, Avg Error: {avg_error}')
            results.append((name, best_accuracy, best_k_nb, avg_error))

            # # Plotting absolute errors for each test instance
            # absolute_errors = np.array(absolute_errors)
            # plt.figure(figsize=(10, 6))
            # for i in range(absolute_errors.shape[1]):
            #     plt.plot(absolute_errors[:, i], label=f'Test Instance {i}')
            # plt.title(f'Absolute Difference Error for {name} (ACA Update)')
            # plt.xlabel('Test Instance Index')
            # plt.ylabel('Absolute Difference Error')
            # plt.show()

        except Exception as e:
            print(f"Error processing dataset {name}: {e}")
            results.append((name, None, None, None))

    # Save results to a CSV file
    df_results = pd.DataFrame(results, columns=['Dataset Name', 'Accuracy', 'Best_k_nb', 'Avg Error'])
    output_file = os.path.join(base_dir, "results3.csv")
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
