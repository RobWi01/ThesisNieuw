###################################### Imports ######################################
import os
import time
import numpy as np

from dtaidistance import dtw
from collections import Counter

from Low_rank_timeseries.util import load_timeseries_and_labels
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.Low_rank_approx.ACA_diag_pivots import ACA_diag_pivots
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

###################################### Main Code ######################################

def main():
    dataset_names = [
            'HouseTwenty', 'InsectEPGSmallTrain', 'InsectEPGRegularTrain', 
        'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Mallat', 'Meat', 'MedicalImages', 
        'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 
        'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 
        'PhalangesOutlinesCorrect', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup' 
    ]

        #     'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown', 'Coffee', 
        # 'CricketX', 'CricketZ', 'CinCECGTorso', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 
        # 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Computers', 'Earthquakes', 'ECG200', 
        # 'ECG5000', 'FaceAll', 'FacesUCR', 'Fish', 'FordA', 'Fungi', 'FreezerRegularTrain', 
        # 'FreezerSmallTrain', 'GunPoint', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 
        # 'GunPointAgeSpan'



    # 'HouseTwenty', 'InsectEPGSmallTrain', 'InsectEPGRegularTrain', 
    #     'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Mallat', 'Meat', 'MedicalImages', 
    #     'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 
    #     'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 
    #     'PhalangesOutlinesCorrect', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 
    #     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 
    #     'SemgHandSubjectCh2', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface2', 
    #     'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 
    #     'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'Wafer', 'Yoga'

    results = []

    for name in dataset_names:
        try:
            train_path = f"C:/Users/robwi/Documents/ThesisClean/Data/{name}/{name}_TRAIN.tsv"
            labels_train, series_train = load_timeseries_and_labels_from_tsv(train_path)
            cp = ClusterProblem(labels_train, series_train, "dtw", similarity=False) # dtw satisfies triangle inequality

            print(f"Processing dataset: {name}")

            tau = 0.001
            kmax = int(0.05*len(labels_train))
            Deltak = 200
            W, inv_deltas, _, gamma, pivot_indices = ACA_diag_pivots(cp, tau, kmax, Deltak)
            size = W.shape[1]

            test_path = f"C:/Users/robwi/Documents/ThesisClean/Data/{name}/{name}_TEST.tsv"
            labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

            # Directly specify the absolute path for clarity and reliability
            base_dir = "C:/Users/robwi/Documents/ThesisFinal/"
            file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{name}_dtw.npy")

            size = len(series_train)
            A = np.load(file_path)

            best_accuracy = 0
            best_k_nb = 1

            for k_nb in range(1, 8):
                correct_predictions = 0
                for i in range(len(series_test)):  # Sample new time series
                    T_new = series_test[i]
                    new_W = update_skeletons(W, inv_deltas, pivot_indices, series_train, T_new, size, 1, gamma, Precomputed=True, Dis_Mat=A, train_size=size, index=i)
                    A_approx = np.zeros((size+1, size+1))
                    reconstruct_matrix(A_approx, new_W, inv_deltas, Distance=False, do_corrections=False)
                    distances = A_approx[size, :size]
                    sorted_indices = np.argsort(distances)
                    top_indices = sorted_indices[:k_nb]
                    extracted_labels = [labels_train[idx] for idx in top_indices]
                    label_counts = Counter(extracted_labels)
                    most_common_label = label_counts.most_common(1)[0][0]

                    if most_common_label == labels_test[i]:
                        correct_predictions += 1

                accuracy = correct_predictions / len(series_test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_k_nb = k_nb

                print(f'k_nb: {k_nb}, Accuracy: {accuracy}')

            print(f'Best k_nb for {name}: {best_k_nb} with accuracy: {best_accuracy}')
            results.append((name, best_accuracy, best_k_nb))
        except Exception as e:
            print(f"Error processing dataset {name}: {e}")
            results.append((name, None, None))

    # Save results to a CSV file
    results_df = np.array(results)
    np.savetxt("results2.csv", results_df, delimiter=",", fmt='%s', header="Dataset,Accuracy,Best_k_nb", comments='')

if __name__ == "__main__":
    main()