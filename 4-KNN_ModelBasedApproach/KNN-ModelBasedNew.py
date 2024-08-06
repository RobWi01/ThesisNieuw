import os
import time
import numpy as np

from dtaidistance import dtw

from collections import Counter

from Low_rank_timeseries.util import load_timeseries_and_labels
from Low_rank_timeseries.util import create_cluster_problem
from Low_rank_timeseries.util import get_amount_of_classes
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.Low_rank_approx.ACA_diag_pivots import ACA_diag_pivots
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix

from Low_rank_timeseries.util import load_labels

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

def apply_threshold_sim(matrix, threshold):
    filtered_matrix = np.where(matrix >= threshold, matrix, 0)
    return filtered_matrix

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
        return np.exp(-1*np.square(np.divide(distance, gamma)))
    
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

    name = 'CBF' # Change this to use a different dataset
    sim_threshold = 0.9

    # Load training data
    train_path =    "C:/Users/robwi/Documents/ThesisClean/Data/" + name + "/" + name + "_TRAIN.tsv"
    labels_org, series_org = load_timeseries_and_labels_from_tsv(train_path)

    # Initialize and process ClusterProblem
    cp = ClusterProblem(labels_org, series_org, "dtw", similarity=False)
    tau = 0.001
    print(len(labels_org))
    kmax = int(0.10*len(labels_org))
    Deltak = 200
    _, _, kbest, gamma, pivot_indices = ACA_diag_pivots(cp, tau, kmax, Deltak)

    # Load test data
    test_path = "C:/Users/robwi/Documents/ThesisClean/Data/" + name + "/" + name + "_TEST.tsv"
    labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

    # Initialize kNN classifier
    # k = 7  # Number of neighbors
    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(series_org, labels_org)

    # # Predict and evaluate for each item in the test set
    # predictions = knn.predict(series_test)
    # accuracy = accuracy_score(labels_test, predictions)
    # print(f'kNN Accuracy: {accuracy}')

    rows = get_orginal_rows(cp, kbest, pivot_indices)
    print(rows)

    #TO-DO: Check sample rate after I get these original rows

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{name}_dtw.npy")
    A = np.load(file_path)

    filtered_rows = apply_threshold_dis(rows, sim_threshold)


    # Find the non-zero indices for each row
    non_zero_indices_per_row = [np.nonzero(row)[0] for row in filtered_rows]

    labels_count_per_row = []
    for non_zero_indices in non_zero_indices_per_row:
        labels_for_this_row = [labels_org[idx] for idx in non_zero_indices]
        label_counts = Counter(labels_for_this_row)
        labels_count_per_row.append(label_counts)


    # Additionally, if you want to perform a similar evaluation using your custom method for all items in the test set:
    correct_predictions = 0
    for i in range(50):
        print('TS number:', i)

        T_new = series_test[i]
        sim_measures = []
        for idx in pivot_indices:
            di = A[idx, size + i]

            # di = dtw.distance(series_org[idx], T_new, use_pruning=True)  # Added use_pruning for efficiency
            sim = distance_to_similarity(di, gamma=gamma, method="Gaussian")
            sim_measures.append(sim[0])

        sim_measures = np.array(sim_measures)
        sorted_indices = np.argsort(-sim_measures)


        j = 0
        new_dict = Counter()
        total_count = 0
        while k_nb > total_count:
            index = sorted_indices[j]
            count = labels_count_per_row[index]  # Assuming labels_count_per_row is pre-computed as before
            new_dict += count
            total_count = sum(new_dict.values())
            j += 1

        most_common_label = new_dict.most_common(1)[0][0]
        if most_common_label == labels_test[i]:
            correct_predictions += 1

    custom_method_accuracy = correct_predictions / 50
    print(f'Custom Method Accuracy: {custom_method_accuracy}')

if __name__ == "__main__":
    main()
