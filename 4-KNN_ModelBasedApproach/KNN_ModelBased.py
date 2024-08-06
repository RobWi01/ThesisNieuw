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

def apply_threshold(matrix, threshold):
    filtered_matrix = np.where(matrix >= threshold, matrix, 0)
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
        "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Mallat", "Meat", "MedicalImages", 
        "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain", "MixedShapesSmallTrain", "MoteStrain",
        "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil", "PhalangesOutlinesCorrect", "Plane",
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

    name = 'CBF' # Change this to use a different dataset
    sim_threshold = 0.9
    k_nb = 4

    # Load training data
    train_path =    "C:/Users/robwi/Documents/ThesisClean/Data/" + name + "/" + name + "_TRAIN.tsv"
    labels_org, series_org = load_timeseries_and_labels_from_tsv(train_path)

    # Initialize and process ClusterProblem
    cp = ClusterProblem(labels_org, series_org, "dtw", similarity=False)
    tau = 0.001
    kmax = 200
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

    #TO-DO: Check sample rate after I get these original rows

    filtered_rows = apply_threshold(rows, sim_threshold)


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
            di = dtw.distance(series_org[idx], T_new, use_pruning=True)  # Added use_pruning for efficiency
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
