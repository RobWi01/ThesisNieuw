####################################################### Imports #######################################################

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from Low_rank_timeseries.util import load_labels


from warnings import simplefilter

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/2-KNN_ACA_Naive') # Change this when uploading to gitlab, working with relative imports?
from KNN_Exact import Get_KNN_Using_Dis_Matrix

from collections import Counter

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


if __name__ == "__main__":
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

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    compare_name = "dtw"

    results = []

    # run block of code and catch warnings
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        K_values = range(1, 8)

        for data_name in dataset_names:

            # Problem here is that you don't know K before hand, I use the cross validation for this now

            print(f'Working on {data_name} ...' )

            # Directly specify the absolute path for clarity and reliability
            data_dir = "C:/Users/robwi/Documents/ThesisFinal/Data/"
            file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{data_name}_dtw.npy")

            train_path = data_dir + data_name + "/" + data_name + "_TRAIN.tsv"
            test_path = data_dir + data_name + "/" + data_name + "_TEST.tsv"

            _, series_train = load_timeseries_and_labels_from_tsv(train_path)
            labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

            labels = load_labels(data_name)

            size = len(series_train)

            A = np.load(file_path)

            best_accuracy = 0
            best_K = None

            for K in K_values:
                correct_predictions = 0

                for index in range(0, len(series_test)):

                    neighbors = Get_KNN_Using_Dis_Matrix(A, size, index, K)

                    # Retrieve the labels for the nearest neighbors
                    neighbor_labels = labels[neighbors]

                    # Use Counter to count the occurrences of each label
                    label_counts = Counter(neighbor_labels)
                    # Determine the majority label
                    majority_label = label_counts.most_common(1)[0][0]

                    if majority_label == labels[size + index]:
                        correct_predictions += 1

                accuracy = correct_predictions / len(labels_test)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_K = K

            print(f'Best K for {data_name}: {best_K}, Accuracy: {best_accuracy}')
            if best_accuracy > 0.50:
                results.append((data_name, best_accuracy, best_K))

    if results:
        df_results = pd.DataFrame(results, columns=['Dataset Name', 'Score', 'Best K'])
        output_file = os.path.join(base_dir, "BestScoresFinal.xlsx")
        df_results.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No dataset had a score greater than 0.50")
