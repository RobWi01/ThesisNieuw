import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from collections import Counter
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/2-KNN_ACA_Naive') # Change this when uploading to gitlab, working with relative imports?
from KNN_Exact import Get_KNN_Using_Dis_Matrix

from Low_rank_timeseries.util import load_labels

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

def add_gaussian_noise(matrix, mean_error, std_error):
    """Add Gaussian noise to the distance matrix based on the given mean and standard deviation."""
    noise = np.random.normal(mean_error, std_error, matrix.shape)
    return matrix + noise

def calculate_mape(true_values, approx_values):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((true_values - approx_values) / true_values)) * 100

def analyze_error_impact(true_distances, approx_distances, labels, k):
    sorted_indices_true = np.argsort(true_distances)
    sorted_indices_approx = np.argsort(approx_distances)

    neighbors_true = sorted_indices_true[:k]
    neighbors_approx = sorted_indices_approx[:k]

    overlap = len(set(neighbors_true).intersection(set(neighbors_approx)))
    accuracy_impact = overlap / k

    return accuracy_impact, neighbors_true, neighbors_approx

def main():
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
        "Yoga": 1
    }

    # Load the original distance matrix and labels
    data_name = "CBF"
    base_dir = "C:/Users/robwi/Documents/ThesisFinal/"
    file_path = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{data_name}_dtw.npy")
    distance_matrix = np.load(file_path)
    labels = load_labels(data_name)

    data_dir = "C:/Users/robwi/Documents/ThesisFinal/Data/"
    train_path = data_dir + data_name + "/" + data_name + "_TRAIN.tsv"
    test_path = data_dir + data_name + "/" + data_name + "_TEST.tsv"

    _, series_train = load_timeseries_and_labels_from_tsv(train_path)
    labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

    size = len(series_train)
    best_K = best_K_values[data_name]
    total_error = 0
    correct_predictions = 0

    mean_error = 0.9018192294143058
    std_error = 9
    offset = 6

    all_errors = []
    absolute_errors = []
    mape_errors = []  # List to collect MAPE values
    accuracy_impact_list = []
    error_details = []

    for index in range(0, len(series_test)):
        true_distances = distance_matrix[size + index, :size]
        approx_distances = add_gaussian_noise(true_distances, mean_error, std_error) + offset

        neighbors_true = Get_KNN_Using_Dis_Matrix(distance_matrix, size, index, best_K)
        sorted_indices = np.argsort(approx_distances)
        neighbors_approx = sorted_indices[:best_K]

        error = np.linalg.norm(true_distances - approx_distances, 2) / np.linalg.norm(true_distances, 2)
        total_error += error

        all_errors.append(error)  # Collecting all errors for analysis
        absolute_errors.append(np.abs(true_distances - approx_distances))  # Collecting absolute errors for plotting

        # Calculate MAPE
        mape = calculate_mape(true_distances, approx_distances)
        mape_errors.append(mape)  # Collect MAPE for each test instance

        accuracy_impact, neighbors_true, neighbors_approx = analyze_error_impact(true_distances, approx_distances, labels, best_K)
        accuracy_impact_list.append(accuracy_impact)

        # Retrieve the labels for the nearest neighbors
        neighbor_labels = labels[neighbors_approx]

        # Use Counter to count the occurrences of each label
        label_counts = Counter(neighbor_labels)
        # Determine the majority label
        majority_label = label_counts.most_common(1)[0][0]

        if majority_label == labels[size + index]:
            correct_predictions += 1

        # Store error details for each point
        error_details.append({
            'index': index,
            'mae': np.mean(np.abs(true_distances - approx_distances)),
            'mse': np.mean((true_distances - approx_distances) ** 2),
            'rmse': np.sqrt(np.mean((true_distances - approx_distances) ** 2)),
            'mape': mape,
            'accuracy_impact': accuracy_impact,
            'neighbors_true': neighbors_true,
            'neighbors_approx': neighbors_approx
        })

    accuracy = correct_predictions / len(labels_test)
    avg_error = total_error / len(series_test)
    avg_mape = np.mean(mape_errors)  # Calculate average MAPE

    print(f"Accuracy: {accuracy}")
    print(f"Average Relative Error: {avg_error}")
    print(f"Average MAPE: {avg_mape}%")
    print(f"Average Accuracy Impact: {np.mean(accuracy_impact_list)}")

    # Plot accuracy impact histogram
    plt.hist(accuracy_impact_list, bins=20, alpha=0.75)
    plt.title("Impact of Approximation Errors on KNN Accuracy")
    plt.xlabel("Accuracy Impact")
    plt.ylabel("Frequency")
    plt.show()

    # Analyze the error distribution
    all_errors = np.array(all_errors)
    plt.hist(all_errors, bins=50, alpha=0.75)
    plt.title("Distribution of Relative Errors")
    plt.xlabel("Relative Error")
    plt.ylabel("Frequency")
    plt.show()

    print(f'Mean Error: {np.mean(all_errors)}')
    print(f'Median Error: {np.median(all_errors)}')
    print(f'Standard Deviation of Error: {np.std(all_errors)}')

    # Plotting absolute errors for each test instance
    absolute_errors = np.array(absolute_errors)
    plt.figure(figsize=(10, 6))
    for i in range(absolute_errors.shape[1]):
        plt.plot(absolute_errors[:, i], label=f'Test Instance {i}')
    plt.title(f'Absolute Difference Error for {data_name}')
    plt.xlabel('Test Instance Index')
    plt.ylabel('Absolute Difference Error')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
    plt.show()

    # Calculate and plot overlap ratio of KNN
    df_error_details = pd.DataFrame(error_details)
    overlap_ratios = []
    for row in df_error_details.itertuples():
        overlap_ratio = len(set(row.neighbors_true).intersection(set(row.neighbors_approx))) / best_K
        overlap_ratios.append(overlap_ratio)

    df_error_details['overlap_ratio'] = overlap_ratios

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='index', y='overlap_ratio', data=df_error_details)
    plt.title(f'Overlap Ratio of KNN for {data_name}')
    plt.xlabel('Test Instance Index')
    plt.ylabel('Overlap Ratio')
    plt.show()

if __name__ == "__main__":
    main()
