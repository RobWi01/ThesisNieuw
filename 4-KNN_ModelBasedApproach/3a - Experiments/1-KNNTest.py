import os
import sys
import numpy as np

## Using the Python version
sys.path.append('C:/Users/robwi/Documents/ThesisFinal/4-KNN_ModelBasedApproach')  # Change this when uploading to gitlab, working with relative imports?
from KNN_ModelBased2 import knn_aca_model_based

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

def calculate_accuracy(true_labels, predicted_labels):
    correct_predictions = (true_labels == predicted_labels)
    accuracy = np.mean(correct_predictions)  # Calculate the proportion of correct predictions
    return accuracy

###################################### Main Code ######################################

def main():
    dataset_names = [
        'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown', 'Coffee', 
        'CricketX', 'CricketZ', 'CinCECGTorso', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 
        'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Computers', 'Earthquakes', 'ECG200', 
        'ECG5000', 'FaceAll', 'FacesUCR', 'Fish', 'FordA', 'Fungi', 'FreezerRegularTrain', 
        'FreezerSmallTrain', 'GunPoint', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 
        'GunPointAgeSpan', 'HandOutlines', 'HouseTwenty', 'InsectEPGSmallTrain', 'InsectEPGRegularTrain', 
        'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Mallat', 'Meat', 'MedicalImages', 
        'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 
        'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 
        'PhalangesOutlinesCorrect', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 
        'SemgHandSubjectCh2', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface2', 
        'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 
        'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'Wafer', 'Yoga'
    ]

    K_nb = 7
    sim_threshold_within_skeletons = 0  # Example value, adjust as needed
    sim_threshold_between_skeletons_Tnew = 0.5  # Example value, adjust as needed

    results = []

    for name in dataset_names:
        try:
            print(f"Processing dataset: {name}")
            predictions = knn_aca_model_based(name, K_nb, sim_threshold_within_skeletons, sim_threshold_between_skeletons_Tnew)
            test_path = f"C:/Users/robwi/Documents/ThesisClean/Data/{name}/{name}_TEST.tsv"
            labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

            # Calculate and print the accuracy
            if predictions is not None and labels_test is not None:
                accuracy = calculate_accuracy(labels_test, predictions)
                print(f"{name} Accuracy: {accuracy * 100:.2f}%")
                results.append((name, accuracy))
            else:
                print(f"Failed to load data or predictions were not provided for {name}.")
                results.append((name, None))
        except Exception as e:
            print(f"Error processing dataset {name}: {e}")
            results.append((name, None))

    # Save results to a CSV file
    results_df = np.array(results)
    np.savetxt("knn_aca_model_based_results.csv", results_df, delimiter=",", fmt='%s', header="Dataset,Accuracy", comments='')

if __name__ == "__main__":
    main()
