import numpy as np
from Low_rank_timeseries.util import load_matrix, load_timeseries, load_labels, get_amount_of_classes
from Low_rank_timeseries.Data_paths import matrix_data_folder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample

def save_sampled_distance_and_similarity_matrix(data_name, distance_name, sample_size, save_path, upsample=False):
    series = load_timeseries(data_name)
    labels = load_labels(data_name)
    print(matrix_data_folder)
    
    similarity_matrix = load_matrix(data_name, distance_name, Distance=False)

    nr_labels = get_amount_of_classes(labels)
    len_full = np.shape(labels)[0]
    proportions_full = np.ndarray(nr_labels)

    for label in range(nr_labels):
        proportions_full[label] = (np.shape(np.where(labels == label + 1)[0])[0]) / len_full

    if upsample and sample_size > len_full:
        train_indices = np.arange(len_full)
        train_indices = resample(train_indices, n_samples=sample_size, replace=True, stratify=labels)
    else:
        skf = StratifiedShuffleSplit(n_splits=1, train_size=sample_size)
        split = skf.split(series, labels)
        proportions_subset = np.ndarray(nr_labels)
        
        for i, (train_indices, test_indices) in enumerate(split):
            train_labels = labels[train_indices]

            for label in range(nr_labels):
                proportions_subset[label] = (np.shape(np.where(train_labels == label + 1)[0])[0]) / sample_size

        print("Proportions full")
        print(proportions_full)
        print("Proportions subset")
        print(proportions_subset)

    sampled_similarity = similarity_matrix[train_indices, :][:, train_indices]

    similarity_path = f"{save_path}/{data_name}_{sample_size}_{distance_name}.npy"

    np.save(similarity_path, sampled_similarity)

# Example usage
name = "Crop"
Distance = "dtw"
save_path = "path_to_save_folder"
sizes = [500, 1000, 5000, 10000, 15000, 20000, 24000]

for size in sizes:
    save_sampled_distance_and_similarity_matrix(name, Distance, size, save_path, upsample=True)
