import os
import sys
import time
import numpy as np

from sklearn.metrics import adjusted_rand_score

from Low_rank_timeseries.util import load_labels

from SpecNystrom import nystrom




if __name__ == '__main__':

    data_name = "Symbols"
    compare_name = "dtw"

    num_samples = 1019    # Number of random samples for Nystrom method
    num_clusters = 6 # Number of clusters


    labels = load_labels(data_name)

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    file_path = os.path.join(base_dir, "Matrices", "Similarity_matrices", f"{data_name}_dtw.npy")
    S = np.load(file_path)

    predicted_labels, evd_time, kmeans_time, total_time = nystrom(num_samples, num_clusters, S)

    ARIscore = adjusted_rand_score(predicted_labels, labels)

    print(ARIscore)

