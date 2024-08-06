# Do here a first run with crop
import os
import sys
import time
import numpy as np

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/1-SpectralClustering_ACA_Components') # Change this when uploading to gitlab, working with relative imports?
from LanczosFastMult import lanczosFastMult

from Low_rank_timeseries.util import create_cluster_problem, load_matrix, load_timeseries, load_svd
from Low_rank_timeseries.Low_rank_approx.ACA_diag import ACA_diag_pivots
from Low_rank_timeseries.Data_paths import matrix_data_folder


from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem

from Low_rank_timeseries.util import load_labels


###################################### Helper Functions ######################################

def downsample_matrix_random(matrix, target_rows, target_cols):
    # Randomly select indices for rows
    rows_indices = np.random.choice(matrix.shape[0], target_rows, replace=False)
    # Randomly select indices for columns
    cols_indices = np.random.choice(matrix.shape[1], target_cols, replace=False)
    
    # Select the rows and columns
    downsampled_matrix = matrix[np.ix_(rows_indices, cols_indices)]
    
    return downsampled_matrix

def upsample_matrix_nearest(matrix, factor):
    return np.repeat(np.repeat(matrix, factor, axis=0), factor, axis=1)

def get_amount_of_classes(labels):
    return len(np.unique(labels))

def load_matrix2(data_name, compare_name, Distance=True):
    """
    Returns the stored solved matrix.
    If no solved matrix is found, None is returned
    :param data_name:
    :param compare_name:
    :param Distance: Boolean to indicate if the distance or similarity matrix should be returned
    :return:
    """

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    file_path = os.path.join(base_dir, "Matrices", "Similarity_matrices", f"{data_name}_{20000}dtw.npy")

    if os.path.isfile(file_path):
        solved_matrix = np.array(np.load(file_path), np.single)
    else:
        solved_matrix = None
    return solved_matrix


def create_cluster_problem2(data_name, compare_name, no_solved_matrix=False, include_svd=False, Distance=True,
                           gamma=None, autodiagonal=True, include_series=True):
    """
    returns an instance of the cluster problem class for the dataset with name 'data_name' and with distance function
    'compare_name'
    :param data_name: The name of the dataset
    :param compare_name: The name of the distance function
    :param no_solved_matrix: Boolean to indicate whether to include the solved matrix in the cluster problem
    :param include_svd: Boolean to indicate whether to include the solved svd in the cluster problem
    :param Distance: Boolean to indicate whether the cluster problem should be a distance or similarity cluster problem
    :param gamma: parameter for the conversion to similarity values.
    :param autodiagonal: Boolean to indicate whether the diagonal elements should be automatically assigned
    :return: cluster problem
    """
    if include_series:
        series = load_timeseries(data_name)
        labels = load_labels(data_name)
    else:
        series = None
        labels = None
    if not no_solved_matrix:
        solved_matrix = load_matrix2(data_name, compare_name, Distance=Distance)
    else:
        solved_matrix = None
    if include_svd:
        Distance_svd = load_svd(data_name, compare_name, Distance=True)
        Similarity_svd = load_svd(data_name, compare_name, Distance=False)
    else:
        Distance_svd = None
        Similarity_svd = None
    cp = ClusterProblem(labels, series, compare_name, solved_matrix=solved_matrix, solved_distance_svd=Distance_svd,
                        solved_similarity_svd=Similarity_svd, similarity=not Distance, gamma=gamma,
                        auto_diagonal=autodiagonal)

    return cp


if __name__ == "__main__":

    data_name = "Crop"
    compare_name = "dtw"

    cp = create_cluster_problem2(data_name, compare_name, no_solved_matrix=False, Distance=False, include_series=False)

    start = time.time()
    tau = 0.001
    kmax = 200
    Deltak = 30  
    W, deltas, kbest, gamma, _ = ACA_diag_pivots(cp, tau, kmax, Deltak)
    end = time.time()
    print("ACA time:", end-start)


    # Load in data for the cluster problem
    true_labels = load_labels(data_name)

    num_cluster = get_amount_of_classes(true_labels)

    lanczos_it = 20

    start = time.time()
    T, Q = lanczosFastMult(W, deltas, lanczos_it)
    T = np.array(T)  # Convert the 2D list to a 2D NumPy array
    Q = np.array(Q)  # Convert the 2D list to a 2D NumPy array
    end = time.time()
    print("Lanczos time:", end-start)

    D, V =  np.linalg.eigh(T)

    # Eigenvectors of S are Q*V
    Q = np.dot(Q, V)
    T = np.dot(np.dot(np.transpose(V),T), V)  # Similarity transform for T

    corr_eig = Q[: , 1:num_cluster+1] 

    norm = np.sqrt(np.sum(corr_eig**2, axis=1, keepdims=True))
    norm_mat = corr_eig / norm

    # Watch out with true labels here, sometimes different order!

    kmeans = KMeans(n_clusters=num_cluster, init='k-means++')
    kmeans.fit(norm_mat)
    predicted_labels = kmeans.predict(norm_mat)

    # ARIscore = adjusted_rand_score(predicted_labels, true_labels)
    # print('ARI score:', ARIscore)

# How I test the run here? It has to do with the problem size