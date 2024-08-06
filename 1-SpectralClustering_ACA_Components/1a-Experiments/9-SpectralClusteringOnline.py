import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph

sys.path.append('C:/Users/robwi/Documents/ThesisFinal/1-SpectralClustering_ACA_Components')
from LanczosFastMult import lanczosFastMult
from Low_rank_timeseries.util import create_cluster_problem, load_matrix, load_timeseries, load_svd
from Low_rank_timeseries.Low_rank_approx.ACA_diag_pivots import ACA_diag_pivots
from Low_rank_timeseries.Data_paths import matrix_data_folder
from Low_rank_timeseries.Low_rank_approx.Cluster_problem import ClusterProblem
from Low_rank_timeseries.util import load_labels
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix

def symmetric_normalized_laplacian(S):
    degree_matrix = np.diag(np.sum(S, axis=1))
    laplacian_matrix = degree_matrix - S
    inv_sqrt_degree_matrix = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    normalized_laplacian_matrix = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix
    return normalized_laplacian_matrix

def get_amount_of_classes(labels):
    return len(np.unique(labels))

def myAlgo(lanczos_it, W, deltas, labels):
    num_cluster = get_amount_of_classes(labels)
    T, Q = lanczosFastMult(W, deltas, lanczos_it)
    T = np.array(T)
    Q = np.array(Q)
    D, V =  np.linalg.eigh(T)
    idx = D.argsort()
    D = D[idx]
    V = V[:, idx]
    Q = np.dot(Q, V)
    # Q = reorthogonalize(Q)
    corr_eig = Q[: , 1:num_cluster+1]
    norm = np.sqrt(np.sum(corr_eig**2, axis=1, keepdims=True))
    norm_mat = corr_eig / norm
    kmeans = KMeans(n_clusters=num_cluster, init='k-means++')
    kmeans.fit(norm_mat)
    predicted_labels = kmeans.predict(norm_mat)
    ARIscore = adjusted_rand_score(predicted_labels, labels)
    print('My method, ARI score:', ARIscore)
    return corr_eig, ARIscore

def reorthogonalize(Q):
    """Reorthogonalize the columns of Q using the Gram-Schmidt process."""
    for i in range(Q.shape[1]):
        for j in range(i):
            Q[:, i] -= np.dot(Q[:, i], Q[:, j]) * Q[:, j]
        Q[:, i] /= np.linalg.norm(Q[:, i])
    return Q

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
            sim = distance_to_similarity(di, gamma=gamma, method="Gaussian")
            # print("sim:", sim)
            # print("approx:", approx)
            # print("test:", (sim[0] - approx))
            # print("wi_new:", wi_new[i])
            wi_new[t-1] = sim[0] - approx
            new_W[i] = wi_new  # Update the skeleton in the list
        else:
            # How to decide which time series to use here
            # di = dtw.distance(T[indices[i]], Tnew)
            wi_new[t-1] = di - approx
            new_W[i] = wi_new  # Update the skeleton in the list

    return new_W

def do_adaptive_update(W, inv_deltas, indices, T, Tnew, n, b, gamma, Precomputed = False, Dis_Mat = None, train_size = 0, index = 0):
    """
    Does an adaptive update.
    """
    k = W.shape[0]
    n = W.shape[1]
    new_W = np.zeros((k, n+1))


    for i in range(k):
        wi = W[i]
        wi_new = np.zeros(n + 1)  # Create a new array for the updated skeleton
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
            sim = distance_to_similarity(di, gamma=gamma, method="Gaussian")
            # print("sim:", sim)
            # print("approx:", approx)
            # print("test:", (sim[0] - approx))
            # print("wi_new:", wi_new[i])
            wi_new[t-1] = sim[0] - approx
        else:
            print("Not precomputed")

        if wi_new[t-1] > inv_deltas[i]:
            new_W
        else:
            new_W[i] = wi_new  # Update the skeleton in the list



    return new_W

            


def extend_and_remove_prior_rows(self, start_index, end_index):
    """
    Step 1 of the adaptive update.
    """
    new_sample_values, new_sample_indices = self.find_new_samples_for_ts(start_index)
    for m in range(start_index+1, end_index):
        tmp_sv, tmp_si = self.find_new_samples_for_ts(m)
        new_sample_indices = np.concatenate((new_sample_indices, tmp_si))
        new_sample_values = np.concatenate((new_sample_values, tmp_sv))

    for i in range(len(self.rows)):
        self.update_state_new_samples(i, self.ACA_states[i], new_sample_values, new_sample_indices)
        for m in range(start_index, end_index):
            new_value = self.cp.sample(m, self.indices[i])
            approx = 0
            for j in range(i):
                approx += self.rows[j][self.indices[i]] * self.rows[j][m] * (1.0 / self.deltas[j])
            new_value -= approx
            self.rows[i] = np.append(self.rows[i], [new_value])
            pivot = self.choose_new_pivot(self.rows[i],  self.ACA_states[i])
            if not pivot == self.indices[i]:
                self.rows = self.rows[:i]
                self.deltas = self.deltas[:i]
                self.indices = self.indices[:i]
                self.ACA_states = self.ACA_states[:i+1]

if __name__ == "__main__":
    dataset_names = [
        "CBF", "Coffee", "Fungi", "GunPointOldVersusYoung", "HouseTwenty"
        # "InsectEPGRegularTrain", "ItalyPowerDemand", "OliveOil", "Plane", "ProximalPhalanxOutlineAgeGroup", "SonyAIBORobotSurface1",
        # "SyntheticControl", "Trace"
    ]

    # Still filter on the really small datasets --> don

    bigger_dataset_names = ["CBF"]

    # "Mallat" 
    # "StarLightCurves",
    # "TwoPatterns", "Symbols"

    # dataset_names = ['Symbols']

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    kmax_values = range(1, 20, 5)
    
    results = []

    for data_name in bigger_dataset_names:
        print(f'Processing dataset: {data_name}')
        compare_name = "dtw"
        labels = load_labels(data_name)
        file_path = os.path.join(base_dir, "Matrices", "Similarity_matrices", f"{data_name}_dtw.npy")
        S = np.load(file_path)
        num_clusters = get_amount_of_classes(labels)

        file_path2 = os.path.join(base_dir, "Matrices", "Distance_matrices", f"{data_name}_dtw.npy")

        A = np.load(file_path)



        scores = []
        scores_recon = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for i in range(10):

                data_dir = "C:/Users/robwi/Documents/ThesisFinal/Data/"

                train_path = data_dir + data_name + "/" + data_name + "_TRAIN.tsv"
                test_path = data_dir + data_name + "/" + data_name + "_TEST.tsv"




                labels_train, series_train = load_timeseries_and_labels_from_tsv(train_path)
                labels_test, series_test = load_timeseries_and_labels_from_tsv(test_path)

                labels = load_labels(data_name)

                size = len(series_train)

                cp = ClusterProblem(labels_train, series_train, "dtw", similarity=False)  # dtw satisfies triangle inequality

                tau = 0.001
                kmax = int(1 * len(series_train))
                Deltak = 20
                new_W2, deltas, kbest, gamma, pivot_indices = ACA_diag_pivots(cp, tau, kmax, Deltak)

                for i in range(len(series_test)):  # Sample new time series
                    T_new = series_test[i]
                    new_W2 = update_skeletons(new_W2, deltas, pivot_indices, series_train, T_new, size, 1, gamma, Precomputed=True, Dis_Mat=A, train_size=size, index=i)

                    eigv, score = myAlgo(15, new_W2, deltas, labels[:size+1])
                    # scores.append(score)
                    print(score)

                    size += 1



            # scores_array_recon = np.array(scores_recon)
            # scores_array = np.array(scores)
            # average_score_recon = np.average(scores_array_recon)
            # mean_score_recon = np.mean(scores_array_recon)
            # std_deviation_recon = np.std(scores_array_recon)
            # Q3_recon = np.percentile(scores_array_recon, 75)
            # Q1_recon = np.percentile(scores_array_recon, 25)
            # IQR_recon = Q3_recon - Q1_recon
            # average_score = np.average(scores_array)
            # mean_score = np.mean(scores_array)
            # std_deviation = np.std(scores_array)
            # Q3 = np.percentile(scores_array, 75)
            # Q1 = np.percentile(scores_array, 25)
            # IQR = Q3 - Q1
            # results.append({
            #     'Dataset': data_name,
            #     'kmax': kmax,
            #     'kbest': kbest,
            #     'Average Score Recon': average_score_recon,
            #     'Mean Score Recon': mean_score_recon,
            #     'Std Deviation Recon': std_deviation_recon,
            #     'IQR Recon': IQR_recon,
            #     'Average Score': average_score,
            #     'Mean Score': mean_score,
            #     'Std Deviation': std_deviation,
            #     'IQR': IQR
            # })

    # df = pd.DataFrame(results)
    # output_path = os.path.join(base_dir, "clustering_results_bigdata_CBF.xlsx")
    # df.to_excel(output_path, index=False)
    # print(f"Results saved to {output_path}")
