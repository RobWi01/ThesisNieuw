import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from Low_rank_timeseries.util import load_labels

####################################################### Helper Functions #######################################################

def symmetric_normalized_laplacian(S):
    # Step 1: Calculate the Degree Matrix D
    degree_matrix = np.diag(np.sum(S, axis=1))
    
    # Step 2: Calculate the Laplacian Matrix L
    laplacian_matrix = degree_matrix - S
    
    # Step 3: Calculate D^(-1/2)
    with np.errstate(divide='ignore'):
        inv_sqrt_degree_matrix = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    inv_sqrt_degree_matrix[np.isinf(inv_sqrt_degree_matrix)] = 0
    
    # Step 4: Calculate the Symmetric Normalized Laplacian L_norm
    normalized_laplacian_matrix = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix
    
    return normalized_laplacian_matrix

def get_amount_of_classes(labels):
    return len(np.unique(labels))

if __name__ == '__main__':
    dataset_names = ["crop"]

    base_dir = "C:\\Users\\robwi\\Documents\\ThesisFinal"
    compare_name = "dtw"

    results = []

    # run block of code and catch warnings
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        for data_name in dataset_names:
            print(f'Working on {data_name} ...')

            labels = load_labels(data_name)

            file_path = os.path.join(base_dir, "Matrices", "Similarity_matrices", f"{data_name}_dtw.npy")

            S = np.load(file_path)

            laplacian_matrix_org = symmetric_normalized_laplacian(S)

            num_clusters = get_amount_of_classes(labels)

            # Use sparse eigenvalue solver for symmetric matrices
            w, v = eigsh(laplacian_matrix_org, k=num_clusters+1, which='SM')

            corr_eig = np.real(v[:, 1:num_clusters+1])  # Take only the real part
            norm = np.sqrt(np.sum(corr_eig**2, axis=1, keepdims=True))
            norm_mat = corr_eig / norm

            # Watch out with true labels here, sometimes different order!
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
            kmeans.fit(norm_mat)
            predicted_labels = kmeans.predict(norm_mat)

            score = adjusted_rand_score(predicted_labels, labels)
            print('Exact matrix score:', score)

            if score > 0.50:
                results.append((data_name, score))

    if results:
        df_results = pd.DataFrame(results, columns=['Dataset Name', 'Score'])
        output_file = os.path.join(base_dir, "High_Score_Datasets.xlsx")
        df_results.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No dataset had a score greater than 0.50")