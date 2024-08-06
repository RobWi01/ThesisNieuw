import numpy as np
import os

def Get_KNN_Using_Dis_Matrix(distance_matrix, train_size, index, K):
    """
    Load a similarity matrix and return the indices of the K nearest neighbors.

    :param name: Name of the dataset, used to construct file name.
    :param size: The number of entries in the dataset.
    :param index: The index of the entry for which neighbors are sought.
    :param K: The number of nearest neighbors to return.
    :return: Indices of the K nearest neighbors.
    """
    # Get the specific row from the matrix
    distances = distance_matrix[train_size + index][0:train_size]

    # Get indices of the sorted elements; np.argsort returns indices that would sort an array
    # We use [::-1] to reverse the result since we assume higher values indicate greater similarity
    sorted_indices = np.argsort(distances)

    # Select the top K indices
    nearest_neighbors = sorted_indices[:K]

    return nearest_neighbors
