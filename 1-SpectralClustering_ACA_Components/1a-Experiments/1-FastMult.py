import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Using the Python version
sys.path.append('C:/Users/robwi/Documents/ThesisFinal/1-SpectralClustering_ACA_Components') # Change this when uploading to gitlab, working with relative imports?
from FastMult import fastMult

## Using the C++ wrapper
sys.path.append('C:\\Users\\robwi\\Documents\\ThesisFinal\\1-SpectralClustering_ACA_Components\\build\\Debug')
import fastMultModule

from Low_rank_timeseries.util import create_cluster_problem
from Low_rank_timeseries.Low_rank_approx.ACA_diag import ACA_diag_pivots
from Low_rank_timeseries.Low_rank_approx.util import reconstruct_matrix


###################################### Helper Functions ######################################

def reconstruct_matrix_renewed(W,delta,matrix, Distance=True, doCorr = True):
    for i in range(0,np.shape(delta)[0]):
        matrix += np.outer(W[i,:],W[i,:])/delta[i]
    if doCorr: 
        if Distance:
            np.fill_diagonal(matrix, 0)
            matrix[np.where(matrix<0)] = 0
        else:
            np.fill_diagonal(matrix, 1)
            matrix[np.where(matrix < 0)] = 0
            matrix[np.where(matrix > 1)] = 1
    else:
        pass


if __name__ == "__main__":
    name = 'Beef' # Change this to use a different dataset

    cp = create_cluster_problem(name, "dtw", Distance=False, include_series=False)

    tau = 0.001
    kmax = 40
    Deltak = 200
    W, inv_deltas, kbest, gamma, _ = ACA_diag_pivots(cp, tau, kmax, Deltak)

    size = W.shape[1]

    S_approx = np.zeros((size, size)) 
    # reconstruct_matrix(S_approx, W, inv_deltas, Distance = False, do_corrections = False)

    
    # Makes corrections on the matrix by default, turn of these corrections for this experiment 

    # Test to see the difference with the old ACA code, deltas have to be inversed now
    reconstruct_matrix_renewed(W, np.ones(len(inv_deltas))/inv_deltas, S_approx, Distance=False, doCorr=False)

    #TO-DO: Explain the different between 10^-8 and 10^-16 here between both reconstructs 
    # DTYPE = np.single, so I expected single precision here

    rel_error = np.zeros(size)
    vec = np.zeros((size,1))

    # TO-DO: explain here which vector I take each time
    for k in range(size):
        vec[k] = 1  
        result1 = np.dot(S_approx, vec)
        result2 = fastMultModule.fastMult(W.T, np.ones(len(inv_deltas))/inv_deltas, vec)
        rel_error[k] = np.linalg.norm(result1 - result2, 2) / np.linalg.norm(result1, 2)
        vec[k] = 0

    # Plotting the relative error, 
    # should see the machine precision here (differs based on using single or double precision)
    plt.figure(figsize=(10, 6))
    plt.plot(rel_error, marker='o', linestyle='-', color='blue')
    plt.title('Relative Error Over 60 Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error')
    plt.grid(True)
    plt.show()


    # Conclusion: The fast multiplication based 
