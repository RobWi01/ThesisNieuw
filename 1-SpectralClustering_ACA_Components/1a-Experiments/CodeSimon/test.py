import copy
import os
import sys
import os

path = os.getcwd()

print(path)
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from cluster_problem import ClusterProblem
from data_loader import load_timeseries_from_tsv
from extendable_aca import ACA
import random as rn
import numpy as np 

func_name = "dtw"

cp = ClusterProblem(series[0:start_index], func_name, compare_args=args, solved_matrix=active_dm)



    args = {"window": len(series) - 1}
    k = len(set(labels))
    file_names = []
    seed_file_name = rn.randint(0, 9999999999)
    for method in methods:
        if random_file:
            file_names.append("results/part2/" + name + "/" + str(seed_file_name) + "_" + method)
        else:
            file_names.append("results/part2/" + name + "/" + name + "_" + method)
    results = read_all_results(file_names, len(series), start_index, skip)
    while len(results[0]) <= iterations:
        if dm is not None:
            active_dm = dm[range(start_index), :]
            active_dm = active_dm[:, range(start_index)]
        else:
            active_dm = None
        results = read_all_results(file_names, len(series), start_index, skip)
        start_index_approx = rn.randint(0, start_index - 1)
        seed = rn.randint(0, 99999999)
        print(name + ":" + " STARTING NEW APPROX: it =", len(results[0]), "start index approx =", start_index_approx,
              "seed =", seed, "skip =", skip)
        approximations = [ACA(cp, tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed)]
        index = start_index
        update_results(approximations, results, labels, active_dm, a_spectral, k, index, start_index, skip, name)
        new_series = []
        while index < len(series) - 1:
            index += 1
            new_series.append(series[index])
            if index % skip == 0:
                if dm is not None:
                    active_dm = dm[range(index), :]
                    active_dm = active_dm[:, range(index)]
                else:
                    active_dm = None
                extend_approximations(approximations, methods, new_series, solved_matrix=active_dm)
                update_results(approximations, results, labels, active_dm, a_spectral, k, index, start_index, skip,
                               name)
                new_series = []

        for file_name, result in zip(file_names, results):
            np.save(file_name, result)