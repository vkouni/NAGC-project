# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:17:54 2021

@author: vaso
"""
import os
import NAGC_class
import NAGC_relu
import build_graph
import evaluate
import init_kmeans
from init_agglomerative import init_agglomerative

# The directory that the results will be saved in
result_path = os.path.join(os.getcwd(), "Results")

# if the result directory does not exist create it
if not os.path.exists(result_path):
    os.mkdir(result_path)

# The datasets used in the experiments
datasets = ["citeseer", "WebKB_univ", "citeseer", "polblog", "cora"]

# For each dataset
for data in datasets:
    # create a results dir for the dataset
    res_folder = os.path.join(result_path, data)
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    data_path = "data/" + data
    # load the dataset and build the graph
    S, W, X, true_clus, flag, A_ori = build_graph.build_graph(data_path)
    
    # The clustering methods to use
    clustering_method = [ "kmeans","agglomerative"]
    # The linkages (applicable in agglomerative only)
    linkages = ["complete", 'average', 'single']
    # The k2 parameter values used in the experiments
    k_2s = [2, 4, 5, 6, 7, 10, 15,20] 
    # The lambda values used in the experiments
    lambda_values = [0, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10] 
    # The rho values used in the experiments
    rho_values = [0.5, 0.55, 0.75, 0.95, 0.995]
    # The non-linear activation functions used in the experiments
    activations = ['sigmoid', 'relu']
    # Initialize the k1 parameter based on the values proposed by Makoto Onizuka et al., 2018
    if data == "WebKB_univ":
        k1 = 4
    elif data == "citeseer":
        k1 = 6
    elif data == "polblog":
        k1 = 2
    else:
        k1 = 7
    # For each clustering method, open the corresponding results CSV file,
    # run all the experiments and output the parameters, runtime and metrics (ARI, entropy, modularity)
    for method in clustering_method:
        if method == "kmeans":
            res_file = open(os.path.join(res_folder, "res" + "_" + method + ".csv"), "a+")
            res_file.write("MODULARITY;ENTROPY;ARI;K2;LAMBDA;RHO;TIME;ACTIVATION\n")
            for k2 in k_2s:
                for lam in lambda_values:
                    for rho in rho_values:
                        for activation in activations:
                            init_kmeans.init_kmeans(k1, data)
                            init_kmeans.init_kmeans(k2, data)

                            iteration = 100  #100 number of iterations
                            init = 1  # 0: random initialization, 1: kmeans initialization
                            if activation == "sigmoid":
                                NAGC = NAGC_class.NAGC(k1, k2, lam, S, W, X, data, init, rho)
                            else:
                                NAGC = NAGC_relu.NAGC(k1, k2, lam, S, W, X, data, init, rho)
                            U, elapsed_time = NAGC.fit_predict()

                            k = k1
                            clus = []
                            for i in range(U.shape[1]):
                                clus.append([])
                            pred = U.argmax(1)
                            for i in range(U.shape[0]):
                                clus[pred[i]].append(i)

                            modularity = evaluate.cal_modularity(clus, S, k)
                            entropy = evaluate.cal_entropy(clus, X, k)
                            ari = evaluate.ARI(true_clus, pred) if true_clus else 0.0
                            print("modularity: " + str(evaluate.cal_modularity(clus, S, k)))
                            print("entropy: " + str(evaluate.cal_entropy(clus, X, k)))
                            if true_clus:
                                print("ARI: " + str(evaluate.ARI(true_clus, pred)))

                            res_file.write("{:.4f};".format(modularity))
                            res_file.write("{:.4f};".format(entropy))
                            res_file.write("{:.4f};".format(ari))
                            res_file.write(str(k2) + ";")
                            res_file.write("{:.4f};".format(lam))
                            res_file.write("{:.4f};".format(rho))
                            res_file.write("{:.4f};".format(elapsed_time))
                            res_file.write(activation + "\n")
            res_file.close()
            print ("Experiments Complete.")
        else:
            res_file = open(os.path.join(res_folder, "res" + "_" + method + ".csv"), "a+")
            res_file.write("MODULARITY;ENTROPY;ARI;K2;LAMBDA;RHO;LINKAGE;TIME;ACTIVATION\n")
            for k2 in k_2s:
                for lam in lambda_values:
                    for rho in rho_values:
                        for activation in activations:
                            for linkage in linkages:

                                init_agglomerative(k1, data, linkage=linkage)
                                init_agglomerative(k2, data, linkage=linkage)
                                iteration = 100  # number of iterations
                                init = 1  # 0: random initialization, 1: kmeans initialization
                                if activation == "sigmoid":
                                    NAGC = NAGC_class.NAGC(k1, k2, lam, S, W, X, data, init, rho)
                                else:
                                    NAGC = NAGC_relu.NAGC(k1, k2, lam, S, W, X, data, init, rho)
                                U, elapsed_time = NAGC.fit_predict()

                                k = k1
                                clus = []
                                for i in range(U.shape[1]):
                                    clus.append([])
                                pred = U.argmax(1)
                                for i in range(U.shape[0]):
                                    clus[pred[i]].append(i)

                                modularity = evaluate.cal_modularity(clus, S, k)
                                entropy = evaluate.cal_entropy(clus, X, k)
                                ari = evaluate.ARI(true_clus, pred) if true_clus else 0.0
                                print("modularity: " + str(evaluate.cal_modularity(clus, S, k)))
                                print("entropy: " + str(evaluate.cal_entropy(clus, X, k)))
                                if true_clus:
                                    print("ARI: " + str(evaluate.ARI(true_clus, pred)))
                                res_file.write("{:.4f};".format(modularity))
                                res_file.write("{:.4f};".format(entropy))
                                res_file.write("{:.4f};".format(ari))
                                res_file.write(str(k2) + ";")
                                res_file.write("{:.4f};".format(lam))
                                res_file.write("{:.4f};".format(rho))
                                res_file.write(linkage + ";")
                                res_file.write("{:.4f};".format(elapsed_time))
                                res_file.write(activation + "\n")
            res_file.close()
            print("Experiments Completed.")
