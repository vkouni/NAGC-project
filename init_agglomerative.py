#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.cluster import AgglomerativeClustering

import argparse
import build_graph
import numpy as np
import csv
import evaluate
from typing import Dict


def clustering(A, k, linkage):
    # initializes cluster list as empty
    clus = []
    # for each cluster adds and empty list3
    # to the clusters
    for i in range(k):
        clus.append([])
    ############# regularization for input attributes
    # for each colums of attribute matrix (attribute)
    # get the max value and divide all values with the max
    for i in range(A.shape[1]):  # A.shape gives the num_rows and num_cols of A - index 1 is columns
        max_att = max(A[:, i])
        if max_att != 0:
            A[:, i] = A[:, i] / max_att

    ############ Learning step of clustering method
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(A)
    pred = model.labels_
    centroids = np.zeros((k, A.shape[1]))
    frequencies = {}
    for i in range(pred.shape[0]):
        centroids[pred[i], :] = np.add(centroids[pred[i], :], A[i, :])
        if pred[i] in frequencies:
            frequencies[pred[i]] += 1
        else:
            frequencies[pred[i]] = 1
    for i in range(k):
        centroids[i, :] = centroids[i, :] / frequencies[i]

    # for each node
    for i in range(len(pred)):
        # go to the cluster center assignment list and append the nodes assigned to each cluster
        clus[pred[i]].append(i)
    return pred, centroids, clus


def initialize_U(A, centroids):
    # create matrix U with rows equal to nodes and columns equal to the number of clusters
    # and initialize all values to zero
    U = np.zeros((len(A), len(centroids)))
    # for each node
    for i in range(U.shape[0]):
        dis_list = []
        # for each cluster center
        for j in range(U.shape[1]):
            # append to the distance list the norm (Euclidean distance) of the node attributes to the centroid
            dis_list.append(np.linalg.norm(A[i] - centroids[j]))
        # for each cluster center
        for j in range(U.shape[1]):
            # assign the following value to the (node,centroid) location in the U matrix
            U[i, j] = (sum(dis_list) - dis_list[j]) / sum(dis_list)
        # convert the cluster assignments to probability
        U[i, :] = U[i, :] / sum(U[i, :])
    return U


def init_agglomerative(k, data, linkage: str = 'complete'):
    print(data)

    path = "data/" + data
    S, S_ori, A, true_clus, flag, A_ori = build_graph.build_graph(path)
    # convert the list of true clusters to a set (unique ocurrence of each cluster)
    # and then back to a list
    clus_list = list(set(true_clus))
    print(clus_list)
    # initializes a cluster dictionary
    # and assign to each cluster a unique index
    clus_dic = {}
    for i in range(len(clus_list)):
        clus_dic[clus_list[i]] = i
    # replace each true cluster by its index
    for i in range(len(true_clus)):
        true_clus[i] = clus_dic[true_clus[i]]

    # Initialize all these empty lists
    pred_l = []
    cent_l = []
    km_l = []
    mod = []
    ent = []
    nmi = []
    ari = []
    # repeates experiment 5 times
    for j in range(5):  # for i=0 to 4
        # get the predicted assignment of each node, the centroids and the
        # nodes assigned to each centroid
        pred, centroids, clusters = clustering(A_ori, k, linkage)
        # append the predicted assignments of the run to the pred_l list
        pred_l.append(pred)
        # append the cetroids of the run to the cent_l list
        cent_l.append(centroids)
        # append the nodes assigned to each cluster for the run to the km_l list
        km_l.append(clusters)
        # calculate ARI score for the run and append to ari list
        ari.append(evaluate.ARI(true_clus, pred))
    # sorts the ari score vector, get the middle (3rd element) and gets the index (location)
    # of the element in the original ari score array
    ind = ari.index(sorted(ari)[2])
    # gets the predicted cluster assignments for the above index (run)
    pred = pred_l[ind]
    # gets the centroids of the above index (run)
    centroids = cent_l[ind]
    # gets the nodes assigned to each cluster for the above index (run)
    kmeans_clus = km_l[ind]
    # creates U matrix
    U = initialize_U(A_ori, centroids)
    # writes the contents of the U matrix to the corresponding csv files
    f = open('initialize/' + data + '_U_' + str(k) + '.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(U)
    f.close()
    # writes the contents of the centroids list to the corresponding file
    f = open('initialize/' + data + '_V_' + str(k) + '.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(centroids)
    f.close()
