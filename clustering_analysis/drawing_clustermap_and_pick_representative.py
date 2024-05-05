#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Shikai Jin'
__date__ = '2024-Mar-26'
__version__ = '2.0'

# Written by Shikai Jin on 2020-Aug-08, latest modified on 2024-Mar-26, modified from Xun's clustern5.py
# Extract the clusters based on mutual-Q value file, draw clustermap and dendrogram
# Example in Linux: python cluster_and_pick_representative.py mutualq.txt


import importlib
import itertools
import math
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import scipy
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import argparse


def convert_to_matrix(fname):
    # This function converts the input file (one column with X*X mutual-Q values to a matrix in numpy format)
    matrix = np.loadtxt(fname)
    seqlen = int(math.sqrt(len(matrix)))
    newmat = np.reshape(matrix, [seqlen, seqlen])
    [seqlen, seqlen] = np.shape(newmat)
    return seqlen, newmat

def draw_clustermap(seqlen, newmat):
    # This function draws the clustermap based on the matrix get from above function

    # Initialize the Qmatrix with zeros and appropriate dimensions
    Qmatrix = pd.DataFrame(0, index=[str(a) for a in range(1, seqlen + 1)], 
                           columns=[str(a) for a in range(1, seqlen + 1)])
    
    # Fill the diagonal with 1s
    for i in range(1, seqlen + 1):
        Qmatrix.at[str(i), str(i)] = 1

    # Populate the upper and lower triangles of the matrix
    for a, b in itertools.combinations(range(1, seqlen + 1),
                                       2):  # Return 2 length subsequences of elements from the input range(1, seqlen+1).
        Qmatrix.at[str(a), str(b)] = newmat[a - 1][b - 1]
        Qmatrix.at[str(b), str(a)] = newmat[b - 1][a - 1]
    
    # Draw the clustermap # figsize=(50.0 / 100 * seqlen, 50.0 / 100 * seqlen), 
    cms = sns.clustermap(Qmatrix, cmap="coolwarm", vmin=0.4, vmax=1.0, linewidths=0.4, linecolor='black',
                        xticklabels=False, yticklabels=False)
    # ax = cms.ax_heatmap
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    cms.savefig("clustermap.png", bbox_inches='tight', dpi=300)
    return Qmatrix, cms

def draw_dendrogram(Qmatrix, cms):
    # This is to print cms results in dendrogram tree with colored diagrams
    index = cms.dendrogram_col.reordered_ind
    plt.figure(figsize=(15, 7))
    scipy.cluster.hierarchy.set_link_color_palette(
        ['m', 'c', 'y', 'k', 'b', 'r', 'g', 'gray', 'gold', 'lightblue', 'lightcyan', 'navy', 'olive', 'pink'])
    # den = scipy.cluster.hierarchy.dendrogram(cms.dendrogram_col.linkage,color_threshold=threshold,labels = Qmatrix.index)#,color_threshold=1.0,leaf_font_size = 3)

    # A for loop to determine which threshold could generates a list that the number of clusters over 5
    #print(cms.dendrogram_col.linkage) What is the output of linkage please check scipy.cluster.hierarchy.linkage
    curr_clusters = 0
    best_threshold = 0.8
    for i in range(0, 8):
        threshold = 0.8 - i * 0.1
        den = scipy.cluster.hierarchy.dendrogram(cms.dendrogram_col.linkage, color_threshold=threshold,
                                                 labels=Qmatrix.index)
        #print(den['color_list']) # The color_list represents the color of Xth link! The last color should always be the link of root cluster.
        # Find out what is den
        Z = Counter(den['color_list'])
        #print(Z)
        if len(Z) >= 6 or len(Z) >= curr_clusters:
            best_threshold = threshold
            curr_clusters = len(Z)
    den = scipy.cluster.hierarchy.dendrogram(cms.dendrogram_col.linkage, color_threshold=best_threshold,
                                                 labels=Qmatrix.index)
    #print(threshold)
    plt.savefig("dendrogram.png", bbox_inches='tight', dpi=600)
    return den

def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    cluster_classes = {}
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l
    return cluster_classes

def get_cluster_max(newmat, cluster):
    # cluster = clusters_idxs[c]
    length = len(cluster)
    Q_sum = np.zeros(length)
    index = []
    for i in range(length):
        index.append(int(cluster[i]))
    for i in range(length):
        for j in range(length):
            indexi = int(index[i]) - 1
            indexj = int(index[j]) - 1
            Q_sum[i] += newmat[indexi][indexj]
    index_max = np.argmax(Q_sum)
    cluster_center = index[index_max]
    #print (cluster_center)
    return cluster_center


def write_clusters(newmat, den, clusters):
    xx = []
    data = ""
    data_center = "" # deal with "'c', 'y', 'b', 'k', 'b'"
    for c in den['color_list']:
        if c not in xx:
            try:
                data += str(clusters[c]) + "\n"
                clustermax = get_cluster_max(newmat, clusters[c])
                data_center += str(clustermax) + "\n"
            except:
                pass
        xx.append(c)

    with open("cluster_print", "w") as fopen:
        fopen.writelines(data)
    with open("cluster_center", "w") as fopen:
        fopen.writelines(data_center)

        
def main():
    parser = argparse.ArgumentParser(description="This script draws the clustermap nad dendrogram from a mutual value matrix.")
    parser.add_argument("input",
                        help="The file name of input.", type=str)
    args = parser.parse_args()

    seqlen, newmat = convert_to_matrix(args.input)
    Qmatrix, cms = draw_clustermap(seqlen, newmat)
    den = draw_dendrogram(Qmatrix, cms)
    clusters = get_cluster_classes(den)
    #print(clusters)
    write_clusters(newmat, den, clusters)


if __name__ == '__main__':
    main()