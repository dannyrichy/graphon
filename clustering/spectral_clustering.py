"""
File that contains all functions related to spectral clustering of graphs
"""
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from config import DEVICE
from clustering.utils import error, generate_graph_laplacian, compute_spectrum_graph_laplacian
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def DSC(dist, no_clusters=2, no_eig_vecs=2):
    """
    Performs the Distance Based Spectral Clustering 
    on the distance matrix as defined in the paper 

    :param dist: Distance matrix
    :type dist: torch.Tensor

    :param no_clusters: Number of clusters
    :type no_clusters: int

    :param no_eig_vecs: Number of eigenvectors to use
    :type no_eig_vecs: int

    :return: Labels
    :rtype: torch.Tensor
    """
    # eigenvalues and eigenvectors decomposition of symmetric matrix
    eig_vals, eig_vecs = torch.linalg.eigh(dist)

    # Sorting eigvalues in ascending order
    eig_vals = torch.argsort(torch.abs(eig_vals))
    
    # Selecting the first no_eig_vecs eigenvectors
    idxs = eig_vals[:no_eig_vecs]
    eig_vecs = eig_vecs[:, idxs].cpu().detach().numpy()

    logger.info("Performing kmeans clustering with %d clusters", no_clusters)
    kmeans = KMeans(n_clusters=no_clusters, random_state=0).fit(eig_vecs)
    return kmeans.labels_


def frobenius_norm(li_graph):
    """
    Returns the Frobenius norm matrix of a list of graphs
    :param li_graph: List of graphs
    :type li_graph: list

    :return: Frobenius norm matrix
    :rtype: torch.Tensor
    """
    no_graphs = len(li_graph)
    dist = torch.zeros((no_graphs, no_graphs), dtype=torch.float64).to(device=DEVICE)
    for i in range(no_graphs):
        for j in range(i + 1):
            dist[i][j] = torch.norm(li_graph[i].float() - li_graph[j].float())
            dist[j][i] = dist[i][j]
    return dist



def spectral_clustering(affinity_mat, num_clusters=3):
    """
    Vanilla spectral clustering algorithm.

    :param affinity_mat: Affinity matrix (adjacency matrix)
    :type affinity_mat: np.array

    :param num_clusters: Number of clusters
    :type num_clusters: int

    :return: Labels
    :rtype: np.array
    """
    # Generates graph Laplacian from affinity matrix
    graph_laplacian = generate_graph_laplacian(df=affinity_mat, nn=8)
    eigen_vals, eigen_vecs = compute_spectrum_graph_laplacian(graph_laplacian)

    eigenvals_sorted_indices = np.argsort(eigen_vals)

    proj_df = pd.DataFrame(eigen_vecs[:, eigenvals_sorted_indices[:num_clusters]])  
    k_means = KMeans(random_state=25, n_clusters=num_clusters)
    k_means.fit(proj_df)
    labels = k_means.predict(proj_df)

    return labels


def graphon_clustering(graphs, true_labels, num_clusters=3):
    """
    Perform data_loader clustering and returns the error scores.

    :param graphs: List of graphs
    :type graphs: list

    :param true_labels: Ground truth
    :type true_labels: list

    :param n0: number of nodes in data_loader approximation
    :type n0: int

    :param num_clusters: Number of clusters for spectral clustering
    :type num_clusters: int

    :return: Error scores for clustering
    :rtype: tuple[float, float]
    """

    # Matrix of pairwise distances between graphons
    dist = frobenius_norm(graphs)

    # Get label name for each graph
    labels = DSC(dist, no_clusters=num_clusters)

    return adjusted_rand_score(true_labels, labels), error(true_labels, labels)
