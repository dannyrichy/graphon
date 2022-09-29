import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy import linalg
from torch.nn.init import sparse


def make_cost_m(conf_matrix):
    """
    Cost function for the Hungarian algorithm.

    :param conf_matrix: Confusion matrix
    :type conf_matrix: np.array

    :return: Cost matrix
    :rtype: np.array
    """
    s = np.max(conf_matrix)
    return - conf_matrix + s


def error(gt_real, labels):
    """
    Hungarian algorithm for error calculation

    :param gt_real: Ground truth
    :type gt_real: list

    :return: Error measure computed using the Hungarian algorithm
    :rtype: float
    """
    cm = confusion_matrix(gt_real, labels)
    # Hungarian algorithm
    indexes = linear_assignment(make_cost_m(cm))  
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    err = 1 - np.trace(cm2) / np.sum(cm2)
    return err


def generate_graph_laplacian(df, nn):
    """
    Compute the (weighted) graph of k-Neighbors for points in X.
    
    :param df: Dataframe
    :type df: pd.DataFrame

    :param nn: Number of nearest neighbors
    :type nn: int

    :return: Graph Laplacian
    :rtype: np.array
    """
    # Creates connectivity matrix.
    connectivity = kneighbors_graph(X=df, n_neighbors=nn, mode='connectivity')

    # Creates adjacency matrix.
    adjacency_matrix_s = (1 / 2) * (connectivity + connectivity.T)
    
    # Graph Laplacian.
    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=adjacency_matrix_s, normed=False)
    graph_laplacian = graph_laplacian_s.toarray()
    return graph_laplacian


def compute_spectrum_graph_laplacian(graph_laplacian):
    """
    Compute eigenvalues and eigenvectors and project
    them onto the real numbers.

    :param graph_laplacian: Graph Laplacian
    :type graph_laplacian: np.array

    :return: Eigenvalues and eigenvectors
    :rtype: np.array, np.array
    """
    eigenvals, eigenvcts = linalg.eigh(graph_laplacian)
    return eigenvals, eigenvcts