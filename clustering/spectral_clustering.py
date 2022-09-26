"""
File that contains all functions related to spectral clustering of graphs
"""
import numpy as np
import pandas as pd
import torch
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from torch.nn.init import sparse

from config import DEVICE
from graphon.hist_estimator import hist_approximate
from utils import error


def kmeans_dist(dist, num_clusters=2):
    w, v = torch.eig(dist, eigenvectors=True)
    w_real = w[:, 0]  # symmetric matrix so no need to bother about the complex part
    sorted_w = torch.argsort(-torch.abs(w_real))
    to_pick_idx = sorted_w[:num_clusters]
    eig_vec = v[:, to_pick_idx]
    eig_vec = eig_vec.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(eig_vec)
    return kmeans.labels_


def distance_matrix(li_graph):
    no_graphs = len(li_graph)
    dist = torch.zeros((no_graphs, no_graphs), dtype=torch.float64).to(device=DEVICE)
    for i in range(no_graphs):
        for j in range(i + 1):
            dist[i][j] = torch.norm(li_graph[i] - li_graph[j])
            dist[j][i] = dist[i][j]
    return dist


def sim_matrix(all_graphs, sigma):
    m = len(all_graphs)
    sim = torch.zeros((m, m), dtype=torch.float64).to(device=DEVICE)
    for i in range(m):
        # n = all_graphs[i].shape[0]
        for j in range(i + 1):
            sim[i][j] = torch.exp(-torch.norm(all_graphs[i] - all_graphs[j]) ** 2 / (sigma[i] * sigma[j]))
            sim[j][i] = sim[i][j]
    return sim


# def sdp(sim, num_clusters=2):
#     sim = sim.cpu().detach().numpy()
#     # Define and solve the CVXPY problem.
#     b = np.ones((sim.shape[0], 1), dtype=np.float64)
#     # Create a symmetric matrix variable.
#     X = cp.Variable((sim.shape[0], sim.shape[1]), symmetric=True)
#     # The operator >> denotes matrix inequality.
#     constraints = [X >> 0]
#     constraints += [X @ b == b]
#     constraints += [cp.trace(X) == num_clusters]
#     prob = cp.Problem(cp.Maximize(cp.trace(sim @ X)),
#                       constraints)
#     prob.solve()
#
#     return X


def generate_graph_laplacian(df, nn):
    """Generate graph Laplacian from data."""
    # Adjacency Matrix.
    connectivity = kneighbors_graph(X=df, n_neighbors=nn, mode='connectivity')
    adjacency_matrix_s = (1 / 2) * (connectivity + connectivity.T)
    # Graph Laplacian.
    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=adjacency_matrix_s, normed=False)
    graph_laplacian = graph_laplacian_s.toarray()
    return graph_laplacian


def compute_spectrum_graph_laplacian(graph_laplacian):
    """Compute eigenvalues and eigenvectors and project
    them onto the real numbers.
    """
    eigenvals, eigenvcts = linalg.eig(graph_laplacian)
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)
    return eigenvals, eigenvcts


def spectral_clustering(affinity_mat, num_clusters=3):
    """

    :param affinity_mat:
    :type affinity_mat:

    :param num_clusters:
    :type num_clusters:

    :return:
    :rtype:
    """
    graph_laplacian = generate_graph_laplacian(df=affinity_mat, nn=8)
    eigenvals, eigenvcts = compute_spectrum_graph_laplacian(graph_laplacian)

    eigenvals_sorted_indices = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]

    zero_eigenvals_index = np.argwhere(abs(eigenvals) < 1e+0)
    # eigenvals[zero_eigenvals_index]

    proj_df = pd.DataFrame(eigenvcts[:, eigenvals_sorted_indices[:num_clusters]])  # zero_eigenvals_index.squeeze()])
    k_means = KMeans(random_state=25, n_clusters=num_clusters)
    k_means.fit(proj_df)
    labels = k_means.predict(proj_df)

    return labels


def simulate_histogram(graphs, gt, check_n0=[5, 10, 15, 20, 25, 30], sigma=[5], num_clusters=3):
    frac_err_spect = []
    rand_idx_spect = []
    rand_idx_sdp = []
    all_err_sdp = []

    for n0 in check_n0:
        # List of graphons
        graphs_apprx = hist_approximate(graphs, n0)

        # Distance between graphons
        dist = distance_matrix(graphs_apprx)
        # Get label name for each graph
        labels = kmeans_dist(dist, num_clusters=num_clusters)

        # Measures of error rate
        rand_idx_spect.append(adjusted_rand_score(gt, labels))
        frac_err_spect.append(error(gt, labels))

        # # [5] for real data
        # # [1,2,3] for simulation
        # dist_sorted = torch.sort(dist, 1).values
        # err_f = []
        # rand_idx_f = []
        #
        # for s in sigma:
        #     sim = sim_matrix(graphs_apprx, dist_sorted[:, s])
        #     X = sdp(sim, num_clusters)
        #
        #     l = spectral_clustering(X.value, num_clusters=num_clusters)
        #     err = error(gt, l)
        #     print('spec error for sigma ', n0, s, err)
        #     err_f.append(err)
        #     rand_idx_f.append(adjusted_rand_score(gt, l))

        # all_err_sdp.append(err_f)
        # rand_idx_sdp.append(rand_idx_f)

    frac_err_spect = np.array(frac_err_spect)

    print('err, ari spect  ', frac_err_spect, rand_idx_spect)
    print('error sdp all sigmas ', all_err_sdp)
    print('rand idx all sigmas ', rand_idx_sdp)
    return frac_err_spect, all_err_sdp, rand_idx_spect, rand_idx_sdp
