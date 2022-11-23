"""
Contains functions related to graph2vec clustering
"""
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralClustering

from clustering.utils import hungarian_error


def graph2vec_clustering(li_emb, true_labels, k, no_eig_vecs=2):
    """
    K means clustering method for graph embeddings

    :param li_emb: List of embeddings
    :type li_emb: list

    :param true_labels: true labels
    :type true_labels: list

    :param k: number of clusters
    :type k: int

    :return: Train metrics
    :rtype: tuple
    """
    # KM = KMeans(n_clusters=k)
    # KM.fit(li_emb)
    SC = SpectralClustering(n_clusters=k, n_components=no_eig_vecs, affinity='nearest_neighbors')
    SC.fit(li_emb)
    
    labels = SC.labels_

    rand_idx_g2v = adjusted_rand_score(true_labels, labels)
    frac_err_g2v = hungarian_error(true_labels, labels)

    return rand_idx_g2v, frac_err_g2v
