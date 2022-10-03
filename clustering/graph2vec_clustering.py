"""
Contains functions related to graph2vec clustering
"""
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from clustering.utils import error


def graph2vec_clustering(li_emb, true_labels, k):
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
    KM = KMeans(n_clusters=k)
    KM.fit(li_emb)

    labels = KM.labels_

    rand_idx_g2v = adjusted_rand_score(true_labels, labels)
    frac_err_g2v = error(true_labels, labels)

    return frac_err_g2v, rand_idx_g2v
