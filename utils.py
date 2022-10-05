import io
import zipfile
from functools import reduce

import matplotlib.pyplot as plt
import networkx as nx
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
import torch
import logging
from clustering.graph2vec_clustering import graph2vec_clustering
from clustering.spectral_clustering import graphon_clustering
from graphon.graphons import SynthGraphons

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger = logging.getLogger(__name__)

def combine_datasets(li_dataset):
    """
    Combining datasets

    :param li_dataset: list of datasets
    :type li_dataset: list

    :return: Reduced datasets with labels
    :rtype: tuple
    """
    result = reduce(lambda x, y: x + y, li_dataset)
    labels = np.array([i for i, dataset in enumerate(li_dataset) for _ in range(len(dataset))])
    return result, labels


def download_datasets(dataset_links=None):
    """
    Function to download dataset

    :param dataset_links:
    :type dataset_links: list

    :return: Downloads and extracts graph dataset
    :rtype: None
    """
    if dataset_links is None:
        dataset_links = ['https://www.chrsmrrs.com/graphkerneldatasets/facebook_ct1.zip',
                         'https://www.chrsmrrs.com/graphkerneldatasets/deezer_ego_nets.zip',
                         'https://www.chrsmrrs.com/graphkerneldatasets/github_stargazers.zip',
                         'https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-BINARY.zip']
    for link in dataset_links:
        zipfile.ZipFile(io.BytesIO(requests.get(link).content)).extractall()


# taken from https://github.com/JiaxuanYou/graph-generation/blob/3444b8ad2fd7ecb6ade45086b4c75f8e2e9f29d1/data.py#L24
def load_graph(min_num_nodes=10, name='ENZYMES'):
    """
    Load real world graph dataset

    :param min_num_nodes: minimum number of nodes in the graph
    :type min_num_nodes: int

    :param name: name of the dataset
    :type name: str

    :return: Graph dataset
    :rtype: list
    """
    print('Loading graph dataset: ' + str(name))
    G = nx.Graph()
    # load data
    path = 'datasets/' + name + '/'
    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    # nx_graphs = []
    max_nodes = 0
    all_nodes = []
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if G_sub.number_of_nodes() >= min_num_nodes:
            adj = nx.adjacency_matrix(G_sub)
            adj = adj.todense()
            adj = torch.Tensor(np.asarray(adj)).to(device=DEVICE)
            graphs.append(adj)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            all_nodes.append(G_sub.number_of_nodes())
    print('Loaded and the total number of graphs are ', len(graphs))
    print('max num of nodes is ', max_nodes)
    print('total graphs ', len(graphs))
    print('histogram of number of nodes in ', name)
    # print(all_nodes)
    plt.hist(all_nodes)
    plt.title(f'Number of nodes for each graph in {name}')
    plt.show()
    return graphs

def classification(embeddings, true_labels, GRAPH2VEC=False):
    """
    Classification of graph embeddings using Random Forests
    
    :param embeddings: List of embeddings
    :type embeddings: list

    :param true_labels: Ground truth labels
    :type true_labels: list
    """
    permutation = np.random.permutation(len(embeddings)) # random shuffling
    X = np.take(embeddings, permutation, axis=0)
    y = np.take(true_labels, permutation, axis=0)

    # cross validation
    # scores = cross_validate(RandomForestClassifier(n_estimators=100), X, y, cv=5, scoring='accuracy', return_train_score=True)
    # print("Training accuracy for classification on embeddings: ", scores['train_score'].mean())
    # print("Test accuracy for classification on embeddings: ", scores['test_score'].mean())

    # random forest classifier with train.test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    print(f"Training accuracy for classification on embeddings using {'Graph2Vec' if GRAPH2VEC else 'Graphons'}: ", clf.score(X_train, y_train))
    print(f"Test accuracy for classification on embeddings using using {'Graph2Vec' if GRAPH2VEC else 'Graphons'}: ", clf.score(X_test, y_test), "\n")
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    return train_score, test_score 

    # no cross validation
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # clf=RandomForestClassifier(n_estimators=10)
    # clf.fit(X_train,y_train)
    # y_pred=clf.predict(X_test)
    # accuracy_classification = metrics.accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy_classification)
    


def clustering(graphs, true_labels, k=2, GRAPH2VEC=False):
    if GRAPH2VEC:
        adjusted_rand_score, error = graph2vec_clustering(li_emb = graphs, true_labels=true_labels, k=2)  
    else: 
        adjusted_rand_score, error = graphon_clustering(graphs, true_labels, num_clusters=k)
    # print(f'Adjusted Rand Score: {adjusted_rand_score} and Error: {error} for {k} clusters.') 
    return adjusted_rand_score, error