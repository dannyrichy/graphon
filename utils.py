import io
import zipfile
from functools import reduce

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
import torch
from clustering.graph2vec_clustering import graph2vec_clustering
from clustering.spectral_clustering import graphon_clustering
from config import DEVICE


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
    # load data_loader
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
    # for i in range(200):
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
    # print('histogram of number of nodes in ', name)
    # print(all_nodes)
    # plt.hist(all_nodes)
    # plt.title(f'Number of nodes for each graph in {name}')
    # plt.show()
    return graphs

def classification(embeddings, true_labels, GRAPH2VEC=False):
    """
    Classification of graph embeddings using Random Forests
    
    :param embeddings: List of embeddings
    :type embeddings: Union[list,numpy.ndarray]

    :param true_labels: Ground truth labels
    :type true_labels: list
    """
    print('Performing classification')
    permutation = np.random.permutation(len(embeddings))  # random shuffling
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



def clustering(graphs, true_labels, k, GRAPH2VEC=False, n_eigenvectors=2):
    if GRAPH2VEC:
        adjusted_rand_score, hungarian_error = graph2vec_clustering(li_emb=graphs, true_labels=true_labels, k=k, no_eig_vecs=n_eigenvectors)  
    else: 
        adjusted_rand_score, hungarian_error = graphon_clustering(graphs, true_labels, num_clusters=k, no_eig_vecs=n_eigenvectors)
    print(f"\nAdjusted Random Score --> {adjusted_rand_score}\nHungarian Score --> {hungarian_error} \nusing {'Graph2Vec' if GRAPH2VEC else 'Graphons'}\n")
    return adjusted_rand_score, hungarian_error


def update_config(sweep_config, config_def):
    """
    Updated the sweep config with the default config, adding the default values 

    :param sweep_config: the sweep config to be updated
    :type sweep_config: dict

    :param config_def: the default config
    :type config_def: dict

    :return: the updated sweep config
    :rtype: dict
    """
    # real_data_params = ['DOWNLOAD_DATA']
    # synth_data_params = ['NUM_GRAPHS_PER_GRAPHON', 'NUM_NODES', 'N0']

    # if config_def['SYNTH_DATA']:
    #     to_remove = real_data_params
    # else:
    #     to_remove = synth_data_params
    
    # for param in to_remove:
    #     if param in sweep_config['parameters']:
    #         del sweep_config['parameters'][param]
    #     del config_def[param]
    # del config_def['SYNTH_DATA']

    if config_def['SWEEP']:
        for key, value in config_def.items():
            if key not in sweep_config['parameters'].keys(): 
                sweep_config['parameters'][key] = {'value': value}
        return sweep_config
    return config_def