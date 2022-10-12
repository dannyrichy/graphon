import pickle
from pathlib import Path

import networkx as nx
from karateclub.graph_embedding import Graph2Vec




def load_embeddings(dir_name):
    """
    Loads the embeddings by name

    :return: a list containing all the embeddings and the corresponding labels
    """
    with open(f'{dir_name}/g2v_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    with open(f'{dir_name}/g2v_true_labels.pkl', 'rb') as f:
        true_labels = pickle.load(f)

    return embeddings, true_labels


def create_g2v_embeddings(graph_list, true_labels, dir_name):
    if not isinstance(graph_list, list) or len(graph_list) == 0:
        raise TypeError("Graph list is an empty list or is not of type list")

    g2v = Graph2Vec()
    g2v.fit([nx.from_numpy_array(graph.numpy()) for graph in graph_list])
    embeddings = g2v.get_embedding()
    parent_dir = Path(dir_name)
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    with open(parent_dir.joinpath('g2v_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)

    with open(parent_dir.joinpath('g2v_true_labels.pkl'), 'wb') as f:
        pickle.dump(true_labels, f)
