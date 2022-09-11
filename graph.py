import networkx as nx
import numpy as np

def load_graph(path):
    graph = nx.Graph()
    adjacency_mat = np.loadtxt(path)
    graph.add_edges_from(adjacency_mat)

    return graph
