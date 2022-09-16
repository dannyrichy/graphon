"""
Ego net functions
"""
import logging

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from graphon.config import DEVICE
from utils import download_datasets

logger = logging.getLogger(__name__)


def load_reddit_big(data_path='reddit/reddit_data.npz', adj_path='reddit/reddit_graph.npz'):
    data = np.load(data_path)
    logging.info('number of labels ', data['label'].shape)
    labels = data['label']
    adj = sp.load_npz(adj_path)
    # reddit should now be a networkx graph
    reddit = nx.from_scipy_sparse_matrix(adj)
    return reddit, labels


def get_ego_nets(download_data=False, hopping_dist=1):
    """
    Function used to generate the egonets

    :param download_data: download the big reddit data or not
    :type download_data: bool

    :param hopping_dist: hopping distance while generating the ego nets
    :type hopping_dist: int

    :return: ego_graphs and labels
    """
    if download_data:
        logger.info("Downloading and extracting graph data")
        download_datasets(dataset_links=['https://data.dgl.ai/dataset/reddit.zip'])

    reddit, labels = load_reddit_big()
    graphs = []
    logger.info("Creating graph datasets")
    for i in tqdm(range(len(labels))):
        # looping through each node
        ego = i
        ego_net = nx.ego_graph(reddit, ego, radius=hopping_dist)
        adj = nx.adjacency_matrix(ego_net)
        adj = adj.todense()
        adj = torch.Tensor(np.asarray(adj)).to(device=DEVICE)

        graphs.append(adj)

    return graphs, labels
