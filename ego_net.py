from cProfile import label
from utils import *
import scipy.sparse as sp
import networkx  as nx

def load_reddit_big():
    data = np.load('reddit/reddit_data.npz')
    print('number of labels ',data['label'].shape)
    labels = data['label']
    adj = sp.load_npz('reddit/reddit_graph.npz')
    #reddit should now be a networkx graph
    reddit = nx.from_scipy_sparse_matrix(adj)
    return reddit, labels


def get_ego_nets(DOWNLOAD_REDDIT = False, hopping_dist = 1):    
    if DOWNLOAD_REDDIT:
        download_datasets(dataset_link=['https://data.dgl.ai/dataset/reddit.zip'])

    reddit, labels = load_reddit_big()
    for i in range(len(labels)):
        #looping through each node
        ego = i
        ego_net = nx.ego_graph(reddit, ego, radius = hopping_dist)

