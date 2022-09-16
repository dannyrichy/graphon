from cProfile import label
from pyexpat import native_encoding
from utils import *
import scipy.sparse as sp
import networkx  as nx
from tqdm import tqdm

def load_reddit_big():
    data = np.load('reddit/reddit_data.npz')
    print('number of labels ',data['label'].shape)
    labels = data['label']
    adj = sp.load_npz('reddit/reddit_graph.npz')
    #reddit should now be a networkx graph
    reddit = nx.from_scipy_sparse_matrix(adj)
    return reddit, labels


def get_ego_nets(DOWNLOAD_REDDIT = False, hopping_dist = 1):    
    """
    Function used to generate the egonets

    :param DOWNLOAD_REDDIT: download the big reddit data or not
    :type DOWNLOAD_REDDIT: boolean

    :param hopping_dist: hopping distance while generating the ego nets
    :type hopping_dist: int

    :return: ego_graphs and labels
    """
    if DOWNLOAD_REDDIT:
        download_datasets(dataset_link=['https://data.dgl.ai/dataset/reddit.zip'])

    reddit, labels = load_reddit_big()
    graphs = []
    for i in tqdm(range(len(labels))):
        #looping through each node
        ego = i
        ego_net = nx.ego_graph(reddit, ego, radius = hopping_dist)
        adj = nx.adjacency_matrix(ego_net)
        adj = adj.todense()
        adj = torch.Tensor(np.asarray(adj)).to(device=device)

        graphs.append(adj)
        
    return graphs, labels


