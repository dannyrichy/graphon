"""
Configurations for the project
"""
import torch

# Flag for downloading the data
DOWNLOAD_DATA = False

# Flag for using GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# directory to store the graph2vec embeddings
EMBEDDING_DIR = 'graph2vec_embeddings/'

# Dictionary of real-world datasets and synthetic datasets
DATA = {'DATASETS' : ['facebook_ct1', 'deezer_ego_nets', 'github_stargazers', 'REDDIT-BINARY'], 
        'SYNTHETIC_DATA' :  [str(i) for i in range(0,10)]}

# Flag to create and store graph2vec embeddings
CREATE_EMBEDDINGS = False

# Number of graphons to create 
NUM_GRAPHONS = 10

# Number of graphs to create per graphon
NUM_GRAPHS_PER_GRAPHONS = 4

# Number of nodes in each synthetic graph
NUM_NODES = 300

