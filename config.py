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

# Flag to create and store graph2vec embeddings
CREATE_EMBEDDINGS = True

# Number of graphons to create 
NUM_GRAPHONS = 4

# Dictionary of real-world datasets and synthetic datasets
DATA = {'DATASETS' : ['facebook_ct1', 'deezer_ego_nets', 'github_stargazers', 'REDDIT-BINARY'], 
        'SYNTHETIC_DATA' :  [str(i) for i in range(0, NUM_GRAPHONS)]}

# Number of graphs to create per graphon
NUM_GRAPHS_PER_GRAPHONS = 500

# Number of nodes in each synthetic graph (None to make the number of nodes random for each graph)
NUM_NODES = None

# Flag to store the graphs generated from graphons
SAVE_GRAPHONS = False

# Directory to store the graphs generated from graphons
SAVE_GRAPHONS_LOC = f'./graphons_dir/{NUM_GRAPHONS}_graphons_{NUM_GRAPHS_PER_GRAPHONS}_graphs.pkl'
