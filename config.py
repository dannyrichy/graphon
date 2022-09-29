"""
Configurations
"""
import torch

DOWNLOAD_DATA = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIR = 'graph2vec_embeddings/'
DATA = {'DATASETS' : ['facebook_ct1', 'deezer_ego_nets', 'github_stargazers', 'REDDIT-BINARY'], 
        'SYNTHETIC_DATA' :  [str(i) for i in range(0,10)]}
CREATE_EMBEDDINGS = False
NUM_GRAPHONS = 10
NUM_GRAPHS_PER_GRAPHONS = 4

