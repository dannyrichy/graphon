"""
Configurations
"""
import torch

DOWNLOAD_DATA = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIR = 'graph2vec_embeddings/'
DATASETS = ['facebook_ct1', 'deezer_ego_nets', 'github_stargazers', 'REDDIT-BINARY']

