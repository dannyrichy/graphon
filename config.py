"""
Configurations
"""
import torch

DOWNLOAD_DATA = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

