"""
Configurations for the project
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")