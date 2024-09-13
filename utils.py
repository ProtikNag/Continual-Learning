import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs if applicable
    torch.backends.cudnn.deterministic = True  # CUDNN deterministic setting
    torch.backends.cudnn.benchmark = False  # Ensure reproducibility for CUDNN
