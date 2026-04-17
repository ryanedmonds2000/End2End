import pandas as pd
import numpy as np
import math
import random
from itertools import combinations
import scipy
from scipy.optimize import curve_fit
import pyepo
from pyepo.model.grb import shortestPathModel
import gurobipy
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class BanditDataset(Dataset):
    """Dataset that randomly selects paths from a pool of valid paths."""
    
    def __init__(self, x, y, paths, seed=None):
        """
        Args:
            x: input features (num_samples, num_features)
            c: cost vectors (num_samples, num_edges)
            paths: array of valid paths (num_paths, num_edges)
            seed: random seed for reproducibility
        """
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.paths = torch.FloatTensor(paths)
        zs = np.empty((0, len(paths[0])))
        cs = np.empty((0, 1))

        for idx in range(len(self.x)):
            path_idx = np.random.choice(len(self.paths))
            z = self.paths[path_idx]
            c = torch.dot(self.y[idx], z)
            zs = np.vstack([zs, z])
            cs = np.vstack([cs, c])

        self.z = torch.FloatTensor(zs)
        self.c = torch.FloatTensor(cs)
        
        if seed is not None:
            np.random.seed(seed)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        
        return self.x[idx], self.z[idx], self.c[idx]
    
# class NuisanceDataset(Dataset):
#     """Dataset once nuisances have been computed."""
    
#     def __init__(self, x, y, paths, seed=None):
#         """
#         Args:
#             x: input features (num_samples, num_features)
#             c: cost vectors (num_samples, num_edges)
#             paths: array of valid paths (num_paths, num_edges)
#             seed: random seed for reproducibility
#         """
#         self.x = torch.FloatTensor(x)
#         self.y = torch.FloatTensor(y)
#         self.paths = torch.FloatTensor(paths)
#         zs = np.empty((0, len(paths[0])))
#         cs = np.empty((0, 1))

#         for idx in range(len(self.x)):
#             path_idx = np.random.choice(len(self.paths))
#             z = self.paths[path_idx]
#             c = torch.dot(self.y[idx], z)
#             zs = np.vstack([zs, z])
#             cs = np.vstack([cs, c])

#         self.z = torch.FloatTensor(zs)
#         self.c = torch.FloatTensor(cs)
        
#         if seed is not None:
#             np.random.seed(seed)
        
#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
        
#         return self.x[idx], self.z[idx], self.c[idx]

class NuisanceRegression(nn.Module):

    def __init__(self, num_features=5, num_edges=40):
        super(NuisanceRegression, self).__init__()
        self.linear = nn.Linear(num_features, num_edges)

    def forward(self, x, path, target):
        # Linear layer to get potential costs for all arcs
        predicted_costs = self.linear(x)  # (batch_size, num_edges)
        # Dot product with binary path to get total path cost
        path_cost = torch.sum(predicted_costs * path, dim=1)  # (batch_size,)
        # Residual: predicted path cost - target
        residual = path_cost - target  # (batch_size,)
        # Mean squared error
        loss = torch.mean(residual**2)
        return loss
    
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(5, 40)

    def forward(self, x):
        out = self.linear(x)
        return out
    
def thetadr(x, z, c, nuisance_model, Sigma, reg=True):
    if reg == True:
        return nuisance_model.linear(torch.FloatTensor(x)) + torch.linalg.inv(Sigma + torch.eye(40)) @ z * (c - torch.dot(z, nuisance_model.linear(torch.FloatTensor(x))))
    else:
        return nuisance_model.linear(torch.FloatTensor(x)) + torch.linalg.inv(Sigma) @ z * (c - torch.dot(z, nuisance_model.linear(torch.FloatTensor(x))))
    

