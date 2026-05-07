from utils import *  
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from pyepo.model.grb.grbmodel import optGrbModel
from pyepo.data.dataset import optDataset
import gurobipy as gp
from gurobipy import GRB
from torch.utils.data import DataLoader
from torch import nn
import torch

SYNTHETIC_FEATS = 197   # X_df after period one-hot encoding
SYNTHETIC_PATHS = 93    # number of network edges

PATHS_FILE = "data/paths.npy"
X_FILE = "data/X_twoyear.csv"
Y_FILE = "data/Y_twoyear.csv"

class UberPathModel(optGrbModel):
    """
    Shortest-path model for the Uber network:
        minimize c^T z
        subject to A z = b
                   0 <= z <= 1
    """
    def __init__(self, A_path="data/A_downtwon_1221to1256.csv",b_path="data/b_downtwon_1221to1256.csv"):
        self.A = pd.read_csv(A_path).to_numpy(dtype=float)
        self.b = pd.read_csv(b_path).to_numpy(dtype=float).reshape(-1)
        self.num_edges = self.A.shape[1]
        super().__init__()

    def _getModel(self):
        m = gp.Model("uber_shortest_path")
        m.Params.OutputFlag = 0
        z = m.addVars(
            range(self.num_edges),
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="z"
        )
        m.modelSense = GRB.MINIMIZE
        for i in range(self.A.shape[0]):
            m.addConstr(gp.quicksum(float(self.A[i, j]) * z[j] for j in range(self.num_edges))== float(self.b[i]))

        return m, z

def thetadr_uber(x, z, c, nuisance_model, Sigma, reg=True): 
    # Approximates Y according to Hu et al.
    x = torch.FloatTensor(x)
    fx = nuisance_model.linear(x)

    if reg == True:
        eye = torch.eye(SYNTHETIC_PATHS)
        return fx + torch.linalg.inv(Sigma + eye) @ z * (c - torch.dot(z, fx))
    else:
        return fx + torch.linalg.inv(Sigma) @ z * (c - torch.dot(z, fx))

def learn_dfl_uber(dataset, testset, num_epochs, sigma, allregrets=False):
    optmodel = UberPathModel()
    pg = pyepo.func.perturbationGradient(optmodel,sigma,two_sides=True,processes=1)
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)
    predmodel = LinearRegression()
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)
    trainregrets = []
    testregrets = []
    print("\nLearning final policy...", flush=True)
    for epoch in tqdm(range(num_epochs)):
        for data in trainloader:
            _x, _y, _w, _z = data
            # forward pass
            cp = predmodel(_x)
            # PG loss
            loss = pg(cp, _y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if allregrets:
            trainregrets.append(pyepo.metric.regret(predmodel, optmodel, trainloader))
            testregrets.append(pyepo.metric.regret(predmodel, optmodel, testloader))

    if not allregrets:
        trainregret = pyepo.metric.regret(predmodel, optmodel, trainloader)
        testregret = pyepo.metric.regret(predmodel, optmodel, testloader)

    if allregrets:
        return predmodel, trainregrets, testregrets
    else:
        return predmodel, trainregret, testregret


def getdata_uber():
    X_df = pd.read_csv(X_FILE)
    Y_df = pd.read_csv(Y_FILE)
    X_df_num = pd.get_dummies(X_df, columns=["Period"], dtype=int) #1-hot encoding of the period (only text label)
    X = X_df_num.to_numpy()
    Y = Y_df.to_numpy()
    
    return X, Y

def load_paths_uber():
    paths = np.load(PATHS_FILE)
    return paths

def get_paths_uber(unexplorable=[]):
    paths = load_paths_uber()
    mask = np.any(paths[:, unexplorable] == 1, axis=1)
    exploredpaths = paths[~mask]
    unexploredpaths = paths[mask]

    return exploredpaths, unexploredpaths

def get_rationality_uber(dataset):
    allpaths = dataset.allpaths       
    Y = dataset.y                     
    cost_matrix = allpaths @ Y.T      
    low = torch.min(cost_matrix, dim=0).values.mean()
    high = torch.max(cost_matrix, dim=0).values.mean()
    mycost = dataset.c.mean()
    return ((high - mycost) / (high - low)).item()