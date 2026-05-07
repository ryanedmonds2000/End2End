import pandas as pd
import numpy as np
from itertools import combinations
from scipy.optimize import curve_fit
import pyepo
from pyepo.model.grb import shortestPathModel
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset
from longestpath import longestPathModel

"""TODO: Utils and experiments currently hard coded to be compatible with 5x5 grid with 40 arcs and 70 edges - fix magic number references when switching to real world data"""

SYNTHETIC_FEATS = 3
SYNTHETIC_PATHS = 40

class BanditDataset(Dataset):
    """Dataset that randomly selects paths from a pool of valid paths."""
    
    def __init__(self, x, y, paths, rationality = None, seed=None, noise=0, numarcs=SYNTHETIC_PATHS):
        """
        Args:
            x: input features (num_samples, num_features)
            y: cost vectors (num_samples, num_edges)
            paths: array of valid paths (num_paths, num_edges)
            seed: random seed for reproducibility
            noise: perturbation to apply to observed costs (useful for importing misaligned data)
        """
        if seed is not None:
            np.random.seed(seed)


        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.allpaths = torch.FloatTensor(paths)
        zs = np.empty((0, numarcs))
        cs = np.empty((0, 1))


        for idx in range(len(self.x)):
            if rationality is not None: # Select path that aligns with rationality
                costs = [torch.dot(self.y[idx], path) for path in self.allpaths]
                worst_cost = max(costs)
                opt_cost = min(costs)
                target = (1-rationality) * (worst_cost - opt_cost) + opt_cost
                diffs = [abs(cost - target) for cost in costs]
                path_idx = np.argmin(diffs)
            else: # Select random path
                path_idx = np.random.choice(len(self.allpaths))
            z = self.allpaths[path_idx]
            c = torch.dot(self.y[idx], z) + np.random.uniform(low=-noise, high=noise)
            zs = np.vstack([zs, z])
            cs = np.vstack([cs, c])

        self.z = torch.FloatTensor(zs)
        self.c = torch.FloatTensor(cs)
        self.paths = np.unique(self.z.detach().numpy(), axis=0)
        
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        
        return self.x[idx], self.z[idx], self.c[idx]
    
    def append(self, imputeset):
        self.x = torch.concat([self.x, imputeset.x])
        self.z = torch.concat([self.z, imputeset.z])
        self.c = torch.concat([self.c, imputeset.c])
        self.y = torch.concat([self.y, imputeset.y])

    def get_unexplored(self):
        set_a = set(map(tuple, self.allpaths.detach().numpy()))
        set_b = set(map(tuple, self.paths))
        return np.array(list(set_a - set_b), dtype=int)

class GammaImputeDataset(Dataset):
    """Dataset that randomly selects paths from a pool of valid paths.
    Perturbs observed costs towards mean of distribution according to gamma [-1, 1]"""
    
    def __init__(self, x, y, paths, mean, gamma = 1, copies = 1, seed=None, noise=0, numfeats=SYNTHETIC_FEATS, numpaths=SYNTHETIC_PATHS):
        """
        Args:
            x: input features (num_samples, num_features)
            y: cost vectors (num_samples, num_edges)
            paths: array of valid paths (num_paths, num_edges)
            seed: random seed for reproducibility
            noise: perturbation to apply to observed costs (useful for importing misaligned data)
        """
        if seed is not None:
            np.random.seed(seed)


        xs = np.empty((0, numfeats))
        ys = np.empty((0, numpaths))
        self.allpaths = torch.FloatTensor(paths)
        zs = np.empty((0, numpaths))
        cs = np.empty((0, 1))


        for i in range(copies):
            for idx in range(len(x)):
                path_idx = np.random.choice(len(self.allpaths))
                z = self.allpaths[path_idx]
                observedy = mean + gamma * (y[idx] - mean)
                c = np.dot(observedy, z) + np.random.uniform(low=-noise, high=noise)
                zs = np.vstack([zs, z])
                # observed_c = mean + gamma * (c - mean)
                cs = np.vstack([cs, c])
                xs = np.vstack([xs, x[idx]])
                ys = np.vstack([ys, observedy])

        self.x = torch.FloatTensor(xs)
        self.y = torch.FloatTensor(ys)
        self.z = torch.FloatTensor(zs)
        self.c = torch.FloatTensor(cs)
        self.paths = np.unique(self.z.detach().numpy(), axis=0)    

class ValImputeDataset(Dataset):
    """Impute a specific value for the given paths"""
    
    def __init__(self, x, y, arcs, val = 0, copies = 1, seed=None, numfeats=SYNTHETIC_FEATS, numpaths=SYNTHETIC_PATHS):
        """
        Args:
            x: input features (num_samples, num_features)
            y: cost vectors (num_samples, num_edges)
            arcs: array of arcs to impute on
            val: value to impute
            copies: amount of data to impute (may be irrelevant?)
            seed: random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        xs = np.empty((0, numfeats))
        ys = np.empty((0, numpaths))
        zs = np.empty((0, numpaths))
        cs = np.empty((0, 1))

        for i in range(copies):
            for arc in arcs:
                idx = np.random.choice(len(x))

                path = np.zeros(numpaths)
                path[arc] = 1
                z = path
                c = val
                xs = np.vstack([xs, x[idx]])
                ys = np.vstack([ys, y[idx]])
                zs = np.vstack([zs, z])
                cs = np.vstack([cs, c])

        self.x = torch.FloatTensor(xs)
        self.y = torch.FloatTensor(ys)
        self.z = torch.FloatTensor(zs) 
        self.c = torch.FloatTensor(cs)
        
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        
        return self.x[idx], self.z[idx], self.c[idx]
    
    def append(self, imputeset):
        self.x = torch.concat([self.x, imputeset.x])
        self.z = torch.concat([self.z, imputeset.z])
        self.c = torch.concat([self.c, imputeset.c])
        self.y = torch.concat([self.y, imputeset.y])

class NuisanceRegression(nn.Module):
    # NN Module for learning nuisance function according to regularized least squares

    def __init__(self, num_features=SYNTHETIC_FEATS, num_edges=SYNTHETIC_PATHS, lam = 0):
        super(NuisanceRegression, self).__init__()
        self.linear = nn.Linear(num_features, num_edges)
        self.lam = lam

    def forward(self, x, path, target):
        # Linear layer to get potential costs for all arcs
        predicted_costs = self.linear(x)  # (batch_size, num_edges)
        # Dot product with binary path to get total path cost
        path_cost = torch.sum(predicted_costs * path, dim=1)  # (batch_size,)
        # Residual: predicted path cost - target
        residual = path_cost - target  # (batch_size,)
        # Mean squared error
        loss = torch.mean(residual**2) + torch.norm(self.linear.weight, p=2)
        return loss
    
class LinearRegression(nn.Module):
    # NN Module for decision focused learning of costs

    def __init__(self, num_features=SYNTHETIC_FEATS, num_edges=SYNTHETIC_PATHS):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_features, num_edges)

    def forward(self, x):
        out = self.linear(x)
        return out
    
def thetadr(x, z, c, nuisance_model, Sigma, reg=True): 
    # Approximates Y according to Hu et al.
    if reg == True:
        return nuisance_model.linear(torch.FloatTensor(x)) + torch.linalg.inv(Sigma + torch.eye(40)) @ z * (c - torch.dot(z, nuisance_model.linear(torch.FloatTensor(x))))
    else:
        return nuisance_model.linear(torch.FloatTensor(x)) + torch.linalg.pinv(Sigma) @ z * (c - torch.dot(z, nuisance_model.linear(torch.FloatTensor(x))))
    

def get_path(verts):
    row = 0
    col = 0
    outs = []
    for i in range(8):
        if i in verts:
            outs.append(9*row + col + 4)
            row += 1
        else:
            outs.append(9*row + col)
            col += 1
    z = np.zeros(40, dtype=int)
    z[outs] = 1
    return z

def get_paths(unexplorable = []):
    # Returns all paths that do not contain the unexplorable arcs
    exploredpaths = np.empty((0, SYNTHETIC_PATHS))
    unexploredpaths = np.empty((0, SYNTHETIC_PATHS))
    for comb in combinations(range(8), 4):
        path = get_path(comb)
        if np.any([path[i] == 1 for i in unexplorable]):
            unexploredpaths = np.vstack([unexploredpaths, path])
        else:
            exploredpaths = np.vstack([exploredpaths, path])
    return exploredpaths, unexploredpaths

def get_unexplored(dataset):
    return (sum(dataset.paths) == 0).nonzero()[0]

def get_unexplored_paths(dataset, allpaths):
    a = allpaths
    b = dataset.paths
    a_view = a.view([('', a.dtype)] * a.shape[1]).ravel()
    b_view = b.view([('', b.dtype)] * b.shape[1]).ravel()

    missing_rows = a[~np.isin(a_view, b_view)]
    return missing_rows

def f(x, theta): 
    # Underlying data generation method
    return theta[0] + x[0]*theta[1] + x[1]*theta[2] + x[2]*theta[3]

def getdata(n, a, w):
    theta = np.vstack([a, w])
    noise = np.random.uniform(low=-0.5, high=0.5, size = (n, 1))
    X = np.random.normal(size = (n, SYNTHETIC_FEATS))
    Y = np.empty((n, len(a)))
    for i in range(n):
        Y[i] = f(X[i], theta) + noise[i]
    return X, Y

def get_rationality(dataset):
    optmodel = shortestPathModel(grid=(5,5))
    irrationalmodel = longestPathModel(grid=(5,5))
    mycost = sum(dataset.c/len(dataset.c))
    tempset = pyepo.data.dataset.optDataset(optmodel, dataset.x, dataset.y)
    low = sum(tempset.objs/len(dataset.c))
    tempset = pyepo.data.dataset.optDataset(irrationalmodel, dataset.x, dataset.y)
    high = sum(tempset.objs/len(dataset.c))
    return (high - mycost)/(high - low)

def learn_nuisance(dataset, num_epochs, lam=0, num_features=SYNTHETIC_FEATS, num_edges=SYNTHETIC_PATHS):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    nuisance_model = NuisanceRegression(num_features=num_features, num_edges=num_edges, lam=lam)
    optimizer = torch.optim.Adam(nuisance_model.parameters(), lr=1e-3)
    print("\nLearning fnuisance function...", flush=True)
    for epoch in tqdm(range(num_epochs)):
        for x_batch, z_batch, c_batch in dataloader:
            # x_batch: (batch_size, 5)
            # z_batch: (batch_size, 40) - binary path vectors
            # c_batch: (batch_size,) - target costs
            
            loss = nuisance_model(x_batch, z_batch, c_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return nuisance_model

def learn_propensities(dataset, indices):
    path_indices = []

    for idx in indices:
        z = dataset.z[idx]
        diffs = [torch.sum((z - path)**2) for path in dataset.paths]
        path_indices.append(torch.argmin(torch.tensor(diffs)).item())

    clf = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='lbfgs')

    X = dataset.x[indices].numpy()
    clf.fit(X, path_indices)

    probs = clf.predict_proba(X)

    class_to_col = {cls: col for col, cls in enumerate(clf.classes_)}
    e_hats = [probs[i, class_to_col[path_indices[i]]] for i in range(len(indices))]

    return e_hats

def learn_gram(dataset, use_propensities=True, num_edges=SYNTHETIC_PATHS):
    # --- NEW: resolve base dataset + indices ---
    if hasattr(dataset, "indices"):  # it's a Subset
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = range(len(dataset))
    # ------------------------------------------
    if use_propensities:
        e_hats = learn_propensities(base_dataset, indices)

        Gram = torch.zeros(size=(num_edges, num_edges))
        weights = []

        for i, idx in enumerate(indices):
            if e_hats[i] > 0:
                weight = 1.0 / e_hats[i]
                z = base_dataset.z[idx]
                Gram += weight * z.reshape(num_edges, 1) @ z.reshape(1, num_edges)
                weights.append(weight)

        if weights:
            Gram /= sum(weights)

    else:
        Gram = torch.zeros(size=(num_edges, num_edges))
        for idx in indices:
            z = base_dataset.z[idx]
            Gram += z.reshape(num_edges, 1) @ z.reshape(1, num_edges)

        Gram /= len(indices)

    return Gram

def learn_dfl(dataset, testset, num_epochs, sigma, allregrets=False, optmodel=None, num_features=SYNTHETIC_FEATS, num_edges=SYNTHETIC_PATHS):
    optmodel = optmodel if optmodel is not None else shortestPathModel(grid=(5,5))
    pg = pyepo.func.perturbationGradient(optmodel, sigma, two_sides=True, processes=2) # decision focused loss term
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)
    predmodel = LinearRegression(num_features, num_edges)
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)
    trainregrets=[]
    testregrets=[]
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


