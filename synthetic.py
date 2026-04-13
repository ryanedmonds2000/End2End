import pandas as pd
import numpy as np
import math
import random
from itertools import combinations
import scipy
from scipy.optimize import curve_fit

d = 40

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

feasiblepaths = np.zeros(shape=(70, 40))
i=0
for comb in combinations(range(8), 4):
    feasiblepaths[i] = get_path(comb)
    i += 1

def f4(x, theta):
    return theta[0] + x[0]*theta[1] + x[1]*theta[2] + x[2]*theta[3]

def getdata4(n):
    w = np.random.uniform(low=0, high=1, size=(3, d))
    a = np.random.normal(loc=3, scale=1, size=(d))
    theta = np.vstack([a, w])
    noise = np.random.uniform(low=-0.5, high=0.5, size = (n, 1))
    X = np.random.normal(size = (n, 3))
    Y = np.empty((n, d))
    for i in range(n):
        Y[i] = f4(X[i], theta) + noise[i]

    

    C = np.zeros(shape=(n, 1))
    Z = np.zeros(shape=(n, 40))
    for i in range(n):
        path = feasiblepaths[np.random.choice(range(70))]
        Z[i] = path
        C[i, 0] = Y[i] @ path
    return X, Y, Z, C


def solmap(y):
    min = np.inf
    out = None
    for z in feasiblepaths:
        if np.dot(y, z) <= min:
            min = np.dot(y, z)
            out = z
    return out