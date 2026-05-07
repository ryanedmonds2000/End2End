from utils import *
from utilsUber import *
import random
from pyepo.data.dataset import optDataset
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import argparse
from sklearn.model_selection import train_test_split
import csv
import json

UBER_FEATS = 197   # X_df after period one-hot encoding
UBER_PATHS = 93    # number of network edges

PATHS_FILE = "data/paths.npy"
X_FILE = "data/X_twoyear.csv"
Y_FILE = "data/Y_twoyear.csv"

def experiment_crossfold(X, Xtest, Y, Ytest, unexplorable = [],  gamma = 0, inputrationality = None, imputecopies = 50, nuisance_epochs = 50, nuisance_lam = 1, dfl_epochs = 50, pgsigma = 0.5, dataseed=42, pathseed=42, folds=5):
    assert(len(X) % folds == 0)
    optmodel = UberPathModel()
    print("Generating Paths...")
    exploredpaths, unexploredpaths = get_paths_uber(unexplorable=unexplorable)



    trainset = BanditDataset(X, Y, exploredpaths, inputrationality, seed=dataseed, numarcs=UBER_PATHS)
    rationality = get_rationality_uber(trainset)
    unexplored = trainset.get_unexplored()
    print("Imputing Data...")
    mean = trainset.y.mean().numpy()
    imputeset = GammaImputeDataset(X, Y, unexplored, mean, gamma=gamma, copies=imputecopies, seed=dataseed, numfeats=UBER_FEATS, numpaths=UBER_PATHS) # Constant cost imputed data
    trainset.append(imputeset)

    kf = KFold(n_splits = folds, shuffle=True, random_state=dataseed)

    Ypred = torch.zeros((len(trainset.x), UBER_PATHS))

    print("Learning Nuisances...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(trainset)))):

        train_subset = Subset(trainset, train_idx)
        val_subset   = Subset(trainset, val_idx)

        # Your custom operation on the remainder
        nuisance = learn_nuisance(train_subset, nuisance_epochs, lam=0, num_features=UBER_FEATS, num_edges=UBER_PATHS)
        gram = learn_gram(train_subset, num_edges=UBER_PATHS)

        Ypred_subset = torch.stack([
            thetadr(*val_subset[i], nuisance, gram, reg=False)
            for i in range(len(val_subset))
        ])
        Ypred[val_idx] = Ypred_subset


    testset = optDataset(optmodel, Xtest, Ytest)
    trainset = optDataset(optmodel, trainset.x, Ypred.detach().numpy())
    print("Learning Final Policy...")
    predmodel, trainregret, testregret = learn_dfl(trainset, testset, num_epochs=dfl_epochs, sigma=pgsigma, allregrets=False, optmodel=optmodel, num_features=UBER_FEATS, num_edges=UBER_PATHS)

    return predmodel, trainregret, testregret, rationality

def main():
    parser = argparse.ArgumentParser(description="semisynthetic data experiment")

    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--gamma", type=float, help="impute gamma", default=0.)
    parser.add_argument("--rat", type=float, help="rationality", default=1.)
    seed = parser.parse_args().seed
    gamma = parser.parse_args().gamma
    rat = parser.parse_args().rat
    random.seed(seed)


    
    X, Y = getdata_uber()
    X, Xtest, Y, Ytest = train_test_split(X, Y, train_size=900)

    print(f"R={rat}, G={gamma}...")
    predmodel, trainregret, testregret, rationality = experiment_crossfold(X, Xtest, Y, Ytest, [], gamma=gamma, inputrationality=rat, dataseed=seed, nuisance_lam=0, folds=2, imputecopies=1)



    with open(f"ResultsUberConst/regret_g{gamma}_r{rat}_s{seed}.json", "w") as f:
        json.dump(testregret, f)

    print(f"Saved to ResultsUberConst/regret_g{gamma}_r{rat}_s{seed}.json")

if __name__ == "__main__":
    main()