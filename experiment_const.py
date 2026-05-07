from utils import *
import random
from pyepo.data.dataset import optDataset
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import argparse
from sklearn.model_selection import train_test_split
import csv

def random_unexplored(seed = 42):
    random.seed(seed)
    exploredpaths = []
    while len(exploredpaths) == 0:
        unexplorable = random.sample(range(40), 4)
        exploredpaths, unexploredpaths = get_paths(unexplorable=unexplorable)
    return exploredpaths, unexploredpaths

def experiment_crossfold(X, Xtest, Y, Ytest, unexplorable, valratio = 1, inputrationality = None, imputecopies = 50, nuisance_epochs = 300, nuisance_lam = 1, dfl_epochs = 50, pgsigma = 0.5, dataseed=42, pathseed=42, folds=5):
    assert(len(X) % folds == 0)
    assert(imputecopies % folds == 0)
    optmodel = shortestPathModel(grid=(5,5))
    print("Generating Paths...")
    if unexplorable is not None:
        exploredpaths, unexploredpaths = get_paths(unexplorable=unexplorable)
    else:
        exploredpaths, unexploredpaths = random_unexplored(pathseed)

    
    dataset = BanditDataset(X, Y, exploredpaths, inputrationality, seed=dataseed)
    trainset = BanditDataset(X, Y, exploredpaths, inputrationality, seed=dataseed)
    rationality = get_rationality(dataset)
    unexplored = get_unexplored(dataset)
    print("Imputing Data...")
    val = valratio * max(dataset.c)
    imputeset = ValImputeDataset(X, Y, unexplored, val, copies=imputecopies, seed=dataseed) # Constant cost imputed data
    trainset.append(imputeset)

    kf = KFold(n_splits = folds, shuffle=True, random_state=dataseed)

    Ypred = torch.zeros((len(trainset.x), 40))

    print("Learning Nuisances...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(trainset)))):

        train_subset = Subset(trainset, train_idx)
        val_subset   = Subset(trainset, val_idx)

        # Your custom operation on the remainder
        nuisance = learn_nuisance(train_subset, nuisance_epochs, lam=nuisance_lam)
        gram = learn_gram(train_subset)

        Ypred_subset = torch.stack([
            thetadr(*val_subset[i], nuisance, gram, reg=False)
            for i in range(len(val_subset))
        ])
        Ypred[val_idx] = Ypred_subset


    testset = optDataset(optmodel, Xtest, Ytest)
    trainset = optDataset(optmodel, trainset.x, Ypred.detach().numpy())
    print("Learning Final Policy...")
    predmodel, trainregret, testregret = learn_dfl(trainset, testset, num_epochs=dfl_epochs, sigma=pgsigma, allregrets=False)

    return predmodel, trainregret, testregret, rationality

def main():
    parser = argparse.ArgumentParser(description="semisynthetic data experiment")

    parser.add_argument("--seed", type=int, help="random seed", default=42)
    seed = parser.parse_args().seed
    random.seed(seed)

    num_data = 2000
    num_feat = 3
    d=40
    w = np.random.uniform(low=0, high=1, size=(3, d))
    a = np.random.normal(loc=np.linspace(2, 6, d), scale=1, size=(d))

    X, Y = getdata(num_data, a, w)
    X, Xtest, Y, Ytest = train_test_split(X, Y, test_size=0.25)

    vals = list(np.linspace(0, 1, 11))
    inputrationalities = list(np.linspace(0, 1, 11))

    regrets = []
    for val in vals:
        for r in inputrationalities:
            print(f"R={r}, Val={val}...")
            predmodel, trainregret, testregret, rationality = experiment_crossfold(X, Xtest, Y, Ytest, [], valratio = val, inputrationality=r, dataseed=42, nuisance_lam=0, folds=2)
            regrets.append(testregret)
            print(f"R={r}, Val={val}, Regret={testregret}")

    n_cols = len(inputrationalities)

    with open(f"Results/regrets_table_{seed}.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # Header row: blank corner + rationality column labels
        writer.writerow(["valratio \\ rationality"] + inputrationalities)

        # Data rows: one per valratio
        for row_idx, valratio in enumerate(vals):
            row_values = regrets[row_idx * n_cols : (row_idx + 1) * n_cols]
            writer.writerow([valratio] + row_values)

    print(f"Saved to regrets_table_{seed}.csv")

if __name__ == "__main__":
    main()