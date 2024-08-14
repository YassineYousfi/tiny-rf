import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset

def dino_dataset(n=8000, rng=np.random.default_rng(42)):
    df = pd.read_csv('datasaurus.csv')
    df = df[df.dataset == 'dino']
    ix = rng.integers(0, len(df), n)
    # add some noise to inflate the data
    x = np.array(df["x"].iloc[ix]) + rng.normal(size=n) * 0.15
    y = np.array(df["y"].iloc[ix]) + rng.normal(size=n) * 0.15
    # normalize...
    x, y = (x/54 - 1) * 4, (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))
