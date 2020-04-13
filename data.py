import os


import pandas as pd
import numpy as np
import json
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score
import time
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from sklearn.model_selection import KFold
import gc
from tqdm import tqdm
from itertools import groupby,accumulate
from random import shuffle
from sklearn.model_selection import GroupKFold,GroupShuffleSplit,LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from pytorch_toolbelt import losses as L
import pandas as pd
import numpy as np
if torch.cuda.is_available():
    train = pd.read_csv('/local/ULIS/data/train_clean.csv')
    test = pd.read_csv('/local/ULIS/data/test_clean.csv')
else:
    train = pd.read_csv('./data/train_clean.csv')
    test = pd.read_csv('./data/test_clean.csv')

train['filter'] = 0
test['filter'] = 2
ts1 = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

ts1['time2'] = pd.cut(ts1['time'], bins=np.linspace(0.0000, 700., num=14 + 1), labels=list(range(14)),
                      include_lowest=True).astype(int)
ts1['time2'] = ts1.groupby('time2')['time'].rank() / 500000.

np.random.seed(321)
ts1['group'] = pd.cut(ts1['time'], bins=np.linspace(0.0000, 700., num=14 * 125 + 1), labels=list(range(14 * 125)),
                      include_lowest=True).astype(int)
np.random.seed(321)

y = ts1.loc[ts1['filter'] == 0, 'open_channels']
group = ts1.loc[ts1['filter'] == 0, 'group']
X = ts1.loc[ts1['filter'] == 0, 'signal']

np.random.seed(321)
skf = GroupKFold(n_splits=5)
splits = [x for x in skf.split(X, y, group)]

use_cols = [col for col in ts1.columns if col not in ['index', 'filter', 'group', 'open_channels', 'time', 'time2']]

# Create numpy array of inputs
for col in use_cols:
    col_mean = ts1[col].mean()
    ts1[col] = ts1[col].fillna(col_mean)

val_preds_all = np.zeros((ts1[ts1['filter'] == 0].shape[0], 11))
test_preds_all = np.zeros((ts1[ts1['filter'] == 2].shape[0], 11))

groups = ts1.loc[ts1['filter'] == 0, 'group']
times = ts1.loc[ts1['filter'] == 0, 'time']
new_splits = []
for sp in splits:
    new_split = []
    new_split.append(np.unique(groups[sp[0]]))
    new_split.append(np.unique(groups[sp[1]]))
    new_splits.append(new_split)

trainval = np.array(list(ts1[ts1['filter'] == 0].groupby('group').apply(lambda x: x[use_cols].values)))
test = np.array(list(ts1[ts1['filter'] == 2].groupby('group').apply(lambda x: x[use_cols].values)))
trainval_y = np.array(list(ts1[ts1['filter'] == 0].groupby('group').apply(lambda x: x[['open_channels']].values)))

trainval = trainval.transpose((0,2,1))
test = test.transpose((0,2,1))

trainval_y = trainval_y.reshape(trainval_y.shape[:2])
test_y = np.zeros((test.shape[0], trainval_y.shape[1]))

trainval = torch.Tensor(trainval)
test = torch.Tensor(test)


class IonDataset(Dataset):
    """Car dataset."""

    def __init__(self, data, labels, training=True, transform=None, flip=0.5, noise_level=0, class_split=0.0):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.training = training
        self.flip = flip
        self.noise_level = noise_level
        self.class_split = class_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        labels = self.labels[idx]
        if np.random.rand() < self.class_split:
            data, labels = class_split(data, labels)
        if np.random.rand() < self.noise_level:
            data = data * torch.FloatTensor(10000).uniform_(1 - self.noise_level, 1 + self.noise_level)
        if np.random.rand() < self.flip:
            data = torch.flip(data, dims=[1])
            labels = np.flip(labels, axis=0).copy().astype(int)

        return [data, labels.astype(int)]