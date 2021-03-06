import torch
import pandas as pd
import numpy as np


def load_data(kalman_filter=False):
    if torch.cuda.is_available():
        if not kalman_filter:
            train = pd.read_csv('/local/ULIS/data/train_clean.csv')
            test = pd.read_csv('/local/ULIS/data/test_clean.csv')
        else:
            train = pd.read_csv('/local/ULIS/data/train_kalman.csv')
            test = pd.read_csv('/local/ULIS/data/test_kalman.csv')
    else:
        if not kalman_filter:
            train = pd.read_csv('./data/train_clean.csv')
            test = pd.read_csv('./data/test_clean.csv')
        else:
            train = pd.read_csv('./data/train_kalman.csv')
            test = pd.read_csv('./data/test_kalman.csv')

    train_signals = [None] * 10
    train_states = [None] * 10
    for i in range(10):
        train_signals[i] = train['signal'].values[i * 500000:(i + 1) * 500000]
        train_states[i] = train['open_channels'].values[i * 500000:(i + 1) * 500000]

    test_signals = [None] * 11
    for i in range(10):
        test_signals[i] = test['signal'].values[i * 100000:(i + 1) * 100000]

    test_signals[10] = test['signal'].values[1000000:2000000]

    train_states = np.array(train_states)
    train_signals = np.array(train_signals)
    train_groups = [[0, 1], [2, 6], [3, 7], [5, 8], [4, 9]]
    test_groups = [[0, 3, 8, 10], [4], [1, 9], [2, 6], [5, 7]]

    return train_states, train_signals, train_groups, test_signals, test_groups


def load_data_2():
    if not torch.cuda.is_available():
        train_data_path = 'data/train_detrend.npz'
        test_data_path = 'data/test_detrend.npz'
    else:
        train_data_path = '/local/ULIS/data/train_detrend.npz'
        test_data_path = '/local/ULIS/data/test_detrend.npz'

    with np.load(train_data_path, allow_pickle=True) as data:
        train_signals = data['train_signal']
        train_states = data['train_opench']

    with np.load(test_data_path, allow_pickle=True) as data:
        test_signals = data['test_signal']

    train_groups = [[0, 1, 2], [3, 7], [4, 8], [6, 9], [5, 10]]
    test_groups = [[0, 3, 8, 10, 11], [4], [1, 9], [2, 6], [5, 7]]

    return train_states, train_signals, train_groups, test_signals, test_groups


