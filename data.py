import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np


def get_data(config):
    train = None
    test = None
    if config.data_type == 'raw':
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')

    elif config.data_type == 'clean':
        train = pd.read_csv('data/train_clean.csv')
        test = pd.read_csv('data/test_clean.csv')

    elif config.data_type == 'kalman_clean':
        train = pd.read_csv('data/train_clean_kalman.csv')
        test = pd.read_csv('data/test_clean_kalman.csv')

    if config.gaussian_noise:
        train_noise = np.random.normal(0, config.gaussian_noise_std, 500 * 10000)
        test_noise = np.random.normal(0, config.gaussian_noise_std, 200 * 10000)
        train['signal'] = train['signal'].values + train_noise
        test['signal'] = test['signal'].values + test_noise

    if config.data_fe == 'shifted_proba':
        Y_train_proba = np.load("data/Y_train_proba.npy")
        Y_test_proba = np.load("data/Y_test_proba.npy")

        for i in range(11):
            train[f"proba_{i}"] = Y_train_proba[:, i]
            test[f"proba_{i}"] = Y_test_proba[:, i]

    sub = pd.read_csv('data/sample_submission.csv')
    return train, test, sub


def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test


def batching(df, batch_size):
    # print(df)
    df['group'] = df.groupby(df.index // batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df


def lag_with_pct_change(df, windows):
    for window in windows:
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df


def run_feat_engineering(df):
    # create batches
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1, 2, 3])

    df = create_rolling_features(df, [3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    return df


def create_rolling_features(df, WINDOWS):
    for window in WINDOWS:
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
        df["rolling_min_max_ratio_" + str(window)] = df["rolling_min_" + str(window)] / df["rolling_max_" + str(window)]
        df["rolling_min_max_diff_" + str(window)] = df["rolling_max_" + str(window)] - df["rolling_min_" + str(window)]

    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    return df


def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis=0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features


class IronDataset(Dataset):
    def __init__(self, data, labels, training=True, transform=None, seq_len=5000, flip=0.5, noise_level=0,
                 class_split=0.0):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.training = training
        self.flip = flip
        self.noise_level = noise_level
        self.class_split = class_split
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        labels = self.labels[idx]

        return [data.astype(np.float32), labels.astype(int)]


class EarlyStopping:
    def __init__(self, patience=7, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def save_model(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)

    def __call__(self, score, model):
        if self.best_score is None or \
                (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            self.save_model(model)
            self.best_score, self.counter = score, 0
            return 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return 2
        return 0
