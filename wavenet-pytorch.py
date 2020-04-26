import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import random
from sklearn.model_selection import GroupKFold
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
from sklearn.metrics import accuracy_score, f1_score

# configurations and main hyperparammeters
EPOCHS = 90  # 150
NNBATCHSIZE = 32
GROUP_BATCH_SIZE = 4000
SEED = 123
LR = 0.001
SPLITS = 5

outdir = 'wavenet_models'
flip = False
noise = False

if not os.path.exists(outdir):
    os.makedirs(outdir)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# read data
def read_data():
    train = pd.read_csv('data/train_clean_kalman.csv',
                        dtype={'time': np.float32, 'signal': np.float32, 'open_channels': np.int32})
    test = pd.read_csv('data/test_clean_kalman.csv',
                       dtype={'time': np.float32, 'signal': np.float32})
    # from https://www.kaggle.com/sggpls/wavenet-with-shifted-rfc-proba and
    # https://www.kaggle.com/c/liverpool-ion-switching/discussion/144645
    Y_train_proba = np.load("data/Y_train_proba.npy")
    Y_test_proba = np.load("data/Y_test_proba.npy")

    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]

    sub = pd.read_csv('data/sample_submission.csv', dtype={'time': np.float32})
    return train, test, sub


# create batches of 4000 observations
def batching(df, batch_size):
    # print(df)
    df['group'] = df.groupby(df.index // batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df


# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test


def lag_with_pct_change(df, windows):
    for window in windows:
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df


def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size=batch_size)
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1, 2, 3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
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


def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis=0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features


def split(GROUP_BATCH_SIZE=4000, SPLITS=5):
    print('Reading Data Started...')
    train, test, sample_submission = read_data()
    train, test = normalize(train, test)
    print('Reading and Normalizing Data Completed')
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train, batch_size=GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size=GROUP_BATCH_SIZE)
    train, test, features = feature_selection(train, test)
    print(train.head())
    print('Feature Engineering Completed...')

    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=SPLITS)
    splits = [x for x in kf.split(train, train[target], group)]
    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])
        new_splits.append(new_split)
    target_cols = ['open_channels']
    print(train.head(), train.shape)
    train_tr = np.array(list(train.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[features].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))
    print(train.shape, test.shape, train_tr.shape)
    return train, test, train_tr, new_splits


# from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver
class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = F.tanh(self.filter_convs[i](x)) * F.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            # x += res
            res = torch.add(res, x)
        return res


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 128
        self.LSTM1 = nn.GRU(input_size=19, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        # self.attention = Attention(input_size,4000)
        # self.rnn = nn.RNN(input_size, 64, 2, batch_first=True, nonlinearity='relu')

        self.wave_block1 = Wave_Block(128, 16, 12)
        self.wave_block2 = Wave_Block(16, 32, 8)
        self.wave_block3 = Wave_Block(32, 64, 4)
        self.wave_block4 = Wave_Block(64, 128, 1)
        self.fc = nn.Linear(128, 11)

    def forward(self, x):
        x, _ = self.LSTM1(x)
        x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        # x,_ = self.LSTM(x)
        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x, _ = self.LSTM(x)
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.rnn(x)
        # x = self.attention(x)
        x = self.fc(x)
        return x


class EarlyStopping:
    def __init__(self, patience=7, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def __call__(self, score, model):
        if self.best_score is None or \
                (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            torch.save(model.state_dict(), self.checkpoint_path)
            self.best_score, self.counter = score, 0
            return 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return 2
        return 0


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


train, test, train_tr, new_splits = split()
from torchcontrib.optim import SWA
import torchcontrib

model = Classifier()

test_y = np.zeros([int(2000000 / GROUP_BATCH_SIZE), GROUP_BATCH_SIZE, 1])
test_dataset = IronDataset(test, test_y, flip=False)
test_dataloader = DataLoader(test_dataset, NNBATCHSIZE, shuffle=False)
test_preds_all = np.zeros((2000000, 11))

oof_score = []
for index, (train_index, val_index, _) in enumerate(new_splits[0:], start=0):
    print("Fold : {}".format(index))
    train_dataset = IronDataset(train[train_index], train_tr[train_index], seq_len=GROUP_BATCH_SIZE, flip=flip,
                                noise_level=noise)
    train_dataloader = DataLoader(train_dataset, NNBATCHSIZE, shuffle=True, num_workers=16)

    valid_dataset = IronDataset(train[val_index], train_tr[val_index], seq_len=GROUP_BATCH_SIZE, flip=False)
    valid_dataloader = DataLoader(valid_dataset, NNBATCHSIZE, shuffle=False)

    it = 0
    model = Classifier()
    model = model.cuda()

    early_stopping = EarlyStopping(patience=40, is_maximize=True,
                                   checkpoint_path=os.path.join(outdir,
                                                                "gru_clean_checkpoint_fold_{}_iter_{}.pt".format(index,
                                                                                                                 it)))

    weight = None  # cal_weights()
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=2, swa_lr=0.0011)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.2)

    avg_train_losses, avg_valid_losses = [], []

    for epoch in range(EPOCHS):

        train_losses, valid_losses = [], []
        tr_loss_cls_item, val_loss_cls_item = [], []

        model.train()  # prep model for training
        train_preds, train_true = torch.Tensor([]).cuda(), torch.LongTensor([]).cuda()  # .to(device)

        print('**********************************')
        print("Folder : {} Epoch : {}".format(index, epoch))
        print("Curr learning_rate: {:0.9f}".format(optimizer.param_groups[0]['lr']))

        # loss_fn(model(input), target).backward()
        for x, y in tqdm(train_dataloader):
            x = x.cuda()
            y = y.cuda()
            # print(x.shape)

            optimizer.zero_grad()
            predictions = model(x)

            predictions_ = predictions.view(-1, predictions.shape[-1])
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            schedular.step(loss)
            # record training lossa
            train_losses.append(loss.item())
            train_true = torch.cat([train_true, y_], 0)
            train_preds = torch.cat([train_preds, predictions_], 0)

        # model.eval()  # prep model for evaluation
        optimizer.update_swa()
        optimizer.swap_swa_sgd()
        val_preds, val_true = torch.Tensor([]).cuda(), torch.LongTensor([]).cuda()
        print('EVALUATION')
        with torch.no_grad():
            for x, y in tqdm(valid_dataloader):
                x = x.cuda()  # .to(device)
                y = y.cuda()  # ..to(device)

                predictions = model(x)
                predictions_ = predictions.view(-1, predictions.shape[-1])
                y_ = y.view(-1)

                loss = criterion(predictions_, y_)

                valid_losses.append(loss.item())

                val_true = torch.cat([val_true, y_], 0)
                val_preds = torch.cat([val_preds, predictions_], 0)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("train_loss: {:0.6f}, valid_loss: {:0.6f}".format(train_loss, valid_loss))

        train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1),
                               labels=list(range(11)), average='macro')

        val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu().detach().numpy().argmax(1),
                             labels=list(range(11)), average='macro')

        schedular.step(val_score)
        print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))
        res = early_stopping(val_score, model)
        # print('fres:', res)
        if res == 2:
            print("Early Stopping")
            print('folder %d global best val max f1 model score %f' % (index, early_stopping.best_score))
            break
        elif res == 1:
            print('save folder %d global val max f1 model score %f' % (index, val_score))
    print('Folder {} finally best global max f1 score is {}'.format(index, early_stopping.best_score))
    oof_score.append(round(early_stopping.best_score, 6))

    model.eval()
    pred_list = []
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x = x.cuda()
            y = y.cuda()

            predictions = model(x)
            predictions_ = predictions.view(-1, predictions.shape[-1])  # shape [128, 4000, 11]
            # print(predictions.shape, F.softmax(predictions_, dim=1).cpu().numpy().shape)
            pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())  # shape (512000, 11)
            # a = input()
        test_preds = np.vstack(pred_list)  # shape [2000000, 11]
        test_preds_all += test_preds

print('all folder score is:%s' % str(oof_score))
print('OOF mean score is: %f' % (sum(oof_score) / len(oof_score)))
print('Generate submission.............')
submission_csv_path = '/kaggle/input/liverpool-ion-switching/sample_submission.csv'
ss = pd.read_csv(submission_csv_path, dtype={'time': str})
test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("./gru_preds.csv", index=False)
print('over')
