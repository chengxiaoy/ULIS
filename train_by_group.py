import os
from torch.utils.data import DataLoader, Dataset
from data import get_data, normalize, run_feat_engineering, feature_selection, batching, IronDataset, EarlyStopping
from sklearn.model_selection import GroupKFold
import numpy as np
from model import Seq2SeqRnn
import torch
from pytorch_toolbelt import losses as L
from helper import EarlyStopping
import time
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
import numpy as np
from data import get_data
from attrdict import AttrDict
from model import getModel

EPOCHS = 90  # 150
NNBATCHSIZE = 32
GROUP_BATCH_SIZE = 4000
SEED = 123
LR = 0.001
SPLITS = 5
# model_name = 'wave_net'
model_name = 'seq2seq'
gpu_id = 1
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
data_type = "kalman_clean"  # raw clean kalman_clean
# data_fe = "shifted_proba"  # none "shifted"
data_fe = None  # none "shifted"
data_group = False
outdir = 'wavenet_models'
flip = False
noise = False
expriment_id = 2
config = AttrDict({'EPOCHS': EPOCHS, 'NNBATCHSIZE': NNBATCHSIZE, 'GROUP_BATCH_SIZE': GROUP_BATCH_SIZE, 'SEED': SEED,
                   'LR': LR, 'SPLITS': SPLITS, 'model_name': model_name, 'device': device, 'outdir': outdir,
                   'expriment_id': expriment_id, 'data_type': data_type, 'data_fe': data_fe, 'noise': noise,
                   'flip': flip})

# read data and batching
train, test, sub = get_data(config)
features = ['signal']
train = batching(train, batch_size=config.GROUP_BATCH_SIZE)
test = batching(test, batch_size=config.GROUP_BATCH_SIZE)

# data feature engineering
if data_fe is not None and data_fe == 'shifted_proba':
    train, test = normalize(train, test)
    train = run_feat_engineering(train)
    test = run_feat_engineering(test)
    train, test, features = feature_selection(train, test)

# cross valid
target = ['open_channels']
group = train['group']
kf = GroupKFold(n_splits=config.SPLITS)
splits = [x for x in kf.split(train, train[target], group)]
new_splits = []
for sp in splits:
    new_split = []
    new_split.append(np.unique(group[sp[0]]))
    new_split.append(np.unique(group[sp[1]]))
    new_split.append(sp[1])
    new_splits.append(new_split)

target_cols = ['open_channels']
train_tr = np.array(list(train.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
train = np.array(list(train.groupby('group').apply(lambda x: x[features].values)))
test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))


def train_(model, train_dataloader, valid_dataloader, criterion, optimizer, schedular, early_stopping, epoch_n,
           group_id, index, writer):
    for epoch in range(epoch_n):
        start_time = time.time()

        print("Epoch : {}".format(epoch))
        # print("learning_rate: {:0.9f}".format(schedular.get_lr()[0]))
        train_losses, valid_losses = [], []

        model.train()  # prep model for training
        train_preds, train_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            predictions = model(x)

            predictions_ = predictions.view(-1, predictions.shape[-1])
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training lossa
            train_losses.append(loss.item())

            train_true = torch.cat([train_true, y_], 0)
            train_preds = torch.cat([train_preds, predictions_], 0)
        schedular.step()

        model.eval()  # prep model for evaluation
        val_preds, val_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)
        with torch.no_grad():
            for x, y in valid_dataloader:
                x = x.to(device)
                y = y.to(device)

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
        print("train_loss: {:0.6f}, valid_loss: {:0.6f}".format(train_loss, valid_loss))

        train_true = train_true.cpu().detach().numpy()
        train_preds = train_preds.cpu().detach().numpy().argmax(1)
        train_score = f1_score(train_true, train_preds, labels=np.unique(train_true), average='macro')
        train_accurancy = np.sum(train_true == train_preds) / len(train_true)

        val_true = val_true.cpu().detach().numpy()
        val_preds = val_preds.cpu().detach().numpy().argmax(1)
        val_score = f1_score(val_true, val_preds, labels=np.unique(val_true), average='macro')
        val_accurancy = np.sum(val_true == val_preds) / len(val_true)

        print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))
        print("train_acc: {:0.6f}, valid_acc: {:0.6f}".format(train_accurancy, val_accurancy))

        writer.add_scalars('group_{}/cv_{}/loss'.format(group_id, index), {'train': train_loss, 'val': valid_loss},
                           epoch)
        writer.add_scalars('group_{}/cv_{}/f1_score'.format(group_id, index), {'train': train_score, 'val': val_score},
                           epoch)
        writer.add_scalars('group_{}/cv_{}/acc'.format(group_id, index),
                           {'train': train_accurancy, 'val': val_accurancy},
                           epoch)
        if early_stopping(valid_loss, model):
            print("Early Stopping...")
            print("Best Val Score: {:0.6f}".format(early_stopping.best_score))
            break

        print("--- %s seconds ---" % (time.time() - start_time))


train_groups = [[0, 1], [2, 6], [3, 7], [5, 8], [4, 9]]
test_groups = [[0, 3, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [4], [1, 9], [2, 6], [5, 7]]


def get_group_index(group_id, train_length, test_length):
    train_indexs = list(range(train_length))
    test_indexs = list(range(test_length))
    train_group = train_groups[group_id]
    test_group = test_groups[group_id]
    train_group_indexs = []
    test_group_indexs = []

    for grp in train_group:
        train_group_indexs.append(train_indexs[grp * (train_length // 10):(grp + 1) * (train_length // 10)])

    for grp in test_group:
        test_group_indexs.append(test_indexs[grp * (test_length // 20):(grp + 1) * (test_length // 20)])

    train_group_indexs = np.concatenate(train_group_indexs)
    test_group_indexs = np.concatenate(test_group_indexs)

    return train_group_indexs, test_group_indexs


writer = SummaryWriter(logdir=os.path.join("board/", str(config.expriment_id)))

pred = np.zeros([20, 100000])
for group_id in range(1):
    group_id = 4
    test_y = np.zeros([int(2000000 / config.GROUP_BATCH_SIZE), config.GROUP_BATCH_SIZE, 1])

    train_group_indexs, test_group_indexs = get_group_index(group_id, len(train), len(test))
    test_dataset = IronDataset(test[test_group_indexs], test_y[test_group_indexs], flip=False, noise_level=0.0)
    test_dataloader = DataLoader(test_dataset, config.NNBATCHSIZE, shuffle=False, num_workers=8, pin_memory=True)

    test_preds_all = np.zeros([len(test_groups[group_id]) * 100000, 11])

    for index in range(5):
        train_index, val_index,_ = new_splits[index]
        train_index = np.intersect1d(train_index, train_group_indexs)
        val_index = np.intersect1d(val_index, train_group_indexs)

        batchsize = 16
        train_dataset = IronDataset(train[train_index], train_tr[train_index], flip=False, noise_level=0.0)
        train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=8, pin_memory=True)

        valid_dataset = IronDataset(train[val_index], train_tr[val_index], flip=False)
        valid_dataloader = DataLoader(valid_dataset, batchsize, shuffle=False, num_workers=4, pin_memory=True)
        model = getModel(config)

        early_stopping = EarlyStopping(patience=20, is_maximize=False,
                                       checkpoint_path="./models/gru_clean_checkpoint_fold_{}_group_{}_exp_{}.pt".format(
                                           index,
                                           group_id, expriment_id))
        criterion = L.FocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10)
        train_(model, train_dataloader, valid_dataloader, criterion, optimizer, schedular, early_stopping, 100, group_id,
              index, writer)

        model.load_state_dict(
            torch.load(
                "./models/gru_clean_checkpoint_fold_{}_group_{}_exp_{}.pt".format(index, group_id, expriment_id)))
        with torch.no_grad():
            pred_list = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x)
                predictions_ = predictions.view(-1, predictions.shape[-1])

                pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())
            test_preds = np.vstack(pred_list)
        test_preds_all += test_preds

    group_pred = np.argmax(test_preds_all, axis=1)

    group_pred = group_pred.reshape(-1, 100000)
    print(group_pred.shape)

    assert group_pred.shape[0] == len(test_groups[group_id])
    # group_pred = group_pred / np.sum(group_pred, axis=1)[:, None]

    pred[test_groups[group_id]] = group_pred

pred = np.concatenate(pred)
ss = pd.read_csv("/local/ULIS/data/sample_submission.csv", dtype={'time': str})

test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': pred.astype(np.int)})
test_pred_frame.to_csv("./gru_preds_{}.csv".format(expriment_id), float_format='%.4f', index=False)
