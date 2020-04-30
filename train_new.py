from attrdict import AttrDict
import torch
from data import get_data, normalize, run_feat_engineering, feature_selection, batching, IronDataset, EarlyStopping
from sklearn.model_selection import GroupKFold
import numpy as np
from model import getModel
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from torchcontrib.optim import SWA
import torchcontrib
import os
import torch.nn.functional as F
import pandas as pd
from tensorboardX import SummaryWriter
import random
from pytorch_toolbelt import losses as L
import json
import time


def buildConfig(gpu_id):
    EPOCHS = 100  # 150
    NNBATCHSIZE = 32
    GROUP_BATCH_SIZE = 4000
    SEED = 123
    LR = 0.001
    SPLITS = 5
    model_name = 'wave_net'
    gpu_id = gpu_id
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    data_type = "kalman_clean"  # raw clean kalman_clean
    data_fe = "shifted_proba"  # none "shifted"
    outdir = 'wavenet_models'
    flip = False
    noise = False
    expriment_id = 0
    loss = 'focal'  # ce or focal
    schedular = 'reduce'  # cos
    use_swa = False
    use_cbr = False

    group_train = False
    config = AttrDict({'EPOCHS': EPOCHS, 'NNBATCHSIZE': NNBATCHSIZE, 'GROUP_BATCH_SIZE': GROUP_BATCH_SIZE, 'SEED': SEED,
                       'LR': LR, 'SPLITS': SPLITS, 'model_name': model_name, 'device': device, 'outdir': outdir,
                       'expriment_id': expriment_id, 'data_type': data_type, 'data_fe': data_fe, 'noise': noise,
                       'flip': flip, 'group_train': group_train, 'loss': loss, "schedular": schedular,
                       'use_swa': use_swa,'use_cbr':use_cbr})
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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


# seed_everything(config.SEED)
def get_data_loader(config, group_id=None):
    # read data and batching
    train, test, sub = get_data(config)
    features = ['signal']
    train = batching(train, batch_size=config.GROUP_BATCH_SIZE)
    test = batching(test, batch_size=config.GROUP_BATCH_SIZE)

    # data feature engineering
    if config.data_fe is not None and config.data_fe == 'shifted_proba':
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
    train_dataloaders = []
    valid_dataloaders = []
    test_dataloaders = []
    for index, (train_index, val_index, _) in enumerate(new_splits[0:], start=0):
        # build dataloader
        test_y = np.zeros([int(2000000 / config.GROUP_BATCH_SIZE), config.GROUP_BATCH_SIZE, 1])
        test_dataset = IronDataset(test, test_y, flip=False)
        if group_id is not None:
            train_group_indexs, test_group_indexs = get_group_index(group_id, len(train), len(test))
            train_index = np.intersect1d(train_index, train_group_indexs)
            val_index = np.intersect1d(val_index, train_group_indexs)
            test_dataset = IronDataset(test[test_group_indexs], test_y[test_group_indexs], flip=False, noise_level=0.0)

        test_dataloader = DataLoader(test_dataset, config.NNBATCHSIZE, shuffle=False)
        train_dataset = IronDataset(train[train_index], train_tr[train_index], seq_len=config.GROUP_BATCH_SIZE,
                                    flip=config.flip,
                                    noise_level=config.noise)
        train_dataloader = DataLoader(train_dataset, config.NNBATCHSIZE, shuffle=True, num_workers=16)

        valid_dataset = IronDataset(train[val_index], train_tr[val_index], seq_len=config.GROUP_BATCH_SIZE, flip=False)
        valid_dataloader = DataLoader(valid_dataset, config.NNBATCHSIZE, shuffle=False)

        train_dataloaders.append(train_dataloader)
        valid_dataloaders.append(valid_dataloader)
        test_dataloaders.append(test_dataloader)
    return train_dataloaders, valid_dataloaders, test_dataloaders


def train_(model, train_dataloader, valid_dataloader, early_stopping,
           group_id, index, config):
    writer = SummaryWriter(logdir=os.path.join("board/", str(config.expriment_id)))
    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model)
    if config.use_swa:
        optimizer = torchcontrib.optim.SWA(optimizer)
    schedular = get_schedular(config, optimizer, len(train_dataloader))

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        print("Epoch : {}".format(epoch))
        # print("learning_rate: {:0.9f}".format(schedular.get_lr()[0]))
        train_losses, valid_losses = [], []

        model.train()  # prep model for training
        train_preds, train_true = torch.Tensor([]).to(config.device), torch.LongTensor([]).to(config.device)
        ii = 0
        for x, y in train_dataloader:
            x = x.to(config.device)
            y = y.to(config.device)

            optimizer.zero_grad()
            predictions = model(x)

            predictions_ = predictions.view(-1, predictions.shape[-1])
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            if config.schedular == 'cos' or config.schedular == 'cyc':
                schedular.step()
            if config.use_swa:
                if ii >= 10 and ii % 2 == 0:
                    optimizer.update_swa()
                ii += 1
            # record training lossa
            train_losses.append(loss.item())

            train_true = torch.cat([train_true, y_], 0)
            train_preds = torch.cat([train_preds, predictions_], 0)
        if config.use_swa:
            optimizer.swap_swa_sgd()

        model.eval()  # prep model for evaluation
        val_preds, val_true = torch.Tensor([]).to(config.device), torch.LongTensor([]).to(config.device)
        with torch.no_grad():
            for x, y in valid_dataloader:
                x = x.to(config.device)
                y = y.to(config.device)

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

        if config.schedular == 'reduce':
            schedular.step(val_score)
        val_accurancy = np.sum(val_true == val_preds) / len(val_true)

        print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))
        # print("train_acc: {:0.6f}, valid_acc: {:0.6f}".format(train_accurancy, val_accurancy))

        writer.add_scalars('group_{}/cv_{}/loss'.format(group_id, index), {'train': train_loss, 'val': valid_loss},
                           epoch)
        writer.add_scalars('group_{}/cv_{}/f1_score'.format(group_id, index), {'train': train_score, 'val': val_score},
                           epoch)
        # writer.add_scalars('group_{}/cv_{}/acc'.format(group_id, index),
        #                    {'train': train_accurancy, 'val': val_accurancy},
        #                    epoch)
        if early_stopping(val_score, model) == 2:
            print("Early Stopping...")
            print("Best Val Score: {:0.6f}".format(early_stopping.best_score))
            writer.add_text("val_score", "valid_f1_score_{}".format(val_score), index)
            break

        print("--- %s seconds ---" % (time.time() - start_time))


def get_criterion(config):
    if config.loss == 'focal':
        criterion = L.FocalLoss()
        return criterion
    elif config.loss == 'ce':
        return torch.nn.CrossEntropyLoss()
    return None


def get_optimizer(config, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    return optimizer


def get_schedular(config, optimizer, loader_n):
    if config.schedular == 'cos':
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=loader_n)

        return schedular
    elif config.schedular == 'reduce':
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.2)
        return schedular

    elif config.schedular == 'cyc':
        schedular = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001,
                                                      step_size_up=loader_n // 2, cycle_momentum=False)
        return schedular


def train_epoch_group(config):
    oof_score = []
    seed_everything(config.SEED)
    pred = np.zeros([20, 100000])
    if config.group_train:
        for group_id in range(5):
            train_dataloaders, valid_dataloaders, test_dataloaders = get_data_loader(config, group_id)

            test_preds_all = np.zeros([len(test_groups[group_id]) * 100000, 11])
            index = 0
            for train_dataloader, valid_dataloader, test_dataloader in zip(train_dataloaders, valid_dataloaders,
                                                                           test_dataloaders):
                early_stopping = EarlyStopping(patience=15, is_maximize=True,
                                               checkpoint_path="./models/gru_clean_checkpoint_fold_{}_group_{}_exp_{}.pt".format(
                                                   index,
                                                   group_id, config.expriment_id))
                model = getModel(config)
                train_(model, train_dataloader, valid_dataloader, early_stopping, group_id, index, config)
                early_stopping.load_best_weights(model)
                pred_list = []
                with torch.no_grad():
                    for x, y in tqdm(test_dataloader):
                        x = x.to(config.device)  # .to(device)
                        y = y.to(config.device)  # ..to(device)
                        predictions = model(x)
                        predictions_ = predictions.view(-1, predictions.shape[-1])  # shape [128, 4000, 11]
                        # print(predictions.shape, F.softmax(predictions_, dim=1).cpu().numpy().shape)
                        pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())  # shape (512000, 11)
                        # a = input()
                test_preds = np.vstack(pred_list)  # shape [2000000, 11]
                test_preds_all += test_preds
                index += 1
            group_pred = np.argmax(test_preds_all, axis=1)
            group_pred = group_pred.reshape(-1, 100000)
            assert group_pred.shape[0] == len(test_groups[group_id])
            pred[test_groups[group_id]] = group_pred
        pred = np.concatenate(pred)
        ss = pd.read_csv("data/sample_submission.csv", dtype={'time': str})
        test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str), 'open_channels': pred.astype(np.int)})
        test_pred_frame.to_csv("./gru_preds_{}.csv".format(config.expriment_id), float_format='%.4f', index=False)

    else:
        pred = np.zeros((2000000, 11))

        train_dataloaders, valid_dataloaders, test_dataloaders = get_data_loader(config, None)
        index = 0

        for train_dataloader, valid_dataloader, test_dataloader in zip(train_dataloaders, valid_dataloaders,
                                                                       test_dataloaders):
            early_stopping = EarlyStopping(patience=20, is_maximize=True,
                                           checkpoint_path="./models/gru_clean_checkpoint_fold_{}_exp_{}.pt".format(
                                               index,
                                               config.expriment_id))
            model = getModel(config)
            train_(model, train_dataloader, valid_dataloader, early_stopping, 5, index, config)
            early_stopping.load_best_weights(model)
            oof_score.append(round(early_stopping.best_score, 6))
            pred_list = []
            with torch.no_grad():
                for x, y in tqdm(test_dataloader):
                    x = x.to(config.device)  # .to(device)
                    y = y.to(config.device)  # ..to(device)
                    predictions = model(x)
                    predictions_ = predictions.view(-1, predictions.shape[-1])  # shape [128, 4000, 11]
                    # print(predictions.shape, F.softmax(predictions_, dim=1).cpu().numpy().shape)
                    pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())  # shape (512000, 11)
                    # a = input()
                test_preds = np.vstack(pred_list)  # shape [2000000, 11]
                pred += test_preds
            index += 1

        submission_csv_path = 'data/sample_submission.csv'
        ss = pd.read_csv(submission_csv_path, dtype={'time': str})
        test_preds_all = pred / np.sum(pred, axis=1)[:, None]
        test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                        'open_channels': np.argmax(test_preds_all, axis=1)})
        test_pred_frame.to_csv("./gru_preds_{}.csv".format(config.expriment_id), index=False)

        print('all folder score is:%s' % str(oof_score))
        print('OOF mean score is: %f' % (sum(oof_score) / len(oof_score)))
        res_dict = {"scores": oof_score, "mean_score": (sum(oof_score) / len(oof_score))}
        with open("res_{}.json".format(config.expriment_id)) as f:
            json.dump(res_dict, f)


def test_config(config):
    epoch = config.EPOCHS
    print("========valid expriment {}=========".format(config.expriment_id))
    config.EPOCHS = 1
    train_epoch_group(config)
    config.EPOCHS = epoch
    print("========valid over expriment {}=========".format(config.expriment_id))
