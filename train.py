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

EPOCHS = 90  # 150
NNBATCHSIZE = 32
GROUP_BATCH_SIZE = 4000
SEED = 123
LR = 0.001
SPLITS = 5
model_name = 'wave_net'
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
data_type = "kalman_clean"  # raw clean kalman_clean
data_fe = "shifted_proba"  # none "shifted"
data_group = False
outdir = 'wavenet_models'
flip = False
noise = False
expriment_id = 5
config = AttrDict({'EPOCHS': EPOCHS, 'NNBATCHSIZE': NNBATCHSIZE, 'GROUP_BATCH_SIZE': GROUP_BATCH_SIZE, 'SEED': SEED,
                   'LR': LR, 'SPLITS': SPLITS, 'model_name': model_name, 'device': device, 'outdir': outdir,
                   'expriment_id': expriment_id, 'data_type': data_type, 'data_fe': data_fe, 'noise': noise,
                   'flip': flip})


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(config.seed)

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

# build model


# build dataloader
test_y = np.zeros([int(2000000 / config.GROUP_BATCH_SIZE), config.GROUP_BATCH_SIZE, 1])
test_dataset = IronDataset(test, test_y, flip=False)
test_dataloader = DataLoader(test_dataset, config.NNBATCHSIZE, shuffle=False)
test_preds_all = np.zeros((2000000, 11))

writer = SummaryWriter(logdir=os.path.join("board/", str(config.expriment_id)))

oof_score = []
for index, (train_index, val_index, _) in enumerate(new_splits[0:], start=0):

    print("Fold : {}".format(index))
    train_dataset = IronDataset(train[train_index], train_tr[train_index], seq_len=config.GROUP_BATCH_SIZE, flip=flip,
                                noise_level=noise)
    train_dataloader = DataLoader(train_dataset, config.NNBATCHSIZE, shuffle=True, num_workers=16)

    valid_dataset = IronDataset(train[val_index], train_tr[val_index], seq_len=config.GROUP_BATCH_SIZE, flip=False)
    valid_dataloader = DataLoader(valid_dataset, config.NNBATCHSIZE, shuffle=False)

    it = 0
    model = getModel(config)

    early_stopping = EarlyStopping(patience=10, is_maximize=True,
                                   checkpoint_path=os.path.join(config.outdir,
                                                                "gru_clean_checkpoint_expid_{}_fold_{}_iter_{}.pt".format(
                                                                    config.expriment_id, index,
                                                                    it)))

    weight = None  # cal_weights()
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    # optimizer = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=2, swa_lr=0.0011)
    optimizer = torchcontrib.optim.SWA(optimizer)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.2)

    avg_train_losses, avg_valid_losses = [], []

    for epoch in range(config.EPOCHS):

        train_losses, valid_losses = [], []
        tr_loss_cls_item, val_loss_cls_item = [], []

        model.train()  # prep model for training
        train_preds, train_true = torch.Tensor([]).to(config.device), torch.LongTensor([]).to(
            config.device)  # .to(device)

        print('**********************************')
        print("Folder : {} Epoch : {}".format(index, epoch))
        print("Curr learning_rate: {:0.9f}".format(optimizer.param_groups[0]['lr']))

        # loss_fn(model(input), target).backward()
        for x, y in tqdm(train_dataloader):
            x = x.to(config.device)
            y = y.to(config.device)
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

            # record training lossa
            train_losses.append(loss.item())
            train_true = torch.cat([train_true, y_], 0)
            train_preds = torch.cat([train_preds, predictions_], 0)

        # schedular.step(loss)
        # 更行swa
        if epoch >= 30 and epoch % 5 == 0:
            optimizer.update_swa()
        # 切换成swa 进行valid 和 save
        optimizer.swap_swa_sgd()
        model.eval()  # prep model for evaluation

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

        writer.add_scalars('cv_{}/loss'.format(index), {'train': train_loss, 'val': valid_loss}, epoch)
        writer.add_scalars('cv_{}/f1_score'.format(index), {'train': train_score, 'val': val_score}, epoch)

        res = early_stopping(val_score, model)

        # 再 切换回来
        optimizer.swap_swa_sgd()
        # print('fres:', res)
        if res == 2:
            print("Early Stopping")
            print('folder %d global best val max f1 model score %f' % (index, early_stopping.best_score))
            break
        elif res == 1:
            print('save folder %d global val max f1 model score %f' % (index, val_score))
    print('Folder {} finally best global max f1 score is {}'.format(index, early_stopping.best_score))
    oof_score.append(round(early_stopping.best_score, 6))

    early_stopping.load_best_weights(model)

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
writer.add_text("score_{}".format(config.expriment_id), 'all folder score is:%s' % str(oof_score))
writer.add_text("mean_score_{}".format(config.expriment_id),
                'OOF mean score is: %f' % (sum(oof_score) / len(oof_score)))
print('all folder score is:%s' % str(oof_score))
print('OOF mean score is: %f' % (sum(oof_score) / len(oof_score)))
print('Generate submission.............')
submission_csv_path = 'data/sample_submission.csv'
ss = pd.read_csv(submission_csv_path, dtype={'time': str})
test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("./gru_preds_{}.csv".format(config.expriment_id), index=False)
print('over')
