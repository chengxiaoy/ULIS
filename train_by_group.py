import os
from data import IonDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import Seq2SeqRnn
import torch
from pytorch_toolbelt import losses as L
from helper import EarlyStopping
import time
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import f1_score
from data import new_splits, trainval, trainval_y, test, test_y, test_preds_all
from tensorboardX import SummaryWriter
import numpy as np

expriment_id = 5
writer = SummaryWriter(logdir=os.path.join("board/", str(expriment_id)))


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, schedular, early_stopping, epoch_n):
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
            predictions = model(x[:, :trainval.shape[1], :])

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

                predictions = model(x[:, :trainval.shape[1], :])
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
        print("train shape {}".format(val_true.shape))

        train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1),
                               labels=list(range(11)), average='macro')

        val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu().detach().numpy().argmax(1),
                             labels=list(range(11)), average='macro')
        print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))

        writer.add_scalars('cv_{}/loss'.format(index), {'train': train_loss, 'val': valid_loss}, epoch)
        writer.add_scalars('cv_{}/f1_score'.format(index), {'train': train_score, 'val': val_score}, epoch)
        if early_stopping(valid_loss, model):
            print("Early Stopping...")
            print("Best Val Score: {:0.6f}".format(early_stopping.best_score))
            break

        print("--- %s seconds ---" % (time.time() - start_time))


if not os.path.exists("./models"):
    os.makedirs("./models")

train_groups = [[0, 1], [2, 6], [3, 7], [5, 8], [4, 9]]
test_groups = [[0, 3, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [4], [1, 9], [2, 6], [5, 7]]

train_data_n = len(trainval)
test_data_n = len(test)

train_indexs = list(range(train_data_n))
test_indexs = list(range(test_data_n))

group_id = 1

pred = np.zeros([20, 100000])
for train_group, test_group in zip(train_groups, test_groups):
    train_index, val_index = new_splits[0]
    train_group_indexs = []
    test_group_indexs = []

    for grp in train_group:
        train_group_indexs.append(train_indexs[grp * (train_data_n // 10):(grp + 1) * (train_data_n // 10)])

    for grp in test_group:
        test_group_indexs.append(test_indexs[grp * (test_data_n // 20):(grp + 1) * (test_data_n // 20)])

    train_group_indexs = np.concatenate(train_group_indexs)
    test_group_indexs = np.concatenate(test_group_indexs)

    test = test[test_group_indexs]
    test_dataset = IonDataset(test, test_y, flip=False, noise_level=0.0, class_split=0.0)
    test_dataloader = DataLoader(test_dataset, 16, shuffle=False, num_workers=8, pin_memory=True)

    test_preds_all = np.zeros([len(test_group) * 100000, 11])

    for index in range(1):
        train_index, val_index = new_splits[index]
        train_index = np.intersect1d(train_index, train_group_indexs)
        val_index = np.intersect1d(val_index, train_group_indexs)

        batchsize = 16
        train_dataset = IonDataset(trainval[train_index], trainval_y[train_index], flip=False, noise_level=0.0,
                                   class_split=0.0)
        train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=8, pin_memory=True)

        valid_dataset = IonDataset(trainval[val_index], trainval_y[val_index], flip=False)
        valid_dataloader = DataLoader(valid_dataset, batchsize, shuffle=False, num_workers=4, pin_memory=True)

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = Seq2SeqRnn(input_size=trainval.shape[1], seq_len=4000, hidden_size=64, output_size=11, num_layers=2,
                           hidden_layers=[64, 64, 64],
                           bidirectional=True).to(device)

        no_of_epochs = 150
        early_stopping = EarlyStopping(patience=10, is_maximize=False,
                                       checkpoint_path="./models/gru_clean_checkpoint_fold_{}_group_{}_exp_{}.pt".format(
                                           index,
                                           group_id, expriment_id))
        criterion = L.FocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10)
        train(model, train_dataloader, valid_dataloader, criterion, optimizer, schedular, early_stopping, 1)

        model.load_state_dict(
            torch.load(
                "./models/gru_clean_checkpoint_fold_{}_group_{}_exp_{}.pt".format(index, group_id, expriment_id)))
        with torch.no_grad():
            pred_list = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x[:, :trainval.shape[1], :])
                predictions_ = predictions.view(-1, predictions.shape[-1])

                pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())
            test_preds = np.vstack(pred_list)
        test_preds_all += test_preds

    group_pred = np.argmax(test_preds_all, axis=1)

    group_pred.reshape(-1, 100000)
    print(group_pred.shape)

    assert group_pred.shape[0] == len(test_group)
    test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]

    pred[test_group] = test_preds_all
ss = pd.read_csv("/local/ULIS/data/sample_submission.csv", dtype={'time': str})

test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("./gru_preds_{}.csv".format(expriment_id), index=False)
