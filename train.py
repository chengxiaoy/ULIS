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

writer = SummaryWriter(logdir=os.path.join("board/", str(1)))

if not os.path.exists("./models"):
    os.makedirs("./models")
for index, (train_index, val_index) in enumerate(new_splits[0:], start=0):
    print("Fold : {}".format(index))

    batchsize = 16
    train_dataset = IonDataset(trainval[train_index], trainval_y[train_index], flip=False, noise_level=0.0,
                               class_split=0.0)
    train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=8, pin_memory=True)

    valid_dataset = IonDataset(trainval[val_index], trainval_y[val_index], flip=False)
    valid_dataloader = DataLoader(valid_dataset, batchsize, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = IonDataset(test, test_y, flip=False, noise_level=0.0, class_split=0.0)
    test_dataloader = DataLoader(test_dataset, batchsize, shuffle=False, num_workers=8, pin_memory=True)
    test_preds_iter = np.zeros((2000000, 11))
    it = 0
    for it in range(1):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = Seq2SeqRnn(input_size=trainval.shape[1], seq_len=4000, hidden_size=64, output_size=11, num_layers=2,
                           hidden_layers=[64, 64, 64],
                           bidirectional=True).to(device)

        no_of_epochs = 150
        early_stopping = EarlyStopping(patience=20, is_maximize=True,
                                       checkpoint_path="./models/gru_clean_checkpoint_fold_{}_iter_{}.pt".format(index,
                                                                                                                 it))
        criterion = L.FocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(train_dataloader))
        avg_train_losses, avg_valid_losses = [], []

        for epoch in range(no_of_epochs):
            start_time = time.time()

            print("Epoch : {}".format(epoch))
            print("learning_rate: {:0.9f}".format(schedular.get_lr()[0]))
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
                schedular.step()
                # record training lossa
                train_losses.append(loss.item())

                train_true = torch.cat([train_true, y_], 0)
                train_preds = torch.cat([train_preds, predictions_], 0)

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
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print("train_loss: {:0.6f}, valid_loss: {:0.6f}".format(train_loss, valid_loss))

            train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1),
                                   labels=list(range(11)), average='macro')

            val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu().detach().numpy().argmax(1),
                                 labels=list(range(11)), average='macro')
            print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))

            writer.add_scalars('cv_{}/loss'.format(index), {'train': train_loss, 'val': valid_loss}, epoch)
            writer.add_scalars('cv_{}/f1_score'.format(index), {'train': train_score, 'val': val_score}, epoch)
            if early_stopping(val_score, model):
                print("Early Stopping...")
                print("Best Val Score: {:0.6f}".format(early_stopping.best_score))
                break

            print("--- %s seconds ---" % (time.time() - start_time))

        model.load_state_dict(torch.load("./models/gru_clean_checkpoint_fold_{}_iter_{}.pt".format(index, it)))
        with torch.no_grad():
            pred_list = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x[:, :trainval.shape[1], :])
                predictions_ = predictions.view(-1, predictions.shape[-1])

                pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())
            test_preds = np.vstack(pred_list)

        test_preds_iter += test_preds
        test_preds_all += test_preds
        if not os.path.exists("./predictions/test"):
            os.makedirs("./predictions/test")
        np.save('./predictions/test/gru_clean_fold_{}_iter_{}_raw.npy'.format(index, it), arr=test_preds_iter)
        np.save('./predictions/test/gru_clean_fold_{}_raw.npy'.format(index), arr=test_preds_all)

ss = pd.read_csv("/local/ULIS/sample_submission.csv", dtype={'time': str})

test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("./gru_preds.csv", index=False)
