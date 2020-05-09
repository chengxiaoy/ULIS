# from hmm_viterbi import *
import joblib

# df_train = pd.read_csv("data/train.csv")
# df_train = pd.read_csv("train_kalman_2.csv")
# signal = df_train['signal'].values[2000000:2500000]
# true_state = df_train['open_channels'].values[2000000:2500000]
#
# since = time.time()
# model = ViterbiModel()
# model.learning(true_state, signal)
# viterbi_state = model.decoding(signal)
#
# print("Accuracy =", accuracy_score(y_pred=viterbi_state, y_true=true_state))
# print("F1 macro =", f1_score(y_pred=viterbi_state, y_true=true_state, average='macro'))
# print("cost {} s".format(time.time() - since))


# path2 = "gru_preds_112.csv"
# path1 = "gru_preds_80.csv"
#
# pd1 = pd.read_csv(path1)
# pd2 = pd.read_csv(path2)
#
# value1 = pd1['open_channels'].values
# value2 = pd2['open_channels'].values
# value1[5*100000:6*100000] = value2[5*100000:6*100000]
# value1[7*100000:8*100000] = value2[7*100000:8*100000]
#
# pd1['open_channels'] = value1
# pd1.to_csv("mix.csv",float_format='%.4f', index=False)

# preds = np.zeros((200 * 10000, 11))
# for i in [89,95]:
#     preds += joblib.load('pred_{}.pkl'.format(i))
#
# submission_csv_path = 'data/sample_submission.csv'
# ss = pd.read_csv(submission_csv_path, dtype={'time': str})
#
# test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
#                                 'open_channels': np.argmax(preds, axis=1)})
# test_pred_frame.to_csv("./gru_preds_mix.csv", index=False)


from pomegranate import *

import numpy as np
#
# X = []
# Z = []
# for i in range(100):
#     z = np.random.randint(0, 5, size=100)
#     x = []
#     for j in z:
#         x.append(np.random.normal(j, 0.1 + 0.1 * j))
#     X.append(x)
#     Z.append(z)
# X = np.array(X)
# # X = X.reshape(100,1000,-1)
#
#
# model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=X[:20])
# viterbi_state = model.viterbi(X[50])
# print(Z[50])
# print(", ".join(state.name for i, state in viterbi_state[1]))

import helper
import pandas as pd

train_states, train_signals, train_groups, test_signals, test_groups = helper.load_data(kalman_filter=False)
test_y_pred = [None] * np.sum([len(x) for x in test_groups])

for index, (train_group, test_group) in enumerate(zip(train_groups, test_groups)):
    since = time.time()

    print("train_groups :", train_group, ", test_groups :", test_group)

    signal_train = np.concatenate(train_signals[train_group])
    true_state_train = np.concatenate(train_states[train_group])

    signal_train = signal_train.reshape(-1, 4000)
    model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=len(np.unique(true_state_train)),
                                           X=signal_train)

    for test_grp in test_group:
        test_list = test_signals[test_grp].reshape(-1, 4000)
        state_list = []
        for test in test_list:
            state = model.viterbi(test)
            for i, s in state[1][1:]:
                state_list.extend(i)
        print(len(state_list))
        test_y_pred[test_grp] = state_list
    print("cost {} s".format(time.time() - since))

test_y_pred = np.concatenate(test_y_pred)

df_subm = pd.read_csv("data/sample_submission.csv")
df_subm['open_channels'] = test_y_pred
df_subm.to_csv("viterbi_new.csv", float_format='%.4f', index=False)
