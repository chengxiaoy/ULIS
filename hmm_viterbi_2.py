import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pomegranate import NormalDistribution
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import normalize
import time



def get_p_o(emiting_pdf, signal):
    offset = 10 ** (-20)  # added to values to avoid problems with log2(0)

    p = np.array([emiting_pdf[x].probability(signal) + offset for x in sorted(emiting_pdf.keys())])
    # p = p/np.sum(p)
    return np.log2(p)


def viterbi_2(p_trans, emiting_pdf, p_in, signal):
    offset = 10 ** (-20)  # added to values to avoid problems with log2(0)

    p_trans_tlog = np.transpose(np.log2(p_trans + offset))  # p_trans, logarithm + transposed
    p_in_log = np.log2(p_in + offset)  # p_in, logarithm

    p_signal_tlog = [None] * len(emiting_pdf.keys())
    for state in sorted(emiting_pdf.keys()):
        p_signal_tlog[state] = np.log2(emiting_pdf[state].probability(signal) + offset)

    p_signal_tlog = np.transpose(np.array(p_signal_tlog))
    # p_signal_tlog = normalize(p_signal_tlog)

    p_state_log = [p_in_log + p_signal_tlog[0]]  # initial state probabilities for signal element 0
    p_max_pre = []
    i = 1
    for s in signal[1:]:
        p_max_pre.append(np.argmax(p_state_log[-1] + p_trans_tlog, axis=1))

        p_state_log.append(
            np.max(p_state_log[-1] + p_trans_tlog, axis=1) + p_signal_tlog[i])  # the Viterbi algorithm
        i=i+1

    max_states = np.argmax(p_state_log, axis=1)  # finding the most probable states
    # return max_states
    last_state = max_states[-1]
    # last_state = 6

    states = []
    states.insert(0, last_state)
    for i in range(len(max_states) - 1):
        states.insert(0, p_max_pre[-1 - i][states[0]])

    print(len(states))
    return states

#
# df_train = pd.read_csv("data/train.csv")
# signal = df_train['signal'].values[2000000:2500000]
# true_state = df_train['open_channels'].values[2000000:2500000]


def calc_markov_p_trans(states):
    max_state = np.max(states)
    states_next = np.roll(states, -1)
    matrix = []
    for i in range(max_state + 1):
        current_row = np.histogram(states_next[states == i], bins=np.arange(max_state + 2))[0]
        if np.sum(current_row) == 0:  # if a state doesn't appear in states...
            current_row = np.ones(max_state + 1) / (max_state + 1)  # ...use uniform probability
        else:
            current_row = current_row / np.sum(current_row)  # normalize to 1
        matrix.append(current_row)
    return np.array(matrix)


p_trans = calc_markov_p_trans(true_state)


def calc_markov_p_signal(state, signal, num_bins=3000):
    states_range = np.arange(state.min(), state.max() + 1)
    signal_bins = np.linspace(signal.min(), signal.max(), num_bins + 1)
    p_signal = np.array([np.histogram(signal[state == s], bins=signal_bins)[0] for s in states_range])
    p_signal = np.array([p / np.sum(p) for p in p_signal])  # normalize to 1
    return p_signal, signal_bins


def calc_markov_p_signal_2(states, signals):
    emiting_pdf = {}
    for state in sorted(np.unique(states)):
        emiting_pdf[state] = NormalDistribution.from_samples(signals[states == state])

    return emiting_pdf


# emiting_pdf = calc_markov_p_signal_2(true_state, signal)


# p_signal, signal_bins = calc_markov_p_signal(true_state, signal)


def digitize_signal(signal, signal_bins):
    signal_dig = np.digitize(signal, bins=signal_bins) - 1  # these -1 and -2 are necessary because of the way...
    signal_dig = np.minimum(signal_dig, len(signal_bins) - 2)  # ... numpy.digitize works
    return signal_dig


# signal_dig = digitize_signal(signal, signal_bins)
# print("Signal bin values:", signal_dig)
# since = time.time()
# p_in = np.ones(len(p_trans)) / len(p_trans)
# print("Initial probability p_in =", p_in)
# viterbi_state = viterbi_2(p_trans, emiting_pdf, p_in, signal)
# print("State sequence as predicted by Viterbi algorithm :", viterbi_state)
#
# print("Accuracy =", accuracy_score(y_pred=viterbi_state, y_true=true_state))
# print("F1 macro =", f1_score(y_pred=viterbi_state, y_true=true_state, average='macro'))
# print("cost {} s".format(time.time()-since))

with np.load('data/train_detrend.npz', allow_pickle=True) as data:
    train_signal = data['train_signal']
    train_opchan = data['train_opench']
    train_groups = data['train_groups']

with np.load('data/test_detrend.npz', allow_pickle=True) as data:
    test_signal = data['test_signal']
    test_groups = data['test_groups']

train_group_ids = [[0, 1, 2], [3, 7], [4, 8], [6, 9], [5, 10]]
test_group_ids = [[0, 3, 8, 10, 11], [4], [1, 9], [2, 6], [5, 7]]
test_y_pred = [None] * 12

for train_groups, test_groups in zip(train_group_ids, test_group_ids):
    since = time.time()

    print("train_groups :", train_groups, ", test_groups :", test_groups)

    signal_train = np.concatenate(train_signal[train_groups])
    true_state_train = np.concatenate(train_opchan[train_groups])

    p_trans = calc_markov_p_trans(true_state_train)
    p_signal = calc_markov_p_signal_2(true_state_train, signal_train)
    p_in = np.ones(len(p_trans)) / len(p_trans)


    for test_grp in test_groups:
        test_y_pred[test_grp] = viterbi_2(p_trans, p_signal, p_in, test_signal[test_grp])
    print("cost {} s".format(time.time()-since))

test_y_pred = np.concatenate(test_y_pred)

df_subm = pd.read_csv("data/sample_submission.csv")
df_subm['open_channels'] = test_y_pred
df_subm.to_csv("viterbi_new_1.csv", float_format='%.4f', index=False)
