import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from pomegranate import NormalDistribution
from sklearn.metrics import f1_score, accuracy_score
import helper
import time
from pykalman import KalmanFilter
import joblib


def viterbi(p_trans, p_signal, p_in, signal, index_id):
    offset = 10 ** (-20)  # added to values to avoid problems with log2(0)

    p_trans_tlog = np.transpose(np.log2(p_trans + offset))  # p_trans, logarithm + transposed
    p_signal_tlog = np.transpose(np.log2(p_signal + offset))  # p_signal, logarithm + transposed
    p_in_log = np.log2(p_in + offset)  # p_in, logarithm

    p_state_log = [p_in_log + p_signal_tlog[signal[0]]]  # initial state probabilities for signal element 0
    p_max_pre = []

    for s in signal[1:]:
        p_state = np.exp2(p_state_log[-1])
        p_state_log[-1] = np.log2(p_state / np.sum(p_state))

        p_max_pre.append(np.argmax(p_state_log[-1] + p_trans_tlog, axis=1))
        p_state_log.append(np.max(p_state_log[-1] + p_trans_tlog, axis=1) + p_signal_tlog[s])  # the Viterbi algorithm

    p_state = np.exp2(p_state_log[-1])
    p_state_log[-1] = np.log2(p_state / np.sum(p_state))
    joblib.dump(p_state_log, "viterbi_pred_test_{}.pkl".format(index_id))

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
        i = i + 1

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


def calc_markov_p_signal(state, signal, num_bins=3000):
    states_range = np.arange(state.min(), state.max() + 1)
    signal_bins = np.linspace(signal.min(), signal.max(), num_bins + 1)
    p_signal = np.array([np.histogram(signal[state == s], bins=signal_bins)[0] for s in states_range])
    p_signal = np.array([p / np.sum(p) for p in p_signal])  # normalize to 1
    return p_signal, signal_bins


# def calc_markov_p_signal_2(states, signals):
#     emiting_pdf = {}
#     for state in np.unique(states):
#         emiting_pdf[state] = NormalDistribution.from_samples(signals[states == state])
#
#     return emiting_pdf


# Function from https://www.kaggle.com/aussie84/train-fare-trends-using-kalman-filter-1d
def Kalman1D(observations, damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state


def digitize_signal(signal, signal_bins):
    signal_dig = np.digitize(signal, bins=signal_bins) - 1  # these -1 and -2 are necessary because of the way...
    signal_dig = np.minimum(signal_dig, len(signal_bins) - 2)  # ... numpy.digitize works
    return signal_dig


class ViterbiModel():
    def __init__(self, kalman_filter=False):
        self.p_trans = None
        self.p_emit = None
        self.p_in = None
        self.signal_bins = None
        self.kalman_filter = kalman_filter

    def learning(self, states, signals):
        if self.kalman_filter:
            observation_covariance = .0015
            signals = Kalman1D(signals, observation_covariance).reshape(-1)

        self.p_trans = calc_markov_p_trans(states)
        # self.p_trans = self.p_trans[1:, 1:]
        # self.p_emit = calc_markov_p_signal_2(states, signals)
        self.p_emit, self.signal_bins = calc_markov_p_signal(states, signals)
        self.p_in = np.ones(len(self.p_trans)) / len(self.p_trans)
        return self

    def decoding(self, signals, index_id):
        if self.kalman_filter:
            observation_covariance = .0015
            signals = Kalman1D(signals, observation_covariance).reshape(-1)
        signal_dig = digitize_signal(signals, self.signal_bins)
        # return viterbi_2(self.p_trans, self.p_emit, self.p_in, signals)
        return viterbi(self.p_trans, self.p_emit, self.p_in, signal_dig, index_id)


def test(train_group):
    train_states, train_signals, train_groups, test_signals, test_groups = helper.load_data(kalman_filter=False)

    signal_train = np.concatenate(train_signals[train_group])
    true_state_train = np.concatenate(train_states[train_group])
    model = ViterbiModel(kalman_filter=False)
    model.learning(true_state_train, signal_train)
    viterbi_state = model.decoding(signal_train, 0)
    print(f1_score(true_state_train, viterbi_state, average='macro'))


def get_train_probe():
    train_states, train_signals, train_groups, test_signals, test_groups = helper.load_data(kalman_filter=False)

    for i in range(1):
        i = 9
        train_state = train_states[i]
        train_signal = train_signals[i]
        model = ViterbiModel(kalman_filter=False)
        model.learning(train_state, train_signal)
        model.decoding(train_signal, i)


def concate():
    train_pro_dict = {}
    for i in range(10):
        train_pro_dict[i] = joblib.load('viterbi_pred_train_{}.pkl'.format(i))

    train_prob = np.zeros((500 * 10000, 11))
    row_index = 0
    for i in range(9):
        prob = np.array(train_pro_dict[i])
        row, col = prob.shape
        train_prob[row_index:row_index + row, 0:col] = np.exp2(prob)
        row_index += row
    train_prob[row_index:, 1:] = np.exp2(np.array(train_pro_dict[9]))
    joblib.dump(train_prob, "train_prob.pkl")


def concate_test():
    train_pro_dict = {}
    for i in range(11):
        train_pro_dict[i] = joblib.load('viterbi_pred_test_{}.pkl'.format(i))

    test_prob = np.zeros((200 * 10000, 11))
    row_index = 0
    for i in range(11):
        prob = np.array(train_pro_dict[i])
        row, col = prob.shape
        test_prob[row_index:row_index + row, 0:col] = np.exp2(prob)
        row_index += row
    joblib.dump(test_prob, "test_prob.pkl")


if __name__ == '__main__':
    concate()
    concate_test()
    # get_train_probe()

    train_states, train_signals, train_groups, test_signals, test_groups = helper.load_data(kalman_filter=False)
    # test_y_pred = [None] * np.sum([len(x) for x in test_groups])
    #
    # for index, (train_group, test_group) in enumerate(zip(train_groups, test_groups)):
    #     since = time.time()
    #
    #     print("train_groups :", train_group, ", test_groups :", test_group)
    #
    #     signal_train = np.concatenate(train_signals[train_group])
    #     true_state_train = np.concatenate(train_states[train_group])
    #     model = ViterbiModel(kalman_filter=False)
    #     model.learning(true_state_train, signal_train)
    #
    #     for test_grp in test_group:
    #         test_y_pred[test_grp] = model.decoding(test_signals[test_grp], test_grp)
    #     print("cost {} s".format(time.time() - since))
    #
    # test_y_pred = np.concatenate(test_y_pred)
    #
    # df_subm = pd.read_csv("data/sample_submission.csv")
    # df_subm['open_channels'] = test_y_pred
    # df_subm.to_csv("viterbi_new.csv", float_format='%.4f', index=False)
