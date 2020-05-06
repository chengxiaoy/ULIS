from hmm_viterbi import *
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

preds = np.zeros((200 * 10000, 11))
for i in range(89, 95):
    preds += joblib.load('pred_{}.pkl'.format(i))

submission_csv_path = 'data/sample_submission.csv'
ss = pd.read_csv(submission_csv_path, dtype={'time': str})

test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(preds, axis=1)})
test_pred_frame.to_csv("./gru_preds_mix.csv", index=False)
