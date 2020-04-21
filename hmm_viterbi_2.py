from hmm_viterbi import *

# df_train = pd.read_csv("data/train.csv")
df_train = pd.read_csv("train_kalman_2.csv")
signal = df_train['signal'].values[2000000:2500000]
true_state = df_train['open_channels'].values[2000000:2500000]

since = time.time()
model = ViterbiModel()
model.learning(true_state, signal)
viterbi_state = model.decoding(signal)

print("Accuracy =", accuracy_score(y_pred=viterbi_state, y_true=true_state))
print("F1 macro =", f1_score(y_pred=viterbi_state, y_true=true_state, average='macro'))
print("cost {} s".format(time.time() - since))
