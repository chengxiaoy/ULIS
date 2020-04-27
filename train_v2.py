import pandas as pd

gru_0 = pd.read_csv('/Users/tezign/Downloads/gru_preds_0.csv')
gru_2 = pd.read_csv('/Users/tezign/Downloads/gru_preds_2.csv')

gru_0_values = gru_0['open_channels'].values
gru_2_values = gru_2['open_channels'].values

gru_0_values[5*100000:6*100000] =  gru_2_values[5*100000:6*100000]
gru_0_values[7*100000:8*100000] =  gru_2_values[7*100000:8*100000]

gru_0['open_channels'] = gru_0_values
gru_0.to_csv("./gru_preds_{}.csv".format("0_2"), float_format='%.4f', index=False)
