from torch import nn
import torch
import torch.nn.functional as F


class Seq2SeqRnn(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size, num_layers=1, bidirectional=False, dropout=.3,
                 hidden_layers=[100, 200]):

        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)
        # Input Layer
        if hidden_layers and len(hidden_layers):
            first_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, hidden_layers[0])

            # Hidden Layers
            self.hidden_layers = nn.ModuleList(
                [first_layer] + [nn.Linear(hidden_layers[i], hidden_layers[i + 1]) for i in
                                 range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers: nn.init.kaiming_normal_(layer.weight.data)

            self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            # output layers
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, self.input_size)
            self.output_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.activation_fn = torch.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)

        outputs, hidden = self.rnn(x)

        x = self.dropout(self.activation_fn(outputs))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)

        x = self.output_layer(x)

        return x


# from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = F.tanh(self.filter_convs[i](x)) * F.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            # x += res
            res = torch.add(res, x)
        return res


class WaveNet(nn.Module):
    def __init__(self, intput_n):
        super().__init__()
        input_size = 128
        self.LSTM1 = nn.GRU(input_size=intput_n, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        # self.attention = Attention(input_size,4000)
        # self.rnn = nn.RNN(input_size, 64, 2, batch_first=True, nonlinearity='relu')

        self.wave_block1 = Wave_Block(128, 16, 12)
        self.wave_block2 = Wave_Block(16, 32, 8)
        self.wave_block3 = Wave_Block(32, 64, 4)
        self.wave_block4 = Wave_Block(64, 128, 1)
        self.fc = nn.Linear(128, 11)

    def forward(self, x):
        x, _ = self.LSTM1(x)
        x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        # x,_ = self.LSTM(x)
        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x, _ = self.LSTM(x)
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.rnn(x)
        # x = self.attention(x)
        x = self.fc(x)
        return x


def getModel(config):
    model = None
    if config.data_fe == 'shifted_proba':
        input_size = 19
    else:
        input_size = 1
    if config.model_name == 'wave_net':
        model = WaveNet(input_size)


    elif config.model_name == 'seq2seq':
        model = Seq2SeqRnn(input_size=input_size, seq_len=4000, hidden_size=64, output_size=11, num_layers=2,
                           hidden_layers=[64, 64, 64],
                           bidirectional=True)
    model.to(config.device)
    return model
