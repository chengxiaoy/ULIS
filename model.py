from torch import nn
import torch
import torch.nn.functional as F
from unet import Unet


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
        # x = x.permute(0, 2, 1)

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


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dilation):
        super(CBR, self).__init__()
        self.cov = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, dilation=dilation,
                             padding=int((kernel - 1) / 2))
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cov(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


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
    def __init__(self, intput_n, config):
        super().__init__()
        input_size = 128
        self.use_cbr = config.use_cbr
        self.dropout = nn.Dropout(config.drop_out)
        self.use_se = config.use_se
        self.residual = config.residual
        if config.use_se:
            self.se1 = SELayer(128)
            self.se2 = SELayer(128)

        if config.use_cbr:
            self.cbr1 = CBR(intput_n, 128, 7, 1, 1)
            self.cbr2 = CBR(128, 32, 7, 1, 1)
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(128)
        else:
            self.LSTM1 = nn.GRU(input_size=intput_n, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
            self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True,
                               bidirectional=True)
        # self.attention = Attention(input_size,4000)
        # self.rnn = nn.RNN(input_size, 64, 2, batch_first=True, nonlinearity='relu')

        self.wave_block1 = Wave_Block(128, 128, 12)
        self.wave_block2 = Wave_Block(128, 128, 8)
        self.wave_block3 = Wave_Block(128, 128, 4)
        self.wave_block4 = Wave_Block(128, 128, 1)
        if config.use_cbr:
            self.fc = nn.Linear(32, 11)
        else:
            self.fc = nn.Linear(128, 11)

    def forward(self, x):
        if not self.use_cbr:
            x, _ = self.LSTM1(x)
        x = x.permute(0, 2, 1)
        if self.use_cbr:
            x = self.cbr1(x)
        if self.use_se:
            x = self.se1(x)

        x1 = self.wave_block1(x)
        if self.use_cbr:
            x1 = self.bn1(x1)
        if self.residual:
            x1 += x

        x2 = self.wave_block2(x1)
        if self.use_cbr:
            x2 = self.bn2(x2)
        if self.residual:
            x2 += x1

        x3 = self.wave_block3(x2)
        if self.use_cbr:
            x3 = self.bn3(x3)
        if self.residual:
            x3 += x2

        # x,_ = self.LSTM(x)
        x4 = self.wave_block4(x3)
        if self.residual:
            x4 += x3
        if self.use_se:
            x4 = self.se2(x4)
        if self.use_cbr:
            x4 = self.cbr2(x4)


        x4 = x4.permute(0, 2, 1)

        if not self.use_cbr:
            x4, _ = self.LSTM(x4)
        x4 = self.dropout(x4)
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.rnn(x)
        # x = self.attention(x)
        x4 = self.fc(x4)
        return x4


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def getModel(config):
    model = None
    if config.data_fe == 'shifted_proba' or config.data_fe == 'shifted_viterbi_proba' or config.data_fe == 'shifted_empty_proba':
        input_size = 19
    elif config.data_fe == 'shifted_mix_proba':
        input_size = 30
    elif config.data_fe == 'shifted':
        input_size = 8
    else:
        input_size = 1
    if config.model_name == 'wave_net':
        model = WaveNet(input_size, config)

    elif config.model_name == 'unet':
        model = Unet(input_size, 11)
    elif config.model_name == 'seq2seq':
        model = Seq2SeqRnn(input_size=input_size, seq_len=config.GROUP_BATCH_SIZE, hidden_size=64, output_size=11,
                           num_layers=2,
                           hidden_layers=[64, 64, 64],
                           bidirectional=True)
    model.to(config.device)
    return model
