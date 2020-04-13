from torch import nn
import torch

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
                          bidirectional=bidirectional, batch_first=True, dropout=0.3)
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
            self.intermediate_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_siz, self.input_size)
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