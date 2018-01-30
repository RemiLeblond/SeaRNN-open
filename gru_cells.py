import torch

from torch.nn.modules.rnn import GRUCell
import torch.nn as nn

"""
    StackedGRU implementation.
"""

class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        assert(input.dim() == 3)
        assert(input.size(0) == 1)
        input = input.squeeze(0)

        states = []
        for i, layer in enumerate(self.layers):
            input = layer(input, hidden[i])
            states += [input]
            if i + 1 != self.num_layers:
                input = self.dropout(input)
        states = torch.stack(states, 0)
        input = input.unsqueeze(0)
        return input, states
