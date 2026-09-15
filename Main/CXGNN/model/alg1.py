"""NNModel: shared encoder for causal survival (used by sPixel_CXGNN pipeline)."""
import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, input_size, output_size, h_size, h_layers):
        super(NNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h_size = h_size
        self.h_layers = h_layers
        layers = [nn.Linear(self.input_size, self.h_size), nn.ReLU()]
        for _ in range(h_layers - 1):
            layers += [nn.Linear(self.h_size, self.h_size), nn.ReLU()]
        layers.append(nn.Linear(self.h_size, self.output_size))
        self.nn = nn.Sequential(*layers)
        self.nn.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, u):
        return torch.sigmoid(self.nn(u))
