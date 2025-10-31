import torch.nn as nn

input_dim = 5
output_dim = 2
hidden_dims = [256, 256, 128, 64]
dropout = 0.3


class OpacityNet(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        # Input batch norm
        layers.append(nn.BatchNorm1d(input_dim))

        # Hidden layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
