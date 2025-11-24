import torch.nn as nn
import torch.nn.functional as F

input_dim = 5
output_dim = 2
hidden_dims = [512, 256, 256, 128, 64]
dropout = 0.2


class OpacityNet(nn.Module):
    def __init__(self, predict_log=True):
        """
        Neural network for opacity prediction

        Args:
            predict_log: If True, predicts log10(opacity). If False, predicts raw opacity with positive constraint.
        """
        super().__init__()
        self.predict_log = predict_log

        layers = []
        dims = [input_dim] + hidden_dims

        # Hidden layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch, 5] = [mH, mHe, mAl, log10(Temperature), log10(Density)]

        Returns:
            If predict_log=True: [batch, 2] = [log10(κR), log10(κP)]
            If predict_log=False: [batch, 2] = [κR, κP] (positive constrained)
        """
        output = self.net(x)

        if self.predict_log:
            # Predicting log-opacities, no constraint needed
            # Positive constraint applied when converting: 10^output
            return output
        else:
            # Predicting raw opacities, enforce positivity with softplus
            return F.softplus(output)
