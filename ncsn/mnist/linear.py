import torch
from torch import nn as nn, Tensor

from ncsn import NCSN


class MnistLinear(NCSN):
    def __init__(self):
        super().__init__()
        self.input_size = (1, 16, 16)
        self.layers = nn.Sequential(
            nn.Linear(16 * 16 + 10, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 16 * 16),
        )

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        batch = inputs.shape[0]
        flattened = torch.cat([inputs.view(batch, -1), labels], -1)
        return self.layers(flattened).view(-1, *self.input_size)