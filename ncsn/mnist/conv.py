
import torch
from torch import Tensor
from torch import nn

from ncsn import NCSN


def conv_pool_layer(
    in_features: int,
    mid_features: int,
    out_features: int
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_features, mid_features, 3, padding=1),
        nn.ELU(),
        nn.Conv2d(mid_features, mid_features, 3, padding=1),
        nn.Conv2d(mid_features, out_features, 1),
        nn.ELU(),
        nn.AvgPool2d(2),
    )


def linear_layer(in_size: int, out_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.LayerNorm(out_size),
        nn.ELU(),
    )


def deconv_layer(
    in_features: int,
    mid_features: int,
    out_features: int
) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.ConvTranspose2d(in_features, mid_features, 3, padding=1),
        nn.ELU(),
        nn.ConvTranspose2d(mid_features, mid_features, 3, padding=1),
        nn.ConvTranspose2d(mid_features, out_features, 1),
        nn.ELU()
    )


class MnistConv(NCSN):
    def __init__(self):
        super().__init__()
        self.input_size = (1, 16, 16)
        self.conv = nn.ModuleList([
            conv_pool_layer(1, 16, 16),
            conv_pool_layer(16, 16, 16),
        ])
        self.linear = nn.Sequential(
            linear_layer(4 * 4 * 16 + 10, 32),
            linear_layer(32, 4 * 4 * 16),
        )
        self.deconv = nn.ModuleList([
            deconv_layer(32, 32, 16),
            deconv_layer(32, 32, 16),
        ])
        self.out = nn.Sequential(
            nn.Conv2d(16, 16, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 8, 1),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        conv0 = self.conv[0](inputs)
        conv1 = self.conv[1](conv0)
        batch, nchan, nrow, ncol = conv1.shape
        linear = torch.cat((conv1.view(batch, -1), labels), -1)
        linear = self.linear(linear).view(batch, nchan, nrow, ncol)
        deconv0 = self.deconv[0](torch.cat((conv1, linear), 1))
        deconv1 = self.deconv[1](torch.cat((conv0, deconv0), 1))
        out = self.out(deconv1)

        return out
