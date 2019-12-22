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
        nn.BatchNorm2d(out_features),
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
        nn.Conv2d(in_features, in_features, 1),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_features, mid_features, 3, padding=1),
        nn.ELU(),
        nn.ConvTranspose2d(mid_features, mid_features, 3, padding=1),
        nn.Conv2d(mid_features, out_features, 1),
        nn.BatchNorm2d(out_features),
        nn.ELU()
    )


class CelebaConv(NCSN):
    def __init__(self):
        super().__init__()
        self.input_size = (3, 48, 64)
        self.conv = nn.Sequential(
            conv_pool_layer(3, 64, 64),  # (24, 32)
            conv_pool_layer(64, 128, 128),  # (12, 16)
            conv_pool_layer(128, 128, 128),  # (6, 8)
        )
        self.linear = nn.Sequential(
            linear_layer(6 * 8 * 128 + 40, 256),
            linear_layer(256, 6 * 8 * 128),
        )
        self.deconv = nn.Sequential(
            deconv_layer(256, 192, 128),
            deconv_layer(256, 192, 128),
            deconv_layer(192, 128, 96),
        )
        self.out = nn.Sequential(
            nn.Conv2d(99, 64, 1),
            nn.ELU(),
            nn.Conv2d(64, 16, 1),
            nn.Conv2d(16, 3, 1),
        )

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        conv0 = self.conv[0](inputs)
        conv1 = self.conv[1](conv0)
        conv2 = self.conv[2](conv1)
        batch, nchan, nrow, ncol = conv2.shape
        linear = torch.cat((conv2.view(batch, -1), labels), -1)
        linear = self.linear(linear).view(batch, nchan, nrow, ncol)
        deconv0 = self.deconv[0](torch.cat((conv2, linear), 1))
        deconv1 = self.deconv[1](torch.cat((conv1, deconv0), 1))
        deconv2 = self.deconv[2](torch.cat((conv0, deconv1), 1))
        out = self.out(torch.cat((inputs, deconv2), 1))

        return out
