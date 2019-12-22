import torch
from torch import nn


def get_module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device