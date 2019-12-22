from math import log

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset


def one_hot(
    label: int,
    num_labels: int,
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu")
) -> Tensor:
    result = torch.zeros(num_labels, dtype=dtype, device=device)
    result[int(label)] = 1

    return result


def format_dataset(
    dataset: Dataset,
    start_sigma: float = 1.0,
    end_sigma: float = 0.01,
    num_classes: int = 10,
) -> Dataset:
    idx = torch.randperm(len(dataset)).tolist()
    raw_inputs = torch.stack([dataset[i][0] for i in idx], 0)
    labels = torch.stack([one_hot(dataset[i][1], num_classes) for i in idx], 0)

    sigmas = torch.linspace(log(start_sigma), log(end_sigma), len(idx)).exp()
    sigmas = sigmas.view(-1, 1, 1, 1)
    noise = torch.randn_like(raw_inputs) * sigmas
    inputs = raw_inputs + noise
    targets = -noise

    return TensorDataset(inputs, labels, targets)
