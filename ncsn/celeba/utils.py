from os import path
from math import log

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import pandas as pd


def _format_celeba_dataset(
    images: Tensor,
    labels: Tensor,
    start_sigma: float = 1.0,
    end_sigma: float = 0.01,
) -> Dataset:
    idx = torch.randperm(images.shape[0])
    images = images[idx]
    labels = labels[idx]

    sigmas = torch.linspace(log(start_sigma), log(end_sigma), idx.shape[0]).exp()
    noise = torch.randn_like(images) * sigmas.view(-1, 1, 1, 1)
    inputs = images + noise
    targets = -noise

    return TensorDataset(inputs, labels, targets)


def load_celeba_data() -> Dataset:
    images = ImageFolder(
        path.join("data", "celeba-dataset", "img_align_celeba"),
        transform=Compose([Resize((48, 64)), ToTensor()])
    )
    images = torch.stack([image[0] for image in images], dim=0)

    labels = pd.read_csv(path.join("data", "celeba-dataset", "list_attr_celeba.csv"))
    labels = torch.from_numpy(labels.to_numpy()[:, 1:].astype(np.float32))

    return _format_celeba_dataset(
        images, labels, start_sigma=1.0, end_sigma=0.01
    )


def load_celeba_small_data() -> Dataset:
    images = ImageFolder(
        path.join("data", "celeba-small-dataset", "img_align_celeba"),
        transform=Compose([Resize((48, 64)), ToTensor()])
    )
    images = torch.stack([image[0] for image in images], dim=0)

    labels = pd.read_csv(path.join("data", "celeba-small-dataset", "list_attr_celeba.csv"))
    labels = torch.from_numpy(labels.to_numpy()[:, 1:].astype(np.float32))

    return _format_celeba_dataset(
        images, labels, start_sigma=1.0, end_sigma=0.01
    )



