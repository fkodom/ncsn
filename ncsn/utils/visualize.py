from typing import Tuple

import torch
from torch import Tensor
from matplotlib.animation import Animation

from ncsn import NCSN
from ncsn.utils.device import get_module_device
from animate import animate_datacubes


def _concatenate_samples(samples: Tensor, size: Tuple[int, int]) -> Tensor:
    nrow, ncol = size
    ntotal = nrow * ncol
    return torch.cat(
        tuple(
            torch.cat(tuple(samples[i : i + nrow]), dim=-2)
            for i in range(0, ntotal, nrow)
        ),
        dim=-1
    )


def visualize_samples(
    samples: Tensor,
    size: Tuple[int, int],
    frame_rate: float = 25,
    frame_size: Tuple[int, int] = (5, 5),
) -> Animation:
    samples = _concatenate_samples(samples, size=size).clamp(0, 1)
    return animate_datacubes(
        (samples.cpu().numpy(),),
        frame_rate=frame_rate,
        frame_size=frame_size,
    )


def visualize_random_samples(
    ncsn: NCSN,
    num_classes: int = 10,
    size: Tuple[int, int] = (8, 8),
    seed: int = None,
) -> Animation:
    if seed is not None:
        torch.manual_seed(seed)

    samples = ncsn.sample(
        size[0] * size[1],
        num_classes=num_classes,
        return_all=True,
    ).squeeze()

    return visualize_samples(samples, size=size)


def visualize_class_samples(
    ncsn: NCSN,
    label: int,
    size: Tuple[int, int] = (8, 8),
    seed: int = None,
) -> Animation:
    if seed is not None:
        torch.manual_seed(seed)

    device = get_module_device(ncsn)
    labels = label * torch.ones(size[0] * size[1], device=device)
    samples = ncsn.sample_from(labels, return_all=True).squeeze()

    return visualize_samples(samples, size=size)


def visualize_class_iterations(
    ncsn: NCSN,
    start_label: int,
    end_label: int,
    num_classes: int = 10,
    num_samples: int = 10,
    seed: int = None,
) -> Animation:
    if seed is not None:
        torch.manual_seed(seed)

    device = get_module_device(ncsn)
    values = torch.linspace(0, 1, num_samples, dtype=torch.float, device=device)
    labels = torch.zeros(num_samples, num_classes, dtype=torch.float, device=device)
    labels[:, end_label] = values
    labels[:, start_label] = values.sort(dim=0, descending=True)[0]
    samples = ncsn.sample_from(
        labels, identical_noise=True, return_all=True
    ).squeeze()

    return visualize_samples(samples, size=(1, num_samples))
