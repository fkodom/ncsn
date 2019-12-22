from math import log
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn

from ncsn.utils.data import one_hot
from ncsn.utils.device import get_module_device


class NCSN(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        return torch.tensor(0.0)

    def sample_from(
        self,
        labels: Tensor,
        num_classes: int = 10,
        steps: int = 50,
        eps: float = 0.98,
        sigma_start: float = 1.0,
        sigma_end: float = 0.01,
        identical_noise: bool = False,
        return_all: bool = False,
    ) -> Tensor:
        n = labels.shape[0]
        device = labels.device
        if len(labels.shape) < 2:
            labels = torch.stack(
                [one_hot(label.item(), num_classes, device=device) for label in labels],
                dim=0
            )
        else:
            labels = labels.type(torch.float)

        sigmas = torch.linspace(log(sigma_start), log(sigma_end), steps, device=device).exp()
        samples = torch.rand((n, *self.input_size), device=device)

        all_samples = []
        for sigma in sigmas:
            if identical_noise:
                noise = sigma * torch.stack(
                    samples.shape[0] * [torch.randn_like(samples[0])],
                    dim=0
                )
            else:
                noise = sigma * torch.randn_like(samples)

            with torch.no_grad():
                samples = samples + (2 * eps) ** 0.5 * noise
                samples = samples + eps * self(samples, labels)
                samples = samples.clamp(0, 1)

            if return_all:
                all_samples.append(samples)

        if return_all:
            return torch.stack(all_samples, 1)  # .clamp(0, 1)
        else:
            return samples.clamp(0, 1)

    def sample(
        self,
        n: int,
        num_classes: int = 10,
        steps: int = 100,
        eps: float = 0.98,
        sigma_start: float = 1.0,
        sigma_end: float = 0.05,
        return_all: bool = False,
    ) -> Tensor:
        device = get_module_device(self)
        labels = torch.randint(0, num_classes, size=(n,), device=device)

        return self.sample_from(
            labels,
            num_classes=num_classes,
            steps=steps,
            eps=eps,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            return_all=return_all,
        )
