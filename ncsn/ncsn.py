import os
from abc import abstractmethod
from typing import Union, Dict
from math import log

import torch
from torch import nn, Tensor
import torch.nn.functional as f
from torch.utils.data import DataLoader

from ncsn.utils.device import get_module_device
from ncsn.utils.visualize import visualize_samples


class NCSN(nn.Module):
    def sample_from(
        self,
        labels: Tensor,
        num_classes: int = 10,
        steps: int = 25,
        epsilon: float = 0.98,
        sigma_start: float = 1.0,
        sigma_end: float = 0.01,
        identical_noise: bool = False,
        return_all: bool = False,
    ) -> Tensor:
        n = labels.shape[0]
        device = labels.device
        if len(labels.shape) < 2:
            labels = one_hot(labels, num_classes)
        labels = labels.float()

        sigmas = torch.linspace(log(sigma_start), log(sigma_end), steps, device=device).exp()
        samples = torch.randn((n, *self.input_size), device=device)

        all_samples = []
        for sigma in sigmas:
            if identical_noise:
                noise = sigma * torch.stack(
                    samples.shape[0] * [torch.randn_like(samples[0])],
                    dim=0,
                )
            else:
                noise = sigma * torch.randn_like(samples)

            with torch.no_grad():
                samples = samples + (2 * epsilon) ** 0.5 * noise
                samples = samples + epsilon * self(samples, labels)
                samples = samples.clamp(0, 1)

            if return_all:
                all_samples.append(samples)

        if return_all:
            return torch.stack(all_samples, 1)
        else:
            return samples

    def sample(
        self,
        n: int,
        num_classes: int = 10,
        steps: int = 25,
        epsilon: float = 0.98,
        sigma_start: float = 1.0,
        sigma_end: float = 0.05,
        video: bool = False,
    ) -> Tensor:
        device = get_module_device(self)
        labels = torch.randint(0, num_classes, size=(n,), device=device)

        return self.sample_from(
            labels,
            num_classes=num_classes,
            steps=steps,
            epsilon=epsilon,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            return_all=video,
        )


def one_hot(labels: Tensor, num_classes: int):
    values = torch.arange(num_classes, dtype=labels.dtype, device=labels.device)
    return (labels.unsqueeze(1) == values.unsqueeze(0)).float()


try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import MLFlowLogger
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

    class NCSNTrainingModule(pl.LightningModule):
        def configure_optimizers(self):
            optimizer = getattr(
                torch.optim, self.hparams.get("optimizer", torch.optim.Adam)
            )
            lr = self.hparams.get("lr", 1e-3)
            return optimizer(self.parameters(), lr=lr)

        @abstractmethod
        def train_dataloader(self) -> DataLoader:
            """Fetches a DataLoader for sampling training data."""

        def training_step(self, batch, idx) -> Union[
            int, Dict[str, Union[Tensor, Dict[str, Tensor]]]
        ]:
            inputs, labels = batch
            num_classes = self.hparams.get("num_classes", 10)
            start_sigma = self.hparams.get("start_sigma", 1.0)
            end_sigma = self.hparams.get("end_sigma", 0.01)

            sigmas = torch.linspace(
                log(start_sigma), log(end_sigma), len(inputs), device=inputs.device
            ).exp().view(-1, 1, 1, 1)
            noise = torch.randn_like(inputs) * sigmas
            inputs = inputs + noise
            targets = -noise
            labels = one_hot(labels, num_classes)

            loss = f.mse_loss(self.forward(inputs, labels), targets.detach())

            return {"loss": loss, "log": {"train_loss": loss}}

        def on_train_end(self) -> None:
            if isinstance(self.logger, MLFlowLogger):
                self.logger.experiment.set_terminated(self.logger.run_id)

                samples = self.sample(64, video=True).detach().squeeze().cpu()
                animation = visualize_samples(samples, size=(8, 8), show=False)

                if not os.path.exists("media"):
                    os.mkdir("media")
                path = os.path.join(
                    "media", f"{self.logger.experiment_name}-{self.logger.tags[MLFLOW_RUN_NAME]}.gif"
                )
                animation.save(path)
                self.logger.experiment.log_artifact(self.logger.run_id, local_path=path)
except ImportError:
    msg = "Please install PyTorch Lightning to train models:\n" \
          "   pip install pytorch-lightning"

    # noinspection PyPep8Naming,PyUnusedLocal
    def NCSNTrainingModule(*args, **kwargs): raise ImportError(msg)
