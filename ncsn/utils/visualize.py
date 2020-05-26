from typing import Sequence, Iterable, Tuple

import torch
from torch import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, Animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ncsn.utils.device import get_module_device


def animate_frames(
    frames: Sequence[np.ndarray],
    frame_rate: float = 10,
    titles: Iterable[str] = (),
    frame_size: Tuple[float, float] = (4, 5),
    colorbar: bool = False,
    show: bool = True,
) -> Animation:
    """Play back datacubes as a Matplotlib animate.  Datacubes are aligned
    side-by-side, and x-and y-axes are shared between all figures.

    :param frames: Tuple or List of datacubes to animate
    :param frame_rate: Frames per second for the animate
    :param titles: Tuple or List of subtitles to display above each datacube animate
    :param frame_size: Frames per second for the animate
    :param colorbar: If True, a colorbar is shown next to each datacube animate
        (default: False)
    :param show: If True, displays the animation in a new window.
    :return: Matplotlib FuncAnimation object, which contains the animate.
        Can be used for easily saving datacube animations to HTML, MP4.
    """
    fig, ax = plt.subplots(1, len(frames), sharex="all", sharey="all", squeeze=False)
    frame_size = (len(frames) * frame_size[0], frame_size[1])
    fig.set_size_inches(frame_size)
    images = []

    frames = tuple((x - x.min()) / (x.max() - x.min()) for x in frames)

    for i, x in enumerate(frames):
        if len(x.shape) > 3:
            chan_axis = np.where(np.array(x[0].shape) == 3)[0][0]
            img = ax[0, i].imshow(np.moveaxis(x[0], chan_axis, -1))
        else:
            img = ax[0, i].imshow(x[0], cmap=plt.gray())
        images.append(img)
        if colorbar:
            divider = make_axes_locatable(ax[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, cax=cax)

    for i, title in enumerate(titles[: len(frames)]):
        ax[0, i].set_title(title)

    def animate(frame_number):
        for image, cube in zip(images, frames):
            if len(x.shape) > 3:
                chan_axis = np.where(np.array(x[frame_number].shape) == 3)[0][0]
                image.set_array(np.moveaxis(cube[frame_number], chan_axis, -1))
            else:
                image.set_array(cube[frame_number])

        return images

    plt.tight_layout()
    animation = FuncAnimation(
        fig,
        animate,
        frames=len(frames[0]),
        interval=int(1000 / frame_rate),
        repeat=True,
        blit=True,
    )
    if show:
        plt.show()

    return animation


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
    show: bool = True,
) -> Animation:
    samples = _concatenate_samples(samples, size=size).clamp(0, 1)
    return animate_frames(
        (samples.cpu().numpy(),),
        frame_rate=frame_rate,
        frame_size=frame_size,
        show=show,
    )


def visualize_random_samples(
    ncsn: nn.Module,
    num_classes: int = 10,
    size: Tuple[int, int] = (8, 8),
    seed: int = None,
    show: bool = True,
) -> Animation:
    if seed is not None:
        torch.manual_seed(seed)

    samples = ncsn.sample(
        size[0] * size[1],
        num_classes=num_classes,
        video=True,
    ).squeeze()

    return visualize_samples(samples, size=size, show=show)


def visualize_class_samples(
    ncsn: nn.Module,
    label: int,
    size: Tuple[int, int] = (8, 8),
    seed: int = None,
    show: bool = True,
) -> Animation:
    if seed is not None:
        torch.manual_seed(seed)

    device = get_module_device(ncsn)
    labels = label * torch.ones(size[0] * size[1], device=device)
    samples = ncsn.sample_from(labels, return_all=True).squeeze()

    return visualize_samples(samples, size=size, show=show)


def visualize_class_iterations(
    ncsn: nn.Module,
    start_label: int,
    end_label: int,
    num_classes: int = 10,
    num_samples: int = 10,
    seed: int = None,
    show: bool = True,
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

    return visualize_samples(samples, size=(1, num_samples), show=show)
