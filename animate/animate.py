from typing import Sequence, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, Animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from animate.annotate import annotate_targets, annotate_detections


def animate_datacubes(
    datacubes: Sequence[np.ndarray],
    targets: np.ndarray = None,
    detections: np.ndarray = None,
    frame_rate: float = 10,
    titles: Iterable[str] = (),
    frame_size: Tuple[float, float] = (4, 5),
    colorbar: bool = False,
    verbose: bool = False,
) -> Animation:
    """Play back datacubes as a Matplotlib animate.  Datacubes are aligned
    side-by-side, and x-and y-axes are shared between all figures.

    :param datacubes: Tuple or List of datacubes to animate
    :param detections: Array of detection coordinates.  Columns contain
        (frame, center_x, center_y, width, height).  Shape:  (N, 5)
    :param targets: Array of target coordinates.  Columns contain
        (frame, center_x, center_y, width, height).  Shape:  (N, 5)
    :param frame_rate: Frames per second for the animate
    :param titles: Tuple or List of subtitles to display above each datacube animate
    :param frame_size: Frames per second for the animate
    :param colorbar: If True, a colorbar is shown next to each datacube animate
        (default: False)
    :param verbose: If True, a progress bar is displayed for annotations.
        (default: False)
    :return: Matplotlib FuncAnimation object, which contains the animate.
        Can be used for easily saving datacube animations to HTML, MP4.
    """
    fig, ax = plt.subplots(1, len(datacubes), sharex="all", sharey="all", squeeze=False)
    frame_size = (len(datacubes) * frame_size[0], frame_size[1])
    fig.set_size_inches(frame_size)
    images = []

    if detections is not None or targets is not None:
        datacubes = tuple((x - x.min()) / (x.max() - x.min()) for x in datacubes)
    if targets is not None:
        datacubes = tuple(
            annotate_targets(x, targets, verbose=verbose) for x in datacubes
        )
    if detections is not None:
        datacubes = tuple(
            annotate_detections(x, detections, verbose=verbose) for x in datacubes
        )

    for i, x in enumerate(datacubes):
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

    for i, title in enumerate(titles[: len(datacubes)]):
        ax[0, i].set_title(title)

    def animate(frame_number):
        for image, cube in zip(images, datacubes):
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
        frames=len(datacubes[0]),
        interval=int(1000 / frame_rate),
        repeat=True,
        blit=True,
    )
    plt.show()

    return animation
