from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm


def _annotate_frame(
    image: np.ndarray,
    positions: np.ndarray,
    marker: str = "rectangle",
    color: str = "green",
) -> np.ndarray:
    """Annotates each target/detection in the image using the specified marker.

    :param image: Image (or frame) to annotate
    :param positions: Array of target coordinates, with either shape (N, 2)
        or (N, 4).  If (N, 2), Columns contain (row, col) coordinates for
        each target.  If (N, 4), columns cotain (frame, row, col, width, height)
        where (width, height) define the marker size for each target.
    :param marker: Specifies the marker shape.  Available: "rectangle", "ellipse"
    :param color: Specifies the marker outline color.  (e.g. green, red, blue)
    :return: Annotated image
    """
    img = Image.new("RGB", image.shape[:2])
    draw = ImageDraw.Draw(img)

    if positions.shape[1] == 2:
        # Default marker radius of 5
        marker_sizes = 5 * np.ones((positions.shape[0], 2), dtype=positions.dtype)
        positions = np.concatenate((positions, marker_sizes), 1)

    positions = np.stack(
        (
            positions[:, 1] - positions[:, 3] / 2,
            positions[:, 0] - positions[:, 2] / 2,
            positions[:, 1] + positions[:, 3] / 2,
            positions[:, 0] + positions[:, 2] / 2,
        ),
        axis=1,
    )

    for row, col, width, height in positions:
        if marker == "rectangle":
            draw.rectangle([row, col, width, height], outline=color, width=1)
        elif marker == "ellipse":
            draw.ellipse([row, col, width + 1, height + 1], outline=color, width=1)
        else:
            raise ValueError(f"Marker color {color} not recognized.")

    annotation = np.array(img).astype(image.dtype)
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    image = image + annotation

    mask = np.any(annotation, -1)
    image[mask, :] = annotation[mask, :]
    image = np.clip(image, 0, 1)

    return image


def _annotate(
    datacube: np.ndarray,
    positions: np.ndarray,
    marker: str = "rectangle",
    color: str = "green",
    annotation_type: str = "",
    verbose: bool = False,
) -> np.ndarray:
    """Annotates each target/detection in the datacube using the specified marker.

    :param datacube: Datacube (set of image frames) to annotate.
        Shape: (frames, rows, cols)
    :param positions: Array of marker coordinates, with either shape (N, 3)
        or (N, 5).  If (N, 3), Columns contain (frame, row, col) coordinates for
        each marker.  If (N, 5), columns cotain (frame, row, col, width, height)
        where (width, height) define the marker size for each marker.
    :param marker: Specifies the marker shape.  Available: "rectangle", "circle"
    :param color: Specifies the marker outline color.  (e.g. green, red, blue)
    :param annotation_type: For internal use only.  Specifies the type of
        annotation being performed, which is only used for verbose outputs.
        Can be any descriptive string for this annotation (e.g. "targets").
    :param verbose: If True, a progress bar is displayed for annotations.
    :return: Annotated datacube.  Shape:  (frames, rows, cols, 3)
    """
    frames = []
    frame_numbers = positions[:, 0].astype(np.int)

    if positions.shape[1] not in [3, 5]:
        raise ValueError(
            f"Invalid shape for positions condition: {positions.shape}."
            f"Must have either 3 or 5 columns."
        )

    if verbose:
        progress_bar = tqdm(
            total=datacube.shape[0], desc=f"Annotating {annotation_type}".strip()
        )
    else:
        progress_bar = None

    for i, frame in enumerate(datacube):
        mask = frame_numbers == i
        frame_detections = positions[mask, 1:]
        annotated_frame = _annotate_frame(
            frame, frame_detections, marker=marker, color=color
        )
        frames.append(annotated_frame)

        if verbose:
            progress_bar.update()

    if verbose:
        progress_bar.close()

    return np.stack(frames, 0)


def annotate_detections(
    datacube: np.ndarray, positions: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """Places a green box around each detection in the datacube.

    :param datacube: Datacube (set of image frames) to annotate.
        Shape: (frames, rows, cols)
    :param positions: Array of detection coordinates, with either shape (N, 3)
        or (N, 5).  If (N, 3), Columns contain (frame, row, col) coordinates for
        each detection.  If (N, 5), columns cotain (frame, row, col, width, height)
        where (width, height) define the marker size for each detection.
    :param verbose: If True, a progress bar is displayed for annotations.
    :return: Annotated datacube.  Shape:  (frames, rows, cols, 3)
    """
    return _annotate(
        datacube,
        positions,
        marker="rectangle",
        color="green",
        annotation_type="detections",
        verbose=verbose,
    )


def annotate_targets(
    datacube: np.ndarray, positions: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """Places a red ellipse around each detection in the datacube.

    :param datacube: Datacube (set of image frames) to annotate.
        Shape: (frames, rows, cols)
    :param positions: Array of target coordinates, with either shape (N, 3)
        or (N, 5).  If (N, 3), Columns contain (frame, row, col) coordinates for
        each target.  If (N, 5), columns cotain (frame, row, col, width, height)
        where (width, height) define the marker size for each target.
    :param verbose: If True, a progress bar is displayed for annotations.
    :return: Annotated datacube.  Shape:  (frames, rows, cols, 3)
    """
    return _annotate(
        datacube,
        positions,
        marker="ellipse",
        color="red",
        annotation_type="targets",
        verbose=verbose,
    )
