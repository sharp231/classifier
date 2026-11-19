from pathlib import Path

import numpy as np

from preprocessing import IMAGE_SIZE, preprocess as preprocess_image


def preprocess(
    filepath: str | Path,
    img_size: int = IMAGE_SIZE,
    crop_right_ratio: float = 1.0,
) -> np.ndarray:
    input_array, _ = preprocess_image(
        filepath,
        image_size=img_size,
        crop_right_ratio=crop_right_ratio,
    )
    return input_array
