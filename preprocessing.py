from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 32


def preprocess(
    filepath: str | Path,
    image_size: int = IMAGE_SIZE,
    crop_right_ratio: float = 1.0,
) -> tuple[np.ndarray, Image.Image]:
    if not 0 < crop_right_ratio <= 1.0:
        raise ValueError("crop_right_ratio must be between 0 and 1")

    with Image.open(filepath) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")

    if crop_right_ratio < 1.0:
        width, height = image.size
        crop_width = max(1, int(width * crop_right_ratio))
        image = image.crop((0, 0, crop_width, height))

    image = ImageOps.fit(
        image,
        (image_size, image_size),
        method=Image.Resampling.LANCZOS,
    )

    input_array = np.asarray(image, dtype=np.float32) / 255.0
    input_array = np.expand_dims(input_array, axis=0)

    return input_array, image