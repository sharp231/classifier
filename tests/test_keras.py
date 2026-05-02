import numpy as np
from PIL import Image

from karas import preprocess


def test_preprocess_returns_normalized_image_array(tmp_path):
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (100, 80), color="red")
    image.save(image_path)

    result = preprocess(image_path)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 32, 32, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_converts_grayscale_to_rgb(tmp_path):
    image_path = tmp_path / "gray.png"
    image = Image.new("L", (100, 100), color=128)
    image.save(image_path)

    result = preprocess(image_path)

    assert result.shape == (1, 32, 32, 3)