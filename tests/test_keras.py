import numpy as np
import pytest
from PIL import Image

from karas import preprocess as legacy_preprocess
from preprocessing import preprocess


def test_preprocess_returns_normalized_image_array(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), color="red").save(image_path)

    input_array, image = preprocess(image_path)

    assert isinstance(input_array, np.ndarray)
    assert input_array.shape == (1, 32, 32, 3)
    assert input_array.dtype == np.float32
    assert input_array.min() >= 0.0
    assert input_array.max() <= 1.0
    assert image.size == (32, 32)
    assert image.mode == "RGB"
    np.testing.assert_allclose(input_array[0, 0, 0], [1.0, 0.0, 0.0])

def test_preprocess_converts_grayscale_to_rgb(tmp_path):
    image_path = tmp_path / "gray.png"
    Image.new("L", (100, 100), color=128).save(image_path)
    input_array, image = preprocess(image_path)

    assert input_array.shape == (1, 32, 32, 3)
    assert image.mode == "RGB"
    np.testing.assert_allclose(input_array, 128 / 255.0, rtol=1e-6)


def test_legacy_preprocess_returns_shared_array(tmp_path):
    image_path = tmp_path / "legacy.png"
    image = Image.new("RGB", (64, 32), color="blue")
    image.paste((255, 0, 0), (0, 0, 32, 32))
    image.save(image_path)

    expected, _ = preprocess(image_path, image_size=16, crop_right_ratio=0.5)
    result = legacy_preprocess(image_path, img_size=16, crop_right_ratio=0.5)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 16, 16, 3)
    np.testing.assert_array_equal(result, expected)


def test_preprocess_crops_right_side(tmp_path):
    image_path = tmp_path / "split.png"
    image = Image.new("RGB", (64, 32), color="blue")
    image.paste((255, 0, 0), (0, 0, 32, 32))
    image.save(image_path)

    input_array, _ = preprocess(image_path, crop_right_ratio=0.5)

    expected = np.zeros((1, 32, 32, 3), dtype=np.float32)
    expected[..., 0] = 1.0
    np.testing.assert_allclose(input_array, expected)


def test_preprocess_applies_exif_rotation(tmp_path):
    image_path = tmp_path / "rotate.png"
    image = Image.new("RGB", (32, 32), color="blue")
    image.paste((255, 0, 0), (0, 0, 16, 32))

    exif = image.getexif()
    exif[274] = 6
    image.save(image_path, exif=exif)

    input_array, _ = preprocess(image_path)

    # Orientation 6の補正後は、左半分の赤が上半分へ移る。
    np.testing.assert_allclose(input_array[0, 8, 16], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(input_array[0, 24, 16], [0.0, 0.0, 1.0])


@pytest.mark.parametrize("crop_ratio", [0.0, -0.1, 1.01])
def test_preprocess_rejects_invalid_crop_ratio(tmp_path, crop_ratio):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), color="red").save(image_path)

    with pytest.raises(ValueError, match="crop_right_ratio"):
        preprocess(image_path, crop_right_ratio=crop_ratio)


@pytest.mark.parametrize("crop_ratio", [0.01, 0.1, 1.0])
def test_preprocess_accepts_valid_crop_ratio_boundaries(tmp_path, crop_ratio):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color="red").save(image_path)

    input_array, image = preprocess(
        image_path,
        crop_right_ratio=crop_ratio,
    )

    assert input_array.shape == (1, 32, 32, 3)
    assert image.size == (32, 32)
