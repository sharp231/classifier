import numpy as np
import pytest
from PIL import Image
from tensorflow import image

import app as app_module
from preprocessing import preprocess
import inference as inference_module


class SequenceModel:
    def __init__(self):
        self.call_count = 0
        self.outputs = [
            [0.9, 0.1, 0.0, 0, 0, 0, 0, 0, 0, 0],
            [0.0, 0.8, 0.2, 0, 0, 0, 0, 0, 0, 0],
            [0.0, 0.3, 0.7, 0, 0, 0, 0, 0, 0, 0],
        ]

    def predict(self, input_array, verbose=0):
        assert input_array.shape == (1, 32, 32, 3)
        assert input_array.dtype == np.float32

        output = self.outputs[self.call_count]
        self.call_count += 1

        return np.array([output], dtype=np.float32)


def test_predict_image_averages_tta_predictions(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 32), color="red").save(image_path)

    model = SequenceModel()
    used_crop_ratios = []

    def recording_preprocess(
        filepath,
        image_size,
        crop_right_ratio,
    ):
        used_crop_ratios.append(crop_right_ratio)

        return preprocess(
            filepath,
            image_size=image_size,
            crop_right_ratio=crop_right_ratio,
        )

    monkeypatch.setattr(app_module, "get_model", lambda: model)
    monkeypatch.setattr(
        inference_module,
        "preprocess",
        recording_preprocess,
    )
    monkeypatch.setattr(
        app_module,
        "TTA_CROP_RATIOS",
        (1.0, 0.8, 0.65),
    )

    predictions = app_module.predict_image(image_path)

    assert model.call_count == 3
    assert used_crop_ratios == [1.0, 0.8, 0.65]
    assert len(predictions) == 3

    assert predictions[0]["label"] == "自動車"
    assert predictions[0]["probability"] == pytest.approx(40.0)

    probabilities = {
        prediction["label"]: prediction["probability"] for prediction in predictions
    }

    assert probabilities == pytest.approx(
        {
            "飛行機": 30.0,
            "自動車": 40.0,
            "鳥": 30.0,
        }
    )


def test_predict_probabilities_returns_all_class_averages(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 32), color="red").save(image_path)

    model = SequenceModel()

    probabilities = inference_module.predict_probabilities(
        image_path,
        model=model,
        crop_ratios=(1.0, 0.8, 0.65),
        class_count=10,
    )

    assert model.call_count == 3
    assert probabilities.shape == (10,)

    np.testing.assert_allclose(
        probabilities,
        [0.3, 0.4, 0.3, 0, 0, 0, 0, 0, 0, 0],
        atol=1e-7,
    )


def test_predict_probabilities_supports_baseline(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 32), color="red").save(image_path)

    model = SequenceModel()

    probabilities = inference_module.predict_probabilities(
        image_path,
        model=model,
        crop_ratios=(1.0,),
        class_count=10,
    )

    assert model.call_count == 1

    np.testing.assert_allclose(
        probabilities,
        [0.9, 0.1, 0.0, 0, 0, 0, 0, 0, 0, 0],
        atol=1e-7,
    )


def test_predict_probabilities_rejects_empty_crop_ratios(tmp_path):
    model = SequenceModel()

    with pytest.raises(ValueError, match="crop_ratios"):
        inference_module.predict_probabilities(
            tmp_path / "unused.png",
            model=model,
            crop_ratios=(),
            class_count=10,
        )

    assert model.call_count == 0


@pytest.mark.parametrize(
    "output",
    [
        [0.5, 0.5],
        [[0.1] * 10],
        [float("nan")] + [0.0] * 9,
    ],
)
def test_predict_probabilities_rejects_invalid_output(tmp_path, output):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color="red").save(image_path)

    model = SequenceModel()
    model.outputs = [output]

    with pytest.raises(RuntimeError, match="output"):
        inference_module.predict_probabilities(
            image_path,
            model=model,
            crop_ratios=(1.0,),
            class_count=10,
        )


def test_predict_image_uses_baseline_by_default(tmp_path, monkeypatch):
    image_path = tmp_path / "baseline.png"
    Image.new("RGB", (64, 32), color="red").save(image_path)

    model = SequenceModel()
    monkeypatch.setattr(app_module, "get_model", lambda: model)

    predictions = app_module.predict_image(image_path)

    assert app_module.TTA_CROP_RATIOS == (1.0,)
    assert model.call_count == 1
    assert len(predictions) == 3
    assert predictions[0]["label"] == "飛行機"
    assert predictions[0]["probability"] == pytest.approx(90.0)
