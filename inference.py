from pathlib import Path
from typing import Protocol

import numpy as np

from preprocessing import IMAGE_SIZE, preprocess


class PredictionModel(Protocol):
    def predict(
        self,
        input_array: np.ndarray,
        verbose: int = 0,
    ) -> np.ndarray: ...


def predict_probabilities(
    filepath: str | Path,
    model: PredictionModel,
    crop_ratios: tuple[float, ...],
    class_count: int,
) -> np.ndarray:
    if not crop_ratios:
        raise ValueError("crop_ratios must not be empty")

    prediction_results = []

    for crop_ratio in crop_ratios:
        input_array, _ = preprocess(
            filepath,
            image_size=IMAGE_SIZE,
            crop_right_ratio=crop_ratio,
        )

        prediction = np.asarray(model.predict(input_array, verbose=0))

        if prediction.shape != (1, class_count):
            raise RuntimeError(
                "Unexpected model output shape:"
                f" expected={(1, class_count)}"
                f" actual={prediction.shape}"
            )

        if not np.isfinite(prediction).all():
            raise RuntimeError("Model output contains non-finite values")

        prediction_results.append(prediction[0])

    return np.mean(
        np.stack(prediction_results, axis=0),
        axis=0,
    )
