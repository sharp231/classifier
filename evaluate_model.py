import argparse
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image

from app import LABELS, MODEL_PATH, get_model
from inference import PredictionModel, predict_probabilities
from tensorflow import keras as tf_keras


def evaluate(
    image_paths: list[Path],
    labels: np.ndarray,
    model: PredictionModel,
    crop_ratios: tuple[float, ...],
) -> dict:
    predicted_labels = []
    started_at = perf_counter()

    for image_path in image_paths:
        probabilities = predict_probabilities(
            image_path,
            model=model,
            crop_ratios=crop_ratios,
            class_count=len(LABELS),
        )
        predicted_labels.append(int(np.argmax(probabilities)))

    elapsed_seconds = perf_counter() - started_at

    predictions = np.asarray(predicted_labels)
    correct = predictions == labels
    per_class = {}

    for class_index, class_name in enumerate(LABELS):
        selected = labels == class_index
        sample_count = int(selected.sum())
        correct_count = int(correct[selected].sum())

        per_class[class_name] = {
            "samples": sample_count,
            "correct": correct_count,
            "accuracy": (correct_count / sample_count if sample_count > 0 else None),
        }

    return {
        "crop_ratios": list(crop_ratios),
        "samples": len(image_paths),
        "accuracy": float(correct.mean()),
        "elapsed_seconds": elapsed_seconds,
        "milliseconds_per_image": (elapsed_seconds / len(image_paths) * 1000),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.limit <= 10_000:
        parser.error("--limit must be between 1 and 10000")

    (_, _), (images, labels) = tf_keras.datasets.cifar10.load_data()
    images = images[: args.limit]
    labels = labels[: args.limit].reshape(-1)

    model = get_model()
    results = {}

    conditions = {
        "baseline": (1.0,),
        "tta": (1.0, 0.8, 0.65),
    }

    with TemporaryDirectory(prefix="classifier-evaluation-") as directory:
        image_paths = []

        for image_index, image_array in enumerate(images):
            image_path = Path(directory) / f"{image_index}.png"
            Image.fromarray(image_array).save(image_path)
            image_paths.append(image_path)

        for crop_ratios in conditions.values():
            predict_probabilities(
                image_paths[0],
                model=model,
                crop_ratios=crop_ratios,
                class_count=len(LABELS),
            )

        for condition_name, crop_ratios in conditions.items():
            print(
                f"Evaluating {condition_name}: {len(image_paths)} images",
                flush=True,
            )

            result = evaluate(
                image_paths,
                labels,
                model,
                crop_ratios,
            )
            results[condition_name] = result

            print(
                f"accuracy={result['accuracy']:.4f}, "
                f"elapsed={result['elapsed_seconds']:.2f}s, "
                f"ms/image={result['milliseconds_per_image']:.2f}",
                flush=True,
            )

    accuracy_difference = results["tta"]["accuracy"] - results["baseline"]["accuracy"]

    report = {
        "model": MODEL_PATH.name,
        "python": platform.python_version(),
        "dataset": "CIFAR-10 test",
        "selection": "first N images in dataset order",
        "samples": len(labels),
        "timing_scope": "image preprocessing and model prediction",
        "timing_excludes": [
            "dataset loading",
            "model loading",
            "temporary PNG creation",
            "warmup",
        ],
        "previous_direct_input_accuracy_rounded": 0.652,
        "evaluation_note": (
            "The training script also uses this test set as validation_data."
        ),
        "results": results,
        "accuracy_difference_percentage_points": accuracy_difference * 100,
    }

    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(f"TTA accuracy difference: {accuracy_difference * 100:+.2f} pp")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
