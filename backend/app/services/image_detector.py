
import io
import threading

import torch
from PIL import Image, UnidentifiedImageError
from transformers import pipeline


MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

_detector = None
_model_lock = threading.Lock()


def get_detector():
    """Load the model once and reuse it for future requests."""
    global _detector

    if _detector is None:
        with _model_lock:
            if _detector is None:
                _detector = pipeline(
                    "image-classification",
                    model=MODEL_NAME,
                    device=-1,
                )

    return _detector


def analyze_image(image_bytes: bytes) -> dict:
    """Validate an image and return the model's actual prediction."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            if uploaded_image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError(
                    "Unsupported image format. Upload a JPEG, PNG, or WEBP image."
                )

            if uploaded_image.width * uploaded_image.height > 25_000_000:
                raise ValueError(
                    "Image dimensions are too large. Use an image with at most "
                    "25 million pixels."
                )

            uploaded_image.verify()

        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            image = uploaded_image.convert("RGB")

    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc

    detector = get_detector()

    with _model_lock:
        predictions = detector(image, top_k=None)

    scores = []

    for item in predictions:
        label = str(item["label"])
        score = float(item["score"])

        scores.append({
            "label": label,
            "score": round(score, 6),
            "score_percent": round(score * 100, 2),
        })

    if not scores:
        raise RuntimeError("The model returned no predictions.")

    scores.sort(key=lambda item: item["score"], reverse=True)
    best_prediction = scores[0]

    return {
        "prediction": best_prediction["label"],
        "confidence": best_prediction["score_percent"],
        "scores": scores,
        "model": MODEL_NAME,
        "device": "cpu",
        "note": (
            "This is a model prediction, not definitive proof that the "
            "image is real or manipulated."
        ),
    }
