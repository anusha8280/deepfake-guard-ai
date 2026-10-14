
import os
import tempfile
import threading

import librosa
import numpy as np
from transformers import pipeline


MODEL_NAME = "Vansh180/deepfake-audio-wav2vec2"

MAX_AUDIO_SECONDS = 30
SAMPLE_RATE = 16000

_audio_detector = None
_model_lock = threading.Lock()


def get_audio_detector():
    """Load the audio model once and reuse it."""
    global _audio_detector

    if _audio_detector is None:
        with _model_lock:
            if _audio_detector is None:
                _audio_detector = pipeline(
                    "audio-classification",
                    model=MODEL_NAME,
                    device=-1,
                )

    return _audio_detector


def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    """Analyze a short audio recording for possible spoofed speech."""

    if not audio_bytes:
        raise ValueError("The uploaded audio file is empty.")

    extension = os.path.splitext(filename)[1].lower()

    supported_extensions = {".wav", ".mp3", ".flac", ".ogg"}

    if extension not in supported_extensions:
        raise ValueError(
            "Unsupported audio format. Upload WAV, MP3, FLAC, or OGG."
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = temporary_file.name

        audio, _ = librosa.load(
            temporary_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=MAX_AUDIO_SECONDS + 1,
        )

        if len(audio) == 0:
            raise ValueError("No readable audio was found in this file.")

        duration_seconds = len(audio) / SAMPLE_RATE

        if duration_seconds > MAX_AUDIO_SECONDS:
            raise ValueError(
                "Audio must not exceed 30 seconds. Upload a shorter clip."
            )

        if not np.isfinite(audio).all():
            raise ValueError("The audio contains invalid numerical values.")

        if float(np.max(np.abs(audio))) < 1e-6:
            raise ValueError("The audio appears to be silent.")

        detector = get_audio_detector()

        with _model_lock:
            predictions = detector(
                {
                    "array": audio.astype(np.float32),
                    "sampling_rate": SAMPLE_RATE,
                },
                top_k=None,
            )

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
            raise RuntimeError("The audio model returned no predictions.")

        scores.sort(key=lambda item: item["score"], reverse=True)

        def classify_label(label: str) -> str:
            normalized = label.strip().lower().replace("_", " ").replace("-", " ")

            if any(word in normalized for word in (
                "spoof", "fake", "deepfake", "synthetic", "generated"
            )):
                return "Possible AI-generated or spoofed speech"

            if any(word in normalized for word in (
                "bonafide", "bona fide", "real", "genuine", "human"
            )):
                return "Likely genuine speech"

            return "Uncertain"

        best = scores[0]

        return {
            "prediction": classify_label(best["label"]),
            "confidence": best["score_percent"],
            "model_label": best["label"],
            "scores": scores,
            "duration_seconds": round(duration_seconds, 2),
            "sample_rate": SAMPLE_RATE,
            "model": MODEL_NAME,
            "device": "cpu",
            "note": (
                "This is a model-based speech screening result. "
                "It may produce false positives or false negatives and "
                "does not prove whether a recording is authentic."
            ),
        }

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
