
import csv
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from app.services.image_detector import analyze_image
from app.services.video_detector import analyze_video
from app.services.audio_detector import analyze_audio


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "evaluation" / "dataset.csv"
RESULTS_FILE = BASE_DIR / "evaluation" / "results.csv"

VALID_MODALITIES = {"image", "video", "audio"}
VALID_LABELS = {"real", "fake"}


def normalize_label(value):
    """Convert a model label to real, fake, or unknown."""
    label = str(value).strip().lower()

    if label in {"real", "genuine", "bonafide", "bona fide"}:
        return "real"

    if label in {"fake", "spoof", "deepfake", "synthetic"}:
        return "fake"

    return "unknown"


def predict_file(modality, file_path):
    """Run the existing detector for the selected media type."""
    file_bytes = file_path.read_bytes()

    if modality == "image":
        result = analyze_image(file_bytes)
        prediction = normalize_label(result["prediction"])
        confidence = float(result["confidence"])

    elif modality == "video":
        result = analyze_video(file_bytes, file_path.name)

        # The current video detector analyzes sampled frames.
        # Its fake-frame score is a percentage from 0 to 100.
        fake_score = float(result["average_fake_frame_score"])
        prediction = "fake" if fake_score >= 50 else "real"
        confidence = fake_score if prediction == "fake" else 100 - fake_score

    elif modality == "audio":
        result = analyze_audio(file_bytes, file_path.name)

        # Use the raw model class label, not a descriptive message.
        prediction = normalize_label(result["model_label"])
        confidence = float(result["confidence"])

    else:
        raise ValueError(f"Unsupported media type: {modality}")

    return prediction, confidence, result


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {DATASET_FILE}\n"
            "Create evaluation/dataset.csv first."
        )

    with DATASET_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The dataset CSV has no data rows.")

    for row_number, row in enumerate(rows, start=2):
        modality = row.get("modality", "").strip().lower()
        label = row.get("label", "").strip().lower()
        relative_path = row.get("path", "").strip()

        if modality not in VALID_MODALITIES:
            raise ValueError(
                f"CSV row {row_number}: invalid modality '{modality}'."
            )

        if label not in VALID_LABELS:
            raise ValueError(
                f"CSV row {row_number}: label must be real or fake."
            )

        if not relative_path:
            raise ValueError(f"CSV row {row_number}: missing file path.")

    return rows


def main():
    rows = load_dataset()
    evaluated = []

    print("DeepfakeGuard AI - Model Evaluation")
    print("=" * 55)
    print(f"Dataset entries: {len(rows)}")

    for index, row in enumerate(rows, start=1):
        modality = row["modality"].strip().lower()
        true_label = row["label"].strip().lower()
        file_path = (BASE_DIR / row["path"].strip()).resolve()

        # Prevent dataset paths from pointing outside the backend directory.
        if not file_path.is_relative_to(BASE_DIR.resolve()):
            raise ValueError(f"Unsafe dataset path: {row['path']}")

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Missing file on CSV row {index + 1}: {file_path}"
            )

        print(
            f"\n[{index}/{len(rows)}] "
            f"{modality}: {file_path.name}"
        )

        try:
            prediction, confidence, result = predict_file(
                modality,
                file_path,
            )

            if prediction not in VALID_LABELS:
                print(
                    "Skipped: model returned an unrecognized label "
                    f"({prediction!r}). Check the model output."
                )
                continue

            print(f"Actual: {true_label}")
            print(f"Predicted: {prediction}")
            print(f"Confidence/score: {confidence:.2f}%")

            evaluated.append({
                "modality": modality,
                "filename": file_path.name,
                "actual": true_label,
                "predicted": prediction,
                "confidence_percent": round(confidence, 2),
                "correct": prediction == true_label,
                "model": result.get("model", ""),
            })

        except Exception as error:
            print(f"ERROR: {type(error).__name__}: {error}")
            raise RuntimeError(
                f"Evaluation stopped at {file_path.name}. "
                "Fix this error before trusting the metrics."
            ) from error

    if not evaluated:
        raise RuntimeError("No valid predictions were evaluated.")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "modality",
                "filename",
                "actual",
                "predicted",
                "confidence_percent",
                "correct",
                "model",
            ],
        )
        writer.writeheader()
        writer.writerows(evaluated)

    print("\n\nEVALUATION SUMMARY")
    print("=" * 55)

    for modality in sorted(VALID_MODALITIES):
        subset = [
            row for row in evaluated
            if row["modality"] == modality
        ]

        if not subset:
            print(f"\n{modality.upper()}: No valid samples")
            continue

        actual = [row["actual"] for row in subset]
        predicted = [row["predicted"] for row in subset]

        print(f"\n{modality.upper()}")
        print(f"Samples evaluated: {len(subset)}")
        print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                actual,
                predicted,
                labels=["real", "fake"],
                target_names=["Real", "Fake"],
                zero_division=0,
            )
        )

        print("Confusion matrix (rows=actual, columns=predicted):")
        print("                 Predicted Real  Predicted Fake")

        matrix = confusion_matrix(
            actual,
            predicted,
            labels=["real", "fake"],
        )

        print(f"Actual Real      {matrix[0][0]:14d}  {matrix[0][1]:14d}")
        print(f"Actual Fake      {matrix[1][0]:14d}  {matrix[1][1]:14d}")

    print(f"\nDetailed results saved to: {RESULTS_FILE}")
    print(
        "\nImportant: these metrics describe only the supplied test files. "
        "They do not establish performance on all real-world media."
    )


if __name__ == "__main__":
    main()
