
import io
import os
import tempfile

import cv2

from app.services.image_detector import analyze_image


MAX_VIDEO_SIZE = 50 * 1024 * 1024
MAX_VIDEO_SECONDS = 120
MAX_FRAMES = 12


def analyze_video(video_bytes: bytes) -> dict:
    if not video_bytes:
        raise ValueError("The uploaded video is empty.")

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise ValueError("Video size must not exceed 50 MB.")

    temporary_path = None
    capture = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as temporary_file:
            temporary_file.write(video_bytes)
            temporary_path = temporary_file.name

        capture = cv2.VideoCapture(temporary_path)

        if not capture.isOpened():
            raise ValueError(
                "Cannot read this video. Please upload a supported MP4 video."
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            raise ValueError("The video has no readable frames.")

        duration = frame_count / fps

        if duration > MAX_VIDEO_SECONDS:
            raise ValueError("Video duration must not exceed 120 seconds.")

        sample_count = min(MAX_FRAMES, frame_count)
        frame_indices = sorted(
            set(
                int(i * frame_count / sample_count)
                for i in range(sample_count)
            )
        )

        frame_results = []

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()

            if not success:
                continue

            height, width = frame.shape[:2]

            if width * height > 12_000_000:
                scale = (12_000_000 / (width * height)) ** 0.5
                frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                )

            success, encoded = cv2.imencode(".jpg", frame)

            if not success:
                continue

            image_result = analyze_image(encoded.tobytes())

            frame_results.append({
                "frame_number": frame_index,
                "timestamp_seconds": round(frame_index / fps, 2),
                "prediction": image_result["prediction"],
                "confidence": image_result["confidence"],
                "scores": image_result["scores"],
            })

        if not frame_results:
            raise ValueError("Could not extract analyzable frames from the video.")

        fake_scores = [
            next(
                (
                    score["score_percent"]
                    for score in result["scores"]
                    if score["label"].strip().lower() == "fake"
                ),
                0.0,
            )
            for result in frame_results
        ]

        average_fake_score = sum(fake_scores) / len(fake_scores)

        return {
            "duration_seconds": round(duration, 2),
            "frames_analyzed": len(frame_results),
            "average_fake_frame_score": round(average_fake_score, 2),
            "frame_results": frame_results,
            "model": "dima806/deepfake_vs_real_image_detection",
            "note": (
                "This is a sampled-frame screening result using an image model, "
                "not a dedicated video deepfake detector. It may miss temporal "
                "manipulation and cannot prove authenticity."
            ),
        }

    finally:
        if capture is not None:
            capture.release()

        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
