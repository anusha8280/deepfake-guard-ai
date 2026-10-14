
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.history import save_successful_result
from app.services.audio_detector import analyze_audio


router = APIRouter(
    prefix="/api/analysis",
    tags=["Audio Analysis"],
)

MAX_AUDIO_SIZE = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


@router.post("/audio")
async def analyze_uploaded_audio(file: UploadFile = File(...)):
    filename = file.filename or "audio.wav"

    import os

    extension = os.path.splitext(filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a WAV, MP3, FLAC, or OGG audio file.",
        )

    try:
        audio_bytes = await file.read(MAX_AUDIO_SIZE + 1)

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Audio file size must not exceed 20 MB.",
            )

        try:
            result = await run_in_threadpool(
                analyze_audio,
                audio_bytes,
                filename,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            print(f"Audio analysis error: {exc}")
            raise HTTPException(
                status_code=500,
                detail=(
                    "Audio analysis failed. Check the backend terminal "
                    "for details."
                ),
            ) from exc

        prediction = str(result.get("prediction", "UNKNOWN")).strip()
        confidence = result.get("confidence")
        save_successful_result(
            filename=filename,
            media_type="audio",
            prediction=prediction,
            score=confidence,
        )

        return {
            "status": "success",
            "filename": filename,
            **result,
        }

    finally:
        await file.close()
