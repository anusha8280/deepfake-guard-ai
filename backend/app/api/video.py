
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.history import save_successful_result
from app.services.video_detector import analyze_video


router = APIRouter(
    prefix="/api/analysis",
    tags=["Video Analysis"],
)

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "application/mp4",
    "video/x-m4v",
}

MAX_VIDEO_SIZE = 50 * 1024 * 1024


@router.post("/video")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    filename = file.filename or ""

    if not filename.lower().endswith((".mp4", ".m4v")):
        raise HTTPException(
            status_code=400,
            detail="Upload an MP4 or M4V video.",
        )

    if file.content_type and file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video content type. Upload an MP4 video.",
        )

    try:
        video_bytes = await file.read(MAX_VIDEO_SIZE + 1)

        if not video_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty.",
            )

        if len(video_bytes) > MAX_VIDEO_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Video size must not exceed 50 MB.",
            )

        try:
            result = await run_in_threadpool(analyze_video, video_bytes)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            print(f"Video analysis error: {exc}")
            raise HTTPException(
                status_code=500,
                detail="Video analysis failed. Check the backend terminal.",
            ) from exc

        average_score = result.get("average_fake_frame_score")
        prediction = "FAKE" if float(average_score or 0) >= 50 else "REAL"
        save_successful_result(
            filename=filename,
            media_type="video",
            prediction=prediction,
            score=average_score,
        )

        return {
            "status": "success",
            "filename": filename,
            **result,
        }

    finally:
        await file.close()
