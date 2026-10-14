
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.history import save_successful_result
from app.services.image_detector import analyze_image

router = APIRouter(
    prefix="/api/analysis",
    tags=["Image Analysis"],
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024


@router.post("/image")
async def analyze_uploaded_image(file: UploadFile = File(...)):
    filename = file.filename or "uploaded-image"

    try:
        image_bytes = await file.read(MAX_IMAGE_SIZE + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB.",
            )

        try:
            result = await run_in_threadpool(
                analyze_image,
                image_bytes,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            print(f"Image analysis error: {exc}")
            raise HTTPException(
                status_code=500,
                detail="Image analysis failed. Check the backend terminal.",
            ) from exc

        prediction = str(result.get("prediction", "UNKNOWN")).strip()
        confidence = result.get("confidence")
        save_successful_result(
            filename=filename,
            media_type="image",
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
