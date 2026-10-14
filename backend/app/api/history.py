from fastapi import APIRouter, HTTPException, Query

from app.core.database import clear_analysis_history, list_analysis_history, save_analysis_history

router = APIRouter(prefix="/api", tags=["History"])


@router.get("/history")
def get_history(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        return list_analysis_history(limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to load analysis history.",
        ) from exc


@router.delete("/history/clear")
def clear_history():
    try:
        clear_analysis_history()
        return {"status": "success", "message": "Analysis history cleared."}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to clear analysis history.",
        ) from exc


def save_successful_result(
    filename: str,
    media_type: str,
    prediction: str,
    score: float | None = None,
) -> None:
    try:
        save_analysis_history(
            filename=filename,
            media_type=media_type,
            prediction=prediction,
            score=score,
        )
    except Exception as exc:
        print(f"History persistence failed: {exc}")
