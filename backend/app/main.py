
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.audio import router as audio_router
from app.api.history import router as history_router
from app.api.image import router as image_router
from app.api.video import router as video_router
from app.core.database import init_db


app = FastAPI(
    title="DeepfakeGuard AI API",
    description="AI-based image, video, and audio deepfake analysis API.",
    version="0.4.0",
)

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(image_router)
app.include_router(video_router)
app.include_router(audio_router)
app.include_router(history_router)


@app.get("/")
def home():
    return {
        "message": "DeepfakeGuard AI API is running",
        "status": "ok",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "application": "DeepfakeGuard AI",
        "version": "0.4.0",
        "detection_models": {
            "image": "configured",
            "video": "sampled_frame_baseline",
            "audio": "Vansh180/deepfake-audio-wav2vec2",
        },
    }
