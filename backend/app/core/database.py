import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATABASE_PATH = Path(__file__).resolve().parent.parent / "analysis_history.db"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                prediction TEXT NOT NULL,
                score REAL,
                timestamp TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_history_timestamp ON analysis_history (timestamp DESC)"
        )


def save_analysis_history(
    filename: str,
    media_type: str,
    prediction: str,
    score: float | None = None,
) -> None:
    safe_score = None
    if score is not None:
        try:
            safe_score = float(score)
        except (TypeError, ValueError):
            safe_score = None

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO analysis_history (filename, media_type, prediction, score, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename or "unknown-file",
                (media_type or "unknown").lower(),
                prediction or "UNKNOWN",
                safe_score,
                timestamp,
            ),
        )


def list_analysis_history(limit: int = 20, offset: int = 0) -> dict[str, Any]:
    normalized_limit = max(1, min(int(limit), 200))
    normalized_offset = max(0, int(offset))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, media_type, prediction, score, timestamp
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (normalized_limit, normalized_offset),
        ).fetchall()
        total = connection.execute(
            "SELECT COUNT(*) FROM analysis_history"
        ).fetchone()[0]

    return {
        "items": [
            {
                "id": row["id"],
                "filename": row["filename"],
                "media_type": row["media_type"],
                "prediction": row["prediction"],
                "score": row["score"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ],
        "total": total,
        "limit": normalized_limit,
        "offset": normalized_offset,
    }


def clear_analysis_history() -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM analysis_history")
