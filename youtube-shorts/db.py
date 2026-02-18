import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT UNIQUE NOT NULL,
            youtube_id TEXT,
            title TEXT,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'success'
        )
    """)
    conn.commit()
    conn.close()


def is_uploaded(video_path: str) -> bool:
    conn = get_connection()
    cursor = conn.execute(
        "SELECT 1 FROM uploads WHERE video_path = ? AND status = 'success'",
        (video_path,),
    )
    result = cursor.fetchone() is not None
    conn.close()
    return result


def record_upload(video_path: str, youtube_id: str, title: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO uploads (video_path, youtube_id, title, uploaded_at, status) VALUES (?, ?, ?, ?, 'success')",
        (video_path, youtube_id, title, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
