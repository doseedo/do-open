#!/usr/bin/env python3
"""YouTube Shorts Auto-Uploader.

Scans a folder for video files with sidecar JSON metadata,
uploads up to `daily_limit` new videos to YouTube as Shorts,
and records uploads in a local SQLite database.
"""

import os
import sys
import json
import logging
import yaml
from pathlib import Path
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from auth import get_authenticated_service
from db import init_db, is_uploaded, record_upload

PROJECT_DIR = Path(__file__).parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_DIR / "upload.log"),
    ],
)
log = logging.getLogger(__name__)


def load_config():
    config_path = PROJECT_DIR / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_pending_videos(videos_folder: str, daily_limit: int) -> list[Path]:
    folder = Path(videos_folder)
    if not folder.exists():
        log.error("Videos folder does not exist: %s", folder)
        return []

    candidates = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in VIDEO_EXTENSIONS and not is_uploaded(str(f)):
            candidates.append(f)

    return candidates[:daily_limit]


def load_sidecar(video_path: Path) -> dict | None:
    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        log.warning("No sidecar JSON for %s — skipping", video_path.name)
        return None
    with open(json_path) as f:
        return json.load(f)


def ensure_shorts_tag(description: str) -> str:
    if "#shorts" not in description.lower():
        description = description.rstrip() + "\n#Shorts"
    return description


def upload_video(youtube, video_path: Path, metadata: dict, privacy: str) -> str | None:
    title = metadata.get("title", video_path.stem)
    description = ensure_shorts_tag(metadata.get("description", ""))
    tags = metadata.get("tags", [])
    category = metadata.get("category", "22")  # 22 = People & Blogs

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/*",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10 MB chunks
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    log.info("Uploading: %s (%s)", title, video_path.name)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info("  Upload progress: %d%%", int(status.progress() * 100))

    video_id = response["id"]
    log.info("Upload complete: https://youtube.com/shorts/%s", video_id)
    return video_id


def main():
    config = load_config()
    videos_folder = config["videos_folder"]
    daily_limit = config.get("daily_limit", 3)
    privacy = config.get("privacy_status", "public")

    init_db()

    pending = find_pending_videos(videos_folder, daily_limit)
    if not pending:
        log.info("No new videos to upload.")
        return

    log.info("Found %d video(s) to upload", len(pending))

    youtube = get_authenticated_service()

    uploaded = 0
    for video_path in pending:
        metadata = load_sidecar(video_path)
        if metadata is None:
            continue

        try:
            video_id = upload_video(youtube, video_path, metadata, privacy)
            if video_id:
                record_upload(str(video_path), video_id, metadata.get("title", video_path.stem))
                uploaded += 1
        except HttpError as e:
            log.error("YouTube API error uploading %s: %s", video_path.name, e)
        except Exception as e:
            log.error("Unexpected error uploading %s: %s", video_path.name, e)

    log.info("Done. Uploaded %d/%d video(s).", uploaded, len(pending))


if __name__ == "__main__":
    main()
