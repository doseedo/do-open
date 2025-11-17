
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
import subprocess



from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from celery.result import AsyncResult
from fastapi.staticfiles import StaticFiles
from video_tasks import process_video
from audio_tasks import export_video_task, celery_app

import os
import uuid
import shutil
import threading
import io

from fastapi.responses import FileResponse

import magic
# from clamd import ClamdUnixSocket
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

# --- App Setup ---

app = FastAPI()

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: HTTPException(status_code=429, detail="Rate limit exceeded"))
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static folders ---

AUDIO_FILES_DIR = "/mnt/models/audiofiles"
STEM_AUDIO_FILES_DIR = "/home/arlo/ScoreAI/stem_audio_files"
TEMP_VIDEO_DIR = "/mnt/models/temp_videos"
TEMP_EXPORT_DIR = "/mnt/models/temp_exports"

os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
os.makedirs(TEMP_EXPORT_DIR, exist_ok=True)

app.mount("/media/audio", StaticFiles(directory=AUDIO_FILES_DIR), name="audio")
app.mount("/temp_exports", StaticFiles(directory=TEMP_EXPORT_DIR), name="temp_exports")

# --- Constants ---
MAX_UPLOAD_SIZE_MB = 100
VALID_VIDEO_MIME_TYPES = {"video/mp4", "video/webm", "video/ogg", "video/quicktime"}

# --- Models ---

class AudioVideoData(BaseModel):
    videoData: str
    audioData: str

# --- Routes ---

@app.post("/uploadvideo/")
@limiter.limit("10/minute")
async def upload_video(request: Request, file: UploadFile = File(...), use_whisper: str = Form("false")):
    contents = await file.read()

    # Size check
    if len(contents) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 100MB allowed.")

    # MIME type check
    mime = magic.Magic(mime=True).from_buffer(contents)
    if mime not in VALID_VIDEO_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {mime}")

    # Save the validated file
    unique_id = str(uuid.uuid4())
    temp_file_path = os.path.join(TEMP_VIDEO_DIR, f"temp_video_{unique_id}.mp4")
    with open(temp_file_path, "wb") as f:
        f.write(contents)

    # Convert use_whisper string to boolean
    use_whisper_bool = use_whisper.lower() in ['true', '1', 'yes']
    print(f"📹 Video upload: use_whisper={use_whisper_bool}")

    task = process_video.delay(temp_file_path, use_whisper_bool)
    return {"task_id": task.id, "video_id": unique_id}

@app.post("/export/")
async def export_video(
    video: UploadFile = File(...),
    audioTracks: str = Form(...),
    file_0: UploadFile = File(None),
    file_1: UploadFile = File(None),
    file_2: UploadFile = File(None),
    file_3: UploadFile = File(None),
    file_4: UploadFile = File(None),
    file_5: UploadFile = File(None),
):
    try:
        temp_dir = os.path.join(TEMP_EXPORT_DIR, str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)

        video_path = os.path.join(temp_dir, "original.mp4")
        with open(video_path, "wb") as f:
            f.write(await video.read())

        uploaded_files = [file_0, file_1, file_2, file_3, file_4, file_5]

        tracks_metadata = json.loads(audioTracks)

        for idx, audio_file in enumerate(uploaded_files):
            if audio_file is not None:
                temp_audio_path = os.path.join(temp_dir, f"temp_track_{idx}.wav")
                with open(temp_audio_path, "wb") as f:
                    f.write(await audio_file.read())

                final_audio_path = os.path.join(temp_dir, f"track_{idx}.wav")

                if idx < len(tracks_metadata):
                    metadata = tracks_metadata[idx]
                    duration = metadata.get("duration")
                    start_time = metadata.get("start", 0)  # start time in seconds
                    metadata["filename"] = f"track_{idx}.wav"
                else:
                    duration = None
                    start_time = 0

                ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_audio_path]

                if start_time < 0:
                    trim_start = abs(start_time)
                    print(f"⛏ Trimming {trim_start:.3f}s from track {idx} (starts before 0)")
                    ffmpeg_cmd += ["-ss", str(trim_start)]

                    # Also reduce the duration if it's defined
                    if duration:
                        new_duration = max(0, duration - trim_start)
                        metadata["duration"] = new_duration
                        ffmpeg_cmd += ["-t", str(new_duration)]

                elif duration:
                    ffmpeg_cmd += ["-t", str(duration)]

                ffmpeg_cmd += [final_audio_path]
                subprocess.run(ffmpeg_cmd, check=True)
                os.remove(temp_audio_path)

                task = export_video_task.delay(
                    video_path=video_path,
                    audio_tracks=tracks_metadata,
                    output_path=os.path.join(temp_dir, "final_output.mp4")
                )

                return {"task_id": str(task.id), "temp_dir": temp_dir}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export initialization failed: {str(e)}")

@app.get("/export/status/{task_id}")
async def export_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": task_result.state}

@app.get("/export-result/{temp_dir}")
async def get_export_result(temp_dir: str):
    try:
        file_path = os.path.join(temp_dir, "final_output.mp4")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Exported video not found.")

        threading.Timer(300, shutil.rmtree, [temp_dir]).start()

        return FileResponse(file_path, media_type="video/mp4", filename="video_with_audio.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve export result: {str(e)}")

@app.post("/exportAudio/")
async def export_audio_only(video: UploadFile = File(...), audio: UploadFile = File(...)):
    try:
        video_path = os.path.join(TEMP_VIDEO_DIR, f"{uuid.uuid4()}.mp4")
        audio_path = os.path.join(TEMP_VIDEO_DIR, f"{uuid.uuid4()}.mp3")

        with open(video_path, "wb") as f:
            f.write(await video.read())
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        task = export_audio_to_video.delay(audio_path, video_path)
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stems/")
async def split_stems(filePath: str = Form(...)):
    full_path = os.path.join(AUDIO_FILES_DIR, filePath)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Original audio file not found.")

    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(STEM_AUDIO_FILES_DIR, f"stem_output_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)

    task = split_audio_stems.delay(full_path, output_dir)
    return {"task_id": task.id, "audio_id": unique_id}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": result.status, "result": result.result}