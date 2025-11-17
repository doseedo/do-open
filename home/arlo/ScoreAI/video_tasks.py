import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
import sys
import subprocess
import tempfile
import colorsys
import concurrent.futures
from collections import defaultdict
import torch
import cv2
import numpy as np

import whisper

from celery_config import celery_app
from google.cloud import videointelligence, vision

# Add GCS storage support
sys.path.append('/home/arlo/Data')
from gcs_storage import upload_to_gcs, get_gcs_url

# Initialize Whisper model once
# whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")



import time
from google.api_core.exceptions import GoogleAPICallError

def safe_annotate_image(vision_client, image, features, retries=3):
    for attempt in range(retries):
        try:
            return vision_client.annotate_image({'image': image, 'features': features})
        except GoogleAPICallError as e:
            print(f"Retry {attempt + 1}/3 due to: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Failed to annotate image after retries")




ANALYSIS_FEATURES = {
    "objects": {
        "min_confidence": 0.75,
        "custom_categories": {
            "vehicle": ["car", "truck", "motorcycle", "bicycle"],
            "animal": ["dog", "cat", "bird", "wildlife"],
            "electronics": ["phone", "computer", "television"]
        }
    },
    "colors": {
        "min_coverage": 0.1,
        "max_colors": 5
    },
    "text": {
        "min_size": 0.01
    }
}

def get_video_duration(file_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        return float(result.stdout.decode().strip())
    except Exception as e:
        print(f"[ERROR] Duration detection failed: {e}")
        return 0

def analyze_frame(vision_client, image_content):
    image = vision.Image(content=image_content)
    features = [
        {'type': vision.Feature.Type.OBJECT_LOCALIZATION},
        {'type': vision.Feature.Type.IMAGE_PROPERTIES},
        {'type': vision.Feature.Type.TEXT_DETECTION},
        {'type': vision.Feature.Type.LABEL_DETECTION, 'max_results': 20},
        {'type': vision.Feature.Type.WEB_DETECTION},
        {'type': vision.Feature.Type.SAFE_SEARCH_DETECTION},
        {'type': vision.Feature.Type.LANDMARK_DETECTION},
        {'type': vision.Feature.Type.LOGO_DETECTION}
    ]
    response = safe_annotate_image(vision_client, image, features)

    analysis = {
        "objects": defaultdict(list),
        "colors": [],
        "text": [],
        "labels": [],
        "safe_search": {},
        "web_entities": []
    }

    for obj in response.localized_object_annotations:
        if obj.score >= ANALYSIS_FEATURES["objects"]["min_confidence"]:
            item = {
                "name": obj.name,
                "confidence": round(obj.score, 2),
                "bounding_box": [(v.x, v.y) for v in obj.bounding_poly.normalized_vertices]
            }
            analysis["objects"][obj.name.lower()].append(item)

    for color in response.image_properties_annotation.dominant_colors.colors:
        if color.pixel_fraction >= ANALYSIS_FEATURES["colors"]["min_coverage"]:
            rgb = color.color
            h, s, v = colorsys.rgb_to_hsv(rgb.red/255, rgb.green/255, rgb.blue/255)
            analysis["colors"].append({
                "hsv": (round(h*360, 1), round(s, 2), round(v, 2)),
                "coverage": round(color.pixel_fraction, 2)
            })
    analysis["colors"] = sorted(analysis["colors"], key=lambda x: -x["coverage"])[:ANALYSIS_FEATURES["colors"]["max_colors"]]

    for text in response.text_annotations:
        vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
        height = abs(vertices[3][1] - vertices[0][1])
        if height >= ANALYSIS_FEATURES["text"]["min_size"]:
            analysis["text"].append({
                "content": text.description,
                "confidence": round(text.confidence, 2),
                "bounding_box": vertices
            })

    for label in response.label_annotations:
        if label.score >= ANALYSIS_FEATURES["objects"]["min_confidence"]:
            analysis["labels"].append({"description": label.description, "confidence": round(label.score, 2)})

    if response.web_detection.web_entities:
        for entity in response.web_detection.web_entities:
            if entity.description and entity.score >= 0.5:
                analysis["web_entities"].append({"description": entity.description, "confidence": round(entity.score, 2)})

    if response.safe_search_annotation:
        safe = response.safe_search_annotation
        analysis["safe_search"] = {
            "violence": safe.violence.name,
            "racy": safe.racy.name,
            "adult": safe.adult.name,
            "spoof": safe.spoof.name,
            "medical": safe.medical.name
        }

    return analysis

def get_whisper_model():
    if not hasattr(get_whisper_model, "_model"):
        print("🎧 Loading Whisper model inside task subprocess")
        get_whisper_model._model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    return get_whisper_model._model

def extract_transcript(file_path, start, end):
    try:
        segment_path = f"/tmp/segment_{start:.2f}_{end:.2f}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", file_path, "-ss", str(start), "-to", str(end),
            "-ar", "16000", "-ac", "1", "-f", "wav", segment_path
        ], check=True, capture_output=True)

        model = get_whisper_model()  # ✅ Safe inside worker subprocess
        result = model.transcribe(segment_path)
        return result["text"]

    except Exception as e:
        print(f"[Whisper Error] {e}")
        return ""

def estimate_motion(prev_path, next_path):
    try:
        prev_img = cv2.imread(prev_path)
        next_img = cv2.imread(next_path)

        if prev_img is None or next_img is None:
            return 0.0

        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(np.mean(magnitude))
    except Exception as e:
        print(f"[Motion Error] {e}")
        return 0.0




def process_scene(args):
    vision_client, file_path, tmpdir, merged_start, end, scene_idx, use_whisper = args
    duration = end - merged_start
    scene = {
        "scene_index": scene_idx,
        "start": merged_start,
        "end": end,
        "key_frames": [],
        "dominant_colors": defaultdict(float),
        "objects": [],
        "detected_text": [],
        "object_tracking": defaultdict(list),
        "labels": [],
        "transcript": ""
    }

    num_frames = max(2, min(6, int(duration // 2)))
    timestamps = [merged_start + duration * i / num_frames for i in range(num_frames)]
    motion_scores = []
    frame_paths = []

    for t in timestamps:
        frame_path = os.path.join(tmpdir, f'scene_{scene_idx}_frame_{t:.2f}.jpg')
        frame_paths.append(frame_path)

        subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", file_path,
                        "-vframes", "1", "-q:v", "2", frame_path], check=True, capture_output=True)
        
        with open(frame_path, 'rb') as f:
            analysis = analyze_frame(vision_client, f.read())

        scene["key_frames"].append({"timestamp": t, "analysis": analysis})

        for color in analysis["colors"]:
            scene["dominant_colors"][tuple(color["hsv"])] += color["coverage"]
        for obj_name in analysis["objects"]:
            scene["objects"].append(obj_name)
            scene["object_tracking"][obj_name].append(t)
        for txt in analysis["text"]:
            if txt["content"] not in scene["detected_text"]:
                scene["detected_text"].append(txt["content"])
        for label in analysis["labels"]:
            if label["description"] not in scene["labels"]:
                scene["labels"].append(label["description"])

    # 🧠 Estimate motion score between consecutive frames
    for i in range(1, len(frame_paths)):
        score = estimate_motion(frame_paths[i - 1], frame_paths[i])
        motion_scores.append(score)

    scene["motion_score"] = float(np.mean(motion_scores)) if motion_scores else 0.0

    motion_score = scene["motion_score"]
    if motion_score < 0.2:
        scene["motion_class"] = "low"
    elif motion_score < 0.7:
        scene["motion_class"] = "medium"
    else:
        scene["motion_class"] = "high"

    scene["dominant_colors"] = sorted(scene["dominant_colors"].items(), key=lambda x: -x[1])[:3]

    # Only run Whisper transcription if enabled
    if use_whisper:
        scene["transcript"] = extract_transcript(file_path, merged_start, end)
        print(f"🎤 Scene {scene_idx} Transcript: {scene['transcript']}")
    else:
        scene["transcript"] = ""
        print(f"⏩ Scene {scene_idx}: Whisper transcription disabled")

    return scene


@celery_app.task(name='video_tasks.process_video')
def process_video(file_path, use_whisper=False):
    try:
        print(f"🎬 Processing video: {file_path}")
        print(f"🎤 Whisper transcription: {'enabled' if use_whisper else 'disabled'}")

        video_client = videointelligence.VideoIntelligenceServiceClient()
        operation = video_client.annotate_video({
            'features': [videointelligence.Feature.SHOT_CHANGE_DETECTION],
            'input_content': open(file_path, 'rb').read()
        })
        shot_changes = [shot.start_time_offset.total_seconds()
                        for shot in operation.result().annotation_results[0].shot_annotations]

        scene_changes = [0.0] + shot_changes + [get_video_duration(file_path)]
        vision_client = vision.ImageAnnotatorClient()

        # Extract audio from video and save as M4A (AAC)
        video_duration = get_video_duration(file_path)

        # Store audio in the same directory as the video file
        video_dir = os.path.dirname(file_path)
        video_base = os.path.splitext(os.path.basename(file_path))[0]
        audio_filename = f"audio_{video_base}.m4a"
        audio_path = os.path.join(video_dir, audio_filename)

        try:
            result = subprocess.run([
                "ffmpeg", "-y", "-i", file_path,
                "-vn",  # No video
                "-acodec", "copy",  # Copy audio stream without re-encoding
                audio_path
            ], check=True, capture_output=True, text=True)

            # Construct URL using the relative path (for backward compatibility)
            audio_relative_path = os.path.join(video_dir, audio_filename)
            audio_url = f"https://doseedo.com/{audio_relative_path}"
            print(f"🎵 Audio extracted to: {audio_url}")

            # Upload to GCS in background (non-blocking)
            def upload_to_gcs_background():
                try:
                    gcs_path = upload_to_gcs(audio_path, prefix="audiofiles", user_id=None, make_public=True)
                    gcs_url = get_gcs_url(gcs_path)
                    print(f"✅ Background GCS upload complete: {gcs_url}")
                except Exception as e:
                    print(f"⚠️  Background GCS upload failed: {e}")
                    print(f"   Local file remains at: {audio_path}")

            # Start upload in background thread
            import threading
            upload_thread = threading.Thread(target=upload_to_gcs_background, daemon=True)
            upload_thread.start()
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Audio extraction failed with exit code {e.returncode}")
            print(f"[FFMPEG STDERR] {e.stderr}")
            print(f"[FFMPEG STDOUT] {e.stdout}")
            audio_url = None
        except Exception as e:
            print(f"[ERROR] Audio extraction failed: {e}")
            audio_url = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = []
            scene_idx = 0
            for i in range(1, len(scene_changes)):
                start = scene_changes[i - 1]
                end = scene_changes[i]
                if (end - start) < 3.0:
                    continue
                tasks.append((vision_client, file_path, tmpdir, start, end, scene_idx, use_whisper))
                scene_idx += 1

            with concurrent.futures.ThreadPoolExecutor() as executor:
                scene_data = list(executor.map(process_scene, tasks))

        return {
            "scene_changes": scene_changes,
            "scene_data": scene_data,
            "duration": video_duration,
            "audio_url": audio_url
        }

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        raise