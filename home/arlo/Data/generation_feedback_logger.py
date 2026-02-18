"""
Generation Feedback Logger
Logs generation parameters with user feedback (like/dislike)
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Literal

# Log file location
LOG_DIR = Path("/home/arlo/Data/generation_logs")
LOG_FILE = LOG_DIR / "generation_feedback.jsonl"

def ensure_log_dir():
    """Create log directory if it doesn't exist"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_generation_feedback(
    params: Dict[str, Any],
    feedback: Literal["like", "dislike"],
    task_id: str = None,
    output_files: list = None,
    user_notes: str = None
) -> Dict[str, Any]:
    """
    Log generation parameters with user feedback

    Args:
        params: Generation parameters dictionary
        feedback: User feedback ("like" or "dislike")
        task_id: Optional Celery task ID
        output_files: Optional list of generated file paths
        user_notes: Optional user notes/comments

    Returns:
        Dictionary with log entry details
    """
    ensure_log_dir()

    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback,
        "task_id": task_id,
        "params": params,
        "output_files": output_files or [],
        "user_notes": user_notes,
    }

    # Append to JSONL file (one JSON object per line)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"✅ Generation feedback logged: {feedback} ({LOG_FILE})")

    return log_entry

def get_feedback_stats() -> Dict[str, Any]:
    """
    Get statistics about logged feedback

    Returns:
        Dictionary with like/dislike counts and percentages
    """
    if not LOG_FILE.exists():
        return {
            "total": 0,
            "likes": 0,
            "dislikes": 0,
            "like_percentage": 0.0
        }

    likes = 0
    dislikes = 0

    with open(LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("feedback") == "like":
                    likes += 1
                elif entry.get("feedback") == "dislike":
                    dislikes += 1

    total = likes + dislikes
    like_percentage = (likes / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "likes": likes,
        "dislikes": dislikes,
        "like_percentage": round(like_percentage, 2)
    }

def get_top_liked_params(limit: int = 10) -> list:
    """
    Get most common parameters from liked generations

    Args:
        limit: Maximum number of results to return

    Returns:
        List of parameter sets from liked generations
    """
    if not LOG_FILE.exists():
        return []

    liked_params = []

    with open(LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("feedback") == "like":
                    liked_params.append({
                        "timestamp": entry["timestamp"],
                        "params": entry["params"],
                        "task_id": entry.get("task_id"),
                        "user_notes": entry.get("user_notes")
                    })

    # Return most recent liked params
    return sorted(liked_params, key=lambda x: x["timestamp"], reverse=True)[:limit]

def analyze_parameter_correlations() -> Dict[str, Any]:
    """
    Analyze which parameter values correlate with likes vs dislikes

    Returns:
        Dictionary with parameter value distributions for likes/dislikes
    """
    if not LOG_FILE.exists():
        return {}

    liked_params = []
    disliked_params = []

    with open(LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                params = entry.get("params", {})

                if entry.get("feedback") == "like":
                    liked_params.append(params)
                elif entry.get("feedback") == "dislike":
                    disliked_params.append(params)

    # Analyze key parameters
    analysis = {
        "likes_count": len(liked_params),
        "dislikes_count": len(disliked_params),
        "parameter_analysis": {}
    }

    # Parameters to analyze
    key_params = [
        "steps", "cfgWeight", "noiseLevel", "instrumentStrength",
        "pianoRollGain", "ampGain", "encodecGain", "pitchFidelityBoost",
        "instrumentGroup", "instrumentSubgroup", "extractFormats"
    ]

    for param in key_params:
        if liked_params:
            liked_values = [p.get(param) for p in liked_params if param in p]
            if liked_values:
                analysis["parameter_analysis"][param] = {
                    "liked_avg": sum(v for v in liked_values if isinstance(v, (int, float))) / len(liked_values) if any(isinstance(v, (int, float)) for v in liked_values) else None,
                    "liked_common": max(set(map(str, liked_values)), key=lambda x: list(map(str, liked_values)).count(x))
                }

    return analysis
