# Backend Integration Guide for Master Track Generation

## Overview
This guide explains how to integrate the master/summed track generation into your backend API.

## Changes Made

### 1. Frontend Changes (doseedo2.html)

#### Track Positioning Fix (Lines 1487-1494)
Fixed CSS to ensure all track lists have proper positioning:
```css
#download-links > li,
#download-links2 > li,
#download-links3 > li {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
}
```
This prevents tracks from appearing offset or overlapping with the container.

#### Master Track Priority (Lines 3775-3793)
Modified `handleACEStepResults()` to ensure the master/summed track is always added first:
```javascript
// Add main/mixed output FIRST if available (will be at index 0)
if (result.mainAudio) {
  filePaths.push(result.mainAudio);
  sceneChanges.push(0); // Start at time 0
}
```

The master track will:
- Always be at index 0
- Always appear at the top when the tracklist is collapsed
- Have the highest z-index for proper visual layering

### 2. Backend Integration

#### Required Python Package
Install pydub for audio processing:
```bash
pip install pydub numpy
```

You'll also need ffmpeg installed on your system:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

#### Using genfrominterface.py

The `genfrominterface.py` module provides two functions:

##### Simple Summing
```python
from genfrominterface import sum_audio_tracks

# Sum all generated tracks
track_paths = ['/path/to/track1.wav', '/path/to/track2.wav', '/path/to/track3.wav']
master_path = '/path/to/output/master.wav'

master_track = sum_audio_tracks(track_paths, master_path, normalize=True)
# Returns: '/path/to/output/master.wav'
```

##### Advanced Mixing with Levels
```python
from genfrominterface import mix_with_levels

# Mix tracks with individual volume control
track_paths = ['/path/to/track1.wav', '/path/to/track2.wav']
levels = [1.0, 0.7]  # Second track at 70% volume
master_path = '/path/to/output/master.wav'

master_track = mix_with_levels(track_paths, master_path, levels, normalize=True)
```

### 3. API Response Format

Your backend API should return results in this format:

```python
{
    "status": "completed",
    "result": {
        "mainAudio": "/download/task_123/master.wav",  # Master/summed track
        "file_paths": [
            "/download/task_123/voice_1.wav",
            "/download/task_123/voice_2.wav",
            "/download/task_123/voice_3.wav"
        ],
        "duration": 10.5,
        # ... other fields
    }
}
```

**Important:** The `mainAudio` field is now read FIRST by the frontend and will be placed at track index 0.

### 4. Example Backend Integration

Here's how to integrate into your FastAPI/Flask backend:

```python
from fastapi import FastAPI, File, UploadFile
from genfrominterface import sum_audio_tracks
import os

app = FastAPI()

@app.post("/api/generate-ace-step")
async def generate_ace_step(params: dict):
    # ... your existing generation logic ...

    # After generating all individual tracks:
    voice_tracks = [
        "/download/task_123/voice_1.wav",
        "/download/task_123/voice_2.wav",
        "/download/task_123/voice_3.wav"
    ]

    # Generate master/summed track
    master_output = f"/download/task_123/master.wav"
    master_track = sum_audio_tracks(voice_tracks, master_output, normalize=True)

    # Return response with mainAudio field
    return {
        "status": "completed",
        "result": {
            "mainAudio": master_track,  # This will be track 0
            "file_paths": voice_tracks,  # These will be tracks 1, 2, 3, etc.
            "duration": 10.5
        }
    }
```

### 5. Track Order in UI

With these changes:
- **Track 0 (First):** Master/summed track (from `result.mainAudio`)
- **Track 1-N:** Individual voice/instrument tracks (from `result.file_paths`)

When collapsed:
- All tracks stack at position `top: 0`
- Track 0 (master) has the highest z-index and appears on top
- This gives users a quick preview of the final mix

When expanded:
- Track 0 appears at the top
- Individual tracks appear below in order

### 6. Command Line Usage

You can also use genfrominterface.py from the command line:

```bash
python genfrominterface.py master.wav track1.wav track2.wav track3.wav
```

## Testing

1. Generate audio with your backend
2. Verify the response includes `mainAudio` field
3. Check that the master track appears first in the UI
4. Test collapsed view - master track should be on top
5. Test expanded view - all tracks should be visible

## Troubleshooting

### Tracks still overlapping?
- Clear browser cache and reload
- Check browser console for positioning errors
- Verify CSS changes are applied

### Master track not appearing first?
- Verify backend returns `mainAudio` field
- Check browser console for `handleACEStepResults` logs
- Ensure the API response format matches the documented structure

### Audio quality issues?
- Adjust normalization in `sum_audio_tracks(normalize=True)`
- Try `mix_with_levels()` for more control
- Check input track quality

## Next Steps

1. Update your backend API to generate master tracks
2. Test with a small number of tracks first
3. Monitor for clipping/distortion
4. Adjust normalization settings as needed
