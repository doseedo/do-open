# ACE-Step Web Interface Integration

This setup provides a web interface (doseedo2.html + javascript2.js) to call the ACE-Step model via FastAPI endpoints added to genfrominterface.py.

## Files Created/Modified

1. **doseedo2.html** - Web interface with ACE-Step model parameter controls
   - Location: `/home/arlo/doseedo2.html`
   - Features: UI sliders for all ACE-Step parameters (steps, seed, adapter_scale, cfg_weight, gains, etc.)

2. **javascript2.js** - Frontend JavaScript that calls the ACE-Step API
   - Location: `/home/arlo/javascript2.js`
   - Calls: `http://localhost:8000/generate`, `/task/{task_id}`, `/download/{process_id}/{filename}`

3. **genfrominterface.py** - Modified to include FastAPI endpoints
   - Location: `/home/arlo/Data/genfrominterface.py`
   - Added: FastAPI app, Celery tasks, and REST endpoints

4. **run_fastapi_server.py** - Standalone script to run the FastAPI server
   - Location: `/home/arlo/Data/run_fastapi_server.py`
   - Purpose: Initializes model and starts FastAPI server on port 8000

## Setup & Usage

### Prerequisites

1. Install required packages:
```bash
pip install fastapi uvicorn celery redis
```

2. Start Redis (for Celery backend):
```bash
redis-server
```

3. Start RabbitMQ (for Celery broker):
```bash
# Ubuntu/Debian
sudo systemctl start rabbitmq-server
```

### Starting the System

#### Step 1: Start Celery Worker

In a terminal, run:
```bash
cd /home/arlo/Data
celery -A genfrominterface.celery_app worker --loglevel=info
```

#### Step 2: Start FastAPI Server

In another terminal, run:
```bash
python /home/arlo/Data/run_fastapi_server.py \
  --checkpoint /mnt/msdd/exps/logs_v2/lightning_logs/2025-09-06_16-12-31_all_groups_ft_v3_capivotpitch_ctrl/checkpoints/last.ckpt \
  --checkpoint_dir /mnt/msdd/checkpoints \
  --manifest /path/to/your/manifest.json \
  --port 8000
```

Replace the paths with your actual checkpoint and manifest paths.

#### Step 3: Open the Web Interface

Open `doseedo2.html` in your web browser:
```bash
# Option 1: Direct file open
firefox /home/arlo/doseedo2.html

# Option 2: Serve via simple HTTP server
cd /home/arlo
python -m http.server 8080
# Then open http://localhost:8080/doseedo2.html
```

## API Endpoints

The following endpoints are now available:

### POST /generate
Generate audio using ACE-Step model

**Form Parameters:**
- `description` (str): Text description of the audio
- `duration` (float): Duration in seconds (default: 3.0)
- `steps` (int): Number of diffusion steps (default: 50)
- `seed` (int): Random seed (0 for random, default: 0)
- `adapter_scale` (float): Adapter scale (default: 1.0)
- `cfg_weight` (float): CFG weight (default: 3.0)
- `instrument_strength` (float): Instrument conditioning strength (default: 1.0)
- `noise_level` (float): Noise level 0-1 (default: 1.0)
- `piano_roll_gain` (float): Piano roll gain (default: 1.0)
- `amp_gain` (float): Amplitude gain (default: 1.0)
- `rframe_gain` (float): RFrame gain (default: 1.0)
- `rbend_gain` (float): RBend gain (default: 1.0)
- `encodec_gain` (float): EnCodec gain (default: 1.0)
- `pitch_fidelity_boost` (float): Pitch fidelity boost (default: 1.0)
- `onset_guidance_boost` (float): Onset guidance boost (default: 2.0)
- `pitch_snap_strength` (float): Pitch snap strength (default: 0.5)
- `audio_file` (File): Optional audio/MIDI file for conditioning

**Returns:**
```json
{"task_id": "uuid-string"}
```

### GET /task/{task_id}
Check generation task status

**Returns:**
```json
{
  "status": "completed",
  "result": ["/download/process_id/0.wav"]
}
```

### GET /download/{process_id}/{filename}
Download generated audio file

## Model Parameters

### Core Generation
- **Steps**: Number of diffusion steps (10-200, default: 50)
  - Higher = better quality but slower
- **Seed**: Random seed for reproducibility (0 = random)
- **Adapter Scale**: Conditioning adapter strength (0-5, default: 1.0)
- **CFG Weight**: Classifier-free guidance weight (1-6, default: 3.0)
  - Higher = stronger instrument conditioning

### Conditioning Gains
- **Piano Roll Gain**: Piano roll conditioning strength (0-4, default: 1.0)
- **Amp Gain**: Amplitude envelope strength (0-2, default: 1.0)
- **RFrame Gain**: Rhythmic frame strength (0-2, default: 1.0)
- **RBend Gain**: Pitch bend strength (0-2, default: 1.0)
- **EnCodec Gain**: Audio token strength (0-2, default: 1.0)

### Advanced Controls
- **Instrument Strength**: Overall instrument conditioning (0-5, default: 1.0)
- **Noise Level**: Mix between pure noise (1.0) and conditioning (0.0)
- **Pitch Fidelity Boost**: Pitch accuracy boost (0-3, default: 1.0)
- **Onset Guidance Boost**: Note onset sharpness (0-5, default: 2.0)
- **Pitch Snap Strength**: Quantize to semitones (0-1, default: 0.5)

## Differences from ac.py

### ac.py (AudioCraft/MusicGen)
- Uses Meta's AudioCraft models (MusicGen, AudioGen)
- Text-to-music generation
- Simpler controls (prompt, duration, melody conditioning)

### genfrominterface.py (ACE-Step)
- Uses custom ACE-Step diffusion model
- Audio-to-audio transformation with extensive conditioning
- Fine-grained control over pitch, rhythm, amplitude, and timbre
- Requires audio/MIDI input for conditioning

## Workflow

1. **Upload a video** (optional) - Analyzes video for labels
2. **Adjust model parameters** - Click the settings gear icon
3. **Upload audio/MIDI reference** - Provides conditioning for generation
4. **Click "Generate Music"** - Starts generation task
5. **Download results** - Appears in download-links section

## Troubleshooting

### "Model not loaded" error
- Make sure you started run_fastapi_server.py with correct checkpoint paths

### "Connection refused" at localhost:8000
- Verify FastAPI server is running
- Check firewall settings

### "Task failed" or Celery errors
- Ensure Celery worker is running
- Check Celery logs for detailed errors
- Verify Redis and RabbitMQ are running

### Audio generation is slow
- Lower the `steps` parameter (try 20-30 for faster results)
- Ensure GPU is being used (check for CUDA availability)

## Notes

- The FastAPI server runs on port 8000 by default
- The Gradio interface (if running separately) uses port 7860
- Generated audio files are saved to `/home/arlo/ScoreAI/audiofiles/ace_step_output_*`
- Celery tasks have a 15-minute soft timeout and 20-minute hard timeout

## Example Usage

```bash
# Terminal 1: Start Celery worker
celery -A genfrominterface.celery_app worker --loglevel=info

# Terminal 2: Start FastAPI server
python /home/arlo/Data/run_fastapi_server.py \
  --checkpoint_dir /mnt/msdd/checkpoints \
  --manifest /path/to/manifest.json

# Terminal 3 (optional): Serve HTML
cd /home/arlo
python -m http.server 8080

# Browser: Open http://localhost:8080/doseedo2.html
```

## Contact

For issues or questions, please check the logs in each terminal window.
