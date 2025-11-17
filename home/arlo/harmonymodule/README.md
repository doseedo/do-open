# Dø - AI Music Generation Platform

Complete production stack for ACE-Step AI music generation, featuring a professional DAW interface and inference backend.

## Repository Structure

```
Do/
├── frontend/              # React DAW application
│   ├── src/
│   │   ├── components/   # UI components (DAW, MIDI, Generation)
│   │   ├── hooks/        # Custom React hooks (playback, waveform, generation)
│   │   ├── services/     # API clients (generation, GCS, chat)
│   │   └── context/      # Global state management
│   └── package.json
├── inference/             # Backend inference stack
│   ├── core/             # Model pipelines and dataloaders
│   ├── api/              # FastAPI server (genfrominterface.py)
│   └── utils/            # Path management and utilities
├── harmonymodule/         # MIDI generation utilities
│   ├── drum_sampler_simple.py
│   ├── arrange.py
│   ├── render.py
│   └── chords.py
└── docs/                  # Documentation
```

## Quick Start

### Frontend (React DAW)
```bash
cd frontend
npm install
npm start              # Development mode (http://localhost:3000)
npm run build          # Production build
```

### Backend (Inference API)
```bash
cd inference/api
python genfrominterface.py --port 8070 --ckpt /path/to/checkpoint.ckpt
```

### Celery Worker (Async Tasks)
```bash
celery -A celery_config.celery_app worker --loglevel=INFO
```

## Features

### 🎹 Frontend DAW
- **Multi-bus timeline** - Organize tracks by instrument type
- **Real-time playback** - Synchronized multi-track audio with waveforms
- **MIDI piano roll** - Interactive note editing with F0 pitch contour mode
- **AI generation panel** - ACE-Step integration with voice/instrument selection
- **Effects processing** - Reverb, delay, EQ, RC20, SpecCraft
- **Project management** - Save/load with GCS cloud storage
- **Theme customization** - Glass morphism UI with color editor

### 🔧 Backend Inference
- **ACE-Step generation** - Multi-voice AI music synthesis
- **Stem separation** - Demucs integration (vocals, drums, bass, other)
- **MIDI synthesis** - Omnisphere rendering
- **FX processing** - Pedalboard effects chain
- **Celery task queue** - Async generation with status polling
- **GCS integration** - Background cloud upload

### 🎵 Harmony Module
- **MIDI generation** - Chord progressions and patterns
- **Drum sampler** - Pattern-based drum synthesis
- **Arrangement tools** - Orchestral MIDI utilities

## Technology Stack

### Frontend
- React 18 + Context API
- Web Audio API (playback engine)
- Canvas API (waveform/MIDI visualization)
- Axios (HTTP client)

### Backend
- PyTorch Lightning (model inference)
- FastAPI + Uvicorn (API server)
- Celery + Redis (task queue)
- Demucs (stem separation)
- Google Cloud Storage (file persistence)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate-ace-step` | POST | Generate audio with ACE-Step |
| `/api/separate-stems` | POST | Separate audio into stems |
| `/api/download-with-fx` | POST | Apply effects to audio |
| `/api/render-omnisphere` | POST | Synthesize MIDI with Omnisphere |
| `/task/{task_id}` | GET | Check task status |

## Configuration

### Path Management
All file operations use centralized path management (`inference/utils/output_paths.py`):

```python
from inference.utils.output_paths import get_output_path, ensure_path_exists

# All outputs go to /mnt/models/ subdirectories
output_dir = ensure_path_exists(get_output_path('ace_step_output', process_id='abc123'))
# Returns: /mnt/models/generated_ui/ace_step_output_abc123
```

### Environment Variables

**Frontend** (`.env`):
```bash
REACT_APP_API_URL=http://localhost:8070
REACT_APP_GCS_BUCKET=score-ai-generations
```

**Backend** (system):
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Deployment

### Production Setup
1. **Build frontend**: `cd frontend && npm run build`
2. **Deploy frontend**: `sudo cp -r build/* /var/www/html/build/`
3. **Configure nginx**: Serve frontend at `/`, proxy `/api/` to port 8070
4. **Start backend**: Run via `start-logs.sh` with tmux
5. **Start Celery**: Launch worker for async task processing

### Nginx Configuration
```nginx
# Frontend
location / {
    root /var/www/html/build;
    try_files $uri /index.html;
}

# Backend API
location /api/ {
    proxy_pass http://localhost:8070;
}
```

## System Requirements

- **GPU**: CUDA-capable (RTX 3090/4090 recommended)
- **RAM**: 48GB+
- **Storage**: Mounted disk at `/mnt/models/`
- **Node.js**: 16+ (frontend)
- **Python**: 3.10+ (backend)

## Documentation

- [Frontend README](frontend/README.md) - DAW interface details
- [Backend README](inference/README.md) - API and model docs (coming soon)
- [Harmony Module README](harmonymodule/README.md) - MIDI utilities

## License

Proprietary - Doseedo AI Music Platform

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React)                     │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │   DAW    │  │   MIDI   │  │  Generation Panel   │   │
│  │ Timeline │  │  Editor  │  │  (ACE-Step UI)      │   │
│  └──────────┘  └──────────┘  └─────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/WebSocket
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Backend API (FastAPI:8070)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Generation  │  │  Stem Sep    │  │  FX Chain    │  │
│  │  Endpoints   │  │  (Demucs)    │  │ (Pedalboard) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ Celery Tasks
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Celery Worker Queue                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │     ACE-Step Inference (PyTorch Lightning)       │   │
│  │  • Multi-voice generation                        │   │
│  │  • ControlNet conditioning                       │   │
│  │  │  • F0 pitch control                            │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Google Cloud Storage                      │
│              (Persistent file storage)                   │
└─────────────────────────────────────────────────────────┘
```

## Contributing

This is a production system. For development:
1. Create feature branch from `main`
2. Test thoroughly in development environment
3. Build and test production bundles
4. Submit PR with detailed description

---

**Built with** ❤️ **by the Doseedo team**
