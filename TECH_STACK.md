# Doseedo Platform - Detailed Tech Stack

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INFRASTRUCTURE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Nginx (HTTPS/SSL)  →  React SPA  →  Python Backend  →  ML Models      │
│       :443              :3000          :8001/:8070         GPU          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# FRONTEND

## Core Framework

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.2.0 | UI framework with hooks & functional components |
| **React DOM** | 18.2.0 | DOM rendering |
| **React Router DOM** | 6.30.2 | Client-side routing (SPA navigation) |
| **React Scripts** | 5.0.1 | Create React App build toolchain |

## State Management

| Technology | Purpose |
|------------|---------|
| **React Context API** | Global state (AppContext.js) |
| **useReducer** | Complex state logic (70+ action types) |
| **useState/useEffect** | Local component state |
| **Custom Hooks** | Reusable stateful logic (10+ hooks) |

### Custom Hooks

```javascript
useAudioPlayback.js     // Web Audio API playback control
useAudioRecorder.js     // MediaRecorder integration
useGeneration.js        // Generation API orchestration
useWaveform.js          // Waveform data processing
useWaveSurfer.js        // WaveSurfer.js integration
useTimeline.js          // Timeline calculations (zoom, scroll)
useMetronome.js         // Tempo-synced metronome
useKeyboardControls.js  // Keyboard shortcuts (Space, Cmd+C/V/Z)
useVideoProcessing.js   // Video scene detection
```

---

## Audio Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **WaveSurfer.js** | 7.4.0 | Waveform visualization & playback |
| **@tonejs/midi** | 2.0.28 | MIDI file parsing & manipulation |
| **soundfont-player** | 0.12.0 | SoundFont MIDI playback |
| **Tuna.js** | 1.0.15 | Web Audio effects (reverb, delay, chorus) |
| **Web Audio API** | Native | Core audio processing |

### Web Audio Architecture

```
                     ┌─────────────┐
                     │  AudioContext│
                     └──────┬──────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
    │  Track 1  │     │  Track 2  │     │  Track N  │
    │ GainNode  │     │ GainNode  │     │ GainNode  │
    │ PanNode   │     │ PanNode   │     │ PanNode   │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                 │
          └────────────┬────┴─────────────────┘
                       │
                 ┌─────▼─────┐
                 │ Bus Mixer │
                 │ (per type)│
                 └─────┬─────┘
                       │
          ┌────────────┴────────────┐
          │                         │
    ┌─────▼─────┐             ┌─────▼─────┐
    │  FX Chain │             │  Reverb   │
    │  (Tuna)   │             │  Send     │
    └─────┬─────┘             └─────┬─────┘
          │                         │
          └────────────┬────────────┘
                       │
                 ┌─────▼─────┐
                 │  Master   │
                 │ GainNode  │
                 └─────┬─────┘
                       │
                 ┌─────▼─────┐
                 │destination│
                 └───────────┘
```

---

## AI & ML Integration

| Library | Version | Purpose |
|---------|---------|---------|
| **@huggingface/inference** | 4.13.0 | HuggingFace model inference |
| **@gradio/client** | 2.0.0-dev.1 | Gradio space integration |
| **OpenAI (via backend)** | - | GPT-4 chat assistant |

---

## UI Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **react-draggable** | 4.5.0 | Drag & drop track elements |
| **react-glass-ui** | 1.2.2 | Glass morphism components |
| **react-slideshow-image** | 4.3.2 | Image carousels |
| **interact.js** | 1.10.0 | Drag, resize, gesture handling |

---

## Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| **axios** | 1.6.0 | HTTP client (alternative to fetch) |
| **JSZip** | 3.10.1 | ZIP file creation (session export) |
| **dotenv** | 17.2.3 | Environment variables |

---

## Styling System

### CSS Architecture

```
src/
├── styles/
│   ├── colors.css              # CSS custom properties (palette)
│   ├── liquid-glass.css        # Glass morphism effects
│   └── glass-theme-background.css  # Background gradients
├── assets/css/
│   ├── App.css                 # Global styles
│   └── original-style5.css     # Legacy styles (69KB)
└── components/
    └── */
        └── *.module.css        # CSS Modules (scoped styles)
```

### CSS Features Used

| Feature | Purpose |
|---------|---------|
| **CSS Modules** | Scoped component styles (42 module files) |
| **CSS Custom Properties** | Dynamic theming (`--bus-label-width`, `--panel-height`) |
| **CSS Grid** | DAW layout, panel arrangement |
| **Flexbox** | Component internal layout |
| **Glass Morphism** | `backdrop-filter: blur()`, gradients |
| **SVG Filters** | Liquid glass effects (LiquidGlassFilters.js) |
| **GPU Acceleration** | `transform`, `will-change` for animations |

### Theme Variables

```css
/* Core palette in colors.css */
--color-primary: #007AFF;
--color-secondary: #5856D6;
--color-accent: #FF9500;
--color-success: #34C759;
--color-warning: #FF9500;
--color-error: #FF3B30;

/* Glass effects */
--glass-bg: rgba(255, 255, 255, 0.1);
--glass-blur: 20px;
--glass-border: rgba(255, 255, 255, 0.2);

/* DAW-specific */
--daw-bg: #1a1a2e;
--track-height: 72px;
--timeline-ruler-height: 30px;
```

---

## Development Tools

| Tool | Purpose |
|------|---------|
| **ESLint** | Code linting (react-app config) |
| **@types/react** | TypeScript definitions (dev) |
| **@types/react-dom** | TypeScript definitions (dev) |
| **React DevTools** | Browser debugging |

---

## Build & Deploy

```bash
# Development
npm start          # Start dev server on :3000

# Production
npm run build      # Create optimized build
# Output: /var/www/html/doseedo-react/build/
```

### Browser Support

```json
{
  "production": [">0.2%", "not dead", "not op_mini all"],
  "development": [
    "last 1 chrome version",
    "last 1 firefox version",
    "last 1 safari version"
  ]
}
```

---

# BACKEND

## Python Version

| Requirement | Version |
|-------------|---------|
| **Python** | 3.10+ |
| **CUDA** | 11.8+ (for GPU) |

---

## Web Framework

| Library | Purpose | Port |
|---------|---------|------|
| **FastAPI** | Main API server | 8070 |
| **Uvicorn** | ASGI server | - |
| **Gradio** | ML interface (legacy) | - |
| **Pydantic** | Request/response validation | - |

### API Server Structure

```python
# genfrominterface.py
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

---

## Task Queue

| Library | Purpose |
|---------|---------|
| **Celery** | Async task queue |
| **Redis** | Message broker (implied) |

```python
from celery import Celery

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)
```

---

## Machine Learning

### Deep Learning

| Library | Purpose |
|---------|---------|
| **PyTorch** | Neural network framework |
| **torchaudio** | Audio processing |
| **PyTorch Lightning** | Training framework |
| **EnCodec** | Neural audio codec |

### ML Models Used

```
┌─────────────────────────────────────────────────────┐
│              ML MODEL ARCHITECTURE                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐     ┌─────────────┐                │
│  │   Pipeline  │────▶│   DiT/UNet  │                │
│  │  (trainer_  │     │  Diffusion  │                │
│  │  performer) │     │   Model     │                │
│  └─────────────┘     └─────────────┘                │
│         │                   │                        │
│         ▼                   ▼                        │
│  ┌─────────────┐     ┌─────────────┐                │
│  │  ControlNet │     │   EnCodec   │                │
│  │  Adapter    │     │   Decoder   │                │
│  └─────────────┘     └─────────────┘                │
│                             │                        │
│                             ▼                        │
│                      ┌─────────────┐                │
│                      │   Audio     │                │
│                      │   Output    │                │
│                      └─────────────┘                │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Traditional ML

| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | ≥1.0.0 | Feature extraction, classification |
| **XGBoost** | ≥1.5.0 | Gradient boosting (genre detection) |

---

## Audio Processing

| Library | Purpose |
|---------|---------|
| **scipy** | Signal processing, optimization |
| **numpy** | Numerical computing |
| **torchaudio** | Audio I/O, transformations |
| **rubberband** | Time-stretching, pitch-shifting |

### External Audio Tools

| Tool | Purpose |
|------|---------|
| **FluidSynth** | MIDI → Audio rendering (SoundFonts) |
| **FFmpeg** | Audio/video transcoding |
| **BasicPitch** | Audio → MIDI transcription |
| **Demucs** | Stem separation |

### SoundFonts Used

```
/home/arlo/Data/soundfonts/
├── Piano.sf2
├── Electric Piano.sf2
├── violin.sf2
├── viola.sf2
├── cello.sf2
├── trumpet.sf2
├── trombone.sf2
├── sax.sf2
├── flute.sf2
├── clarinet.sf2
├── bassoon.sf2
├── acoustic guitar.sf2
├── electric guitar.sf2
└── electric bass.sf2

/usr/share/sounds/sf2/
└── FluidR3_GM.sf2 (fallback)
```

---

## MIDI Processing

| Library | Version | Purpose |
|---------|---------|---------|
| **mido** | ≥1.2.10 | MIDI file I/O |
| **python-rtmidi** | ≥1.4.9 | Real-time MIDI |
| **pretty_midi** | - | High-level MIDI analysis |

---

## AI Integration

| Library | Purpose |
|---------|---------|
| **OpenAI** | GPT-4 chat (chatbot_service.py) |
| **Anthropic** | Claude integration (optional) |

```python
# chatbot_service.py
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,
    max_tokens=1000
)
```

---

## Data & Visualization

| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | ≥1.3.0 | Data manipulation |
| **matplotlib** | ≥3.4.0 | Plotting |
| **seaborn** | ≥0.11.0 | Statistical visualization |

---

## Utilities

| Library | Purpose |
|---------|---------|
| **tqdm** | Progress bars |
| **click** | CLI utilities |
| **python-dotenv** | Environment variables |

---

# INFRASTRUCTURE

## Web Server

| Component | Details |
|-----------|---------|
| **Nginx** | Reverse proxy, SSL termination |
| **Protocol** | HTTPS (TLS 1.2/1.3) |
| **HTTP/2** | Enabled |

### Nginx Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name doseedo.com www.doseedo.com;

    # SSL
    ssl_certificate /etc/ssl/certs/doseedo.com.crt;
    ssl_certificate_key /etc/ssl/private/doseedo.com_key.txt;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Static files (React build)
    root /var/www/html/doseedo-react/build;

    # API proxy
    location /api/ {
        proxy_pass http://localhost:8070;
    }
}
```

### Port Mapping

| Port | Service | Description |
|------|---------|-------------|
| 443 | Nginx | HTTPS (public) |
| 8001 | Auth Server | Registration, login |
| 8070 | Main API | genfrominterface.py |
| 8080 | Genome Server | Music genome API |
| 8090 | Chat Service | GPT-4 chatbot |
| 8095 | Labels API | Audio labeling |
| 8096 | Monitor API | Data monitoring |

---

## Security Headers

```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

---

## File Storage

| Path | Purpose |
|------|---------|
| `/var/www/html/doseedo-react/build/` | React production build |
| `/home/arlo/Data/soundfonts/` | SoundFont instruments |
| `/home/arlo/ScoreAI/audiofiles/` | Generated audio files |
| `/home/arlo/ScoreAI/temp_videos/` | Temporary video processing |
| `/mnt/msdd/` | Large file storage (generations) |

---

## GPU Resources

| Requirement | Specification |
|-------------|---------------|
| **GPU** | NVIDIA (CUDA-compatible) |
| **VRAM** | 24GB+ recommended |
| **Precision** | FP16 (half precision) |

```python
torch.set_float32_matmul_precision("high")
MAX_WINDOW_SLOW = 2048  # ~47.5 seconds at 43.066 fps
```

---

# DEPLOYMENT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              INTERNET                                    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │    Nginx       │
                         │   :443 SSL     │
                         └────────┬───────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │   Static    │        │  API Routes │        │    Auth     │
   │   Files     │        │   /api/*    │        │   /token    │
   │   React     │        │             │        │  /register  │
   └─────────────┘        └──────┬──────┘        └──────┬──────┘
                                 │                      │
                                 ▼                      ▼
                          ┌─────────────┐        ┌─────────────┐
                          │ FastAPI     │        │ Auth Server │
                          │ :8070       │        │ :8001       │
                          └──────┬──────┘        └─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │   Celery    │        │   Redis     │        │   GPU       │
   │   Workers   │◀──────▶│   Broker    │        │   PyTorch   │
   └─────────────┘        └─────────────┘        └─────────────┘
                                                        │
                                                        ▼
                                                 ┌─────────────┐
                                                 │  FluidSynth │
                                                 │  FFmpeg     │
                                                 │  Demucs     │
                                                 │  BasicPitch │
                                                 └─────────────┘
```

---

# VERSION SUMMARY

## Frontend Dependencies

```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-router-dom": "^6.30.2",
  "react-scripts": "5.0.1",
  "wavesurfer.js": "^7.4.0",
  "@tonejs/midi": "^2.0.28",
  "soundfont-player": "^0.12.0",
  "tunajs": "^1.0.15",
  "@huggingface/inference": "^4.13.0",
  "@gradio/client": "^2.0.0-dev.1",
  "axios": "^1.6.0",
  "react-draggable": "^4.5.0",
  "react-glass-ui": "^1.2.2",
  "interactjs": "^1.10.0",
  "jszip": "^3.10.1"
}
```

## Backend Dependencies

```
# Core
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# ML
torch
torchaudio
scikit-learn>=1.0.0
xgboost>=1.5.0

# Web
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
celery

# MIDI
mido>=1.2.10
python-rtmidi>=1.4.9
pretty_midi

# AI
openai
anthropic>=0.3.0

# Utilities
tqdm>=4.62.0
click>=8.0.0
python-dotenv>=0.19.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

*Generated: 2025-12-26*
