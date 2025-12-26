# Doseedo GitHub Repository Structure
## Complete Overview: Frontend, Backend, Network, Services & Data Stores

---

## Repository Information

| Field | Value |
|-------|-------|
| **Repository** | `doseedo/Do` |
| **URL** | `github.com/doseedo/Do` |
| **Current Branch** | `claude/review-frontend-backend-z1IAJ` |
| **Main Branch** | `main` |

---

# REPOSITORY TREE OVERVIEW

```
doseedo/Do/
│
├── 📁 var/www/html/                    # FRONTEND (React SPA)
│   ├── doseedo-react/                  # Main React application (2.4MB)
│   └── chatbot_service.py              # GPT-4 chat service
│
├── 📁 home/arlo/                       # BACKEND SERVICES
│   ├── Data/                           # Main backend scripts (2.4MB)
│   ├── ScoreAI/                        # Video processing service
│   └── harmonymodule/                  # Harmony/melody modules
│
├── 📁 midi_generator/                  # ML MUSIC GENERATION (203MB)
│   ├── api/                            # REST API server
│   ├── core/                           # Core music modules
│   ├── models/                         # ML models
│   ├── training/                       # Training scripts
│   └── ...                             # 46 subdirectories
│
├── 📁 web-audio-plugins/               # AUDIO EFFECTS (1.7MB)
│   ├── vintage/                        # Vintage emulations
│   ├── creative/                       # Creative effects
│   └── ...                             # 15 plugin categories
│
├── 📁 etc/nginx/                       # INFRASTRUCTURE
│   └── sites-available/default         # Nginx configuration
│
├── 📁 parameters/                      # CONFIGURATION
├── 📁 docs/                            # DOCUMENTATION
├── 📁 tests/                           # TEST SUITE
├── 📁 scripts/                         # UTILITY SCRIPTS
└── 📁 output/                          # GENERATED OUTPUT
```

---

# FRONTEND STRUCTURE

## Location: `/var/www/html/doseedo-react/`

```
doseedo-react/                          # React SPA (2.4MB)
│
├── 📄 package.json                     # Dependencies & scripts
├── 📄 README.md                        # Frontend documentation
│
├── 📁 public/                          # Static assets
│   ├── index.html                      # HTML entry point
│   ├── favicon/                        # App icons
│   ├── assets/                         # Static assets
│   └── examples/                       # Example files
│
├── 📁 src/                             # Source code
│   │
│   ├── 📄 index.js                     # React entry point
│   ├── 📄 App.js                       # Main application (619 lines)
│   │
│   ├── 📁 components/                  # UI Components (32 folders, 75+ files)
│   │   ├── DAW/                        # Digital Audio Workstation (36 files)
│   │   │   ├── DAWOptimized.js         # Main DAW container
│   │   │   ├── Timeline.js             # Time ruler
│   │   │   ├── BusRow.js               # Audio bus component
│   │   │   ├── TrackBox.js             # Track display
│   │   │   ├── TransportControls.js    # Play/pause/stop
│   │   │   ├── MIDITrackVisualization.js # Piano roll
│   │   │   └── ... (30 more)
│   │   │
│   │   ├── GenerationPanel/            # AI Generation UI
│   │   │   ├── GenerationPanel.js
│   │   │   └── GenerationPanelOptimized.js
│   │   │
│   │   ├── VideoUpload/                # Video processing
│   │   ├── MIDIChart/                  # MIDI editor
│   │   ├── AudioWaveform/              # Waveform display
│   │   ├── ChatWindow/                 # AI chat
│   │   ├── FXView/                     # Effects chain
│   │   ├── ChordWindow/                # Chord editor
│   │   ├── Dashboard/                  # Project manager
│   │   ├── Navbar/                     # Top navigation
│   │   ├── Sidebar/                    # Side panels
│   │   ├── Tools/                      # Utility tools
│   │   └── ... (20 more)
│   │
│   ├── 📁 services/                    # API Layer (15 files)
│   │   ├── generationAPI.js            # Audio generation
│   │   ├── midiGenerationAPI.js        # MIDI generation
│   │   ├── videoAPI.js                 # Video processing
│   │   ├── chatAPI.js                  # AI chat
│   │   ├── sessionAPI.js               # Session management
│   │   ├── authService.js              # Authentication
│   │   ├── drumSamplerAPI.js           # Drum patterns
│   │   ├── chordAPI.js                 # Chord rendering
│   │   ├── huggingfaceAPI.js           # HF models
│   │   ├── gcsUploadService.js         # Cloud storage
│   │   ├── sessionService.js           # Local storage
│   │   ├── sessionExportService.js     # Export
│   │   ├── tunaFX.js                   # Tuna effects
│   │   ├── pluginFX.js                 # Plugin effects
│   │   └── saveService.js              # Save operations
│   │
│   ├── 📁 hooks/                       # Custom React Hooks (10 files)
│   │   ├── useAudioPlayback.js         # Web Audio playback
│   │   ├── useAudioRecorder.js         # Recording
│   │   ├── useGeneration.js            # Generation flow
│   │   ├── useWaveform.js              # Waveform data
│   │   ├── useWaveSurfer.js            # WaveSurfer integration
│   │   ├── useTimeline.js              # Timeline math
│   │   ├── useMetronome.js             # Metronome
│   │   ├── useKeyboardControls.js      # Shortcuts
│   │   └── useVideoProcessing.js       # Video hooks
│   │
│   ├── 📁 context/                     # State Management
│   │   ├── AppContext.js               # Global state (~1500 lines)
│   │   └── AuthContext.js              # Auth state
│   │
│   ├── 📁 utils/                       # Utilities (8 files)
│   │   ├── api.js                      # HTTP helpers
│   │   ├── audioUtils.js               # Audio utilities
│   │   ├── midiParser.js               # MIDI parsing
│   │   ├── midiPlayer.js               # MIDI playback
│   │   ├── pitchDetection.js           # Pitch detection
│   │   ├── themeManager.js             # Theme handling
│   │   └── feedbackLogger.js           # User feedback
│   │
│   ├── 📁 styles/                      # Global Styles
│   │   ├── colors.css                  # Color palette
│   │   ├── liquid-glass.css            # Glass effects
│   │   └── glass-theme-background.css  # Backgrounds
│   │
│   └── 📁 assets/                      # Assets
│       └── css/
│           ├── App.css                 # App styles
│           └── original-style5.css     # Legacy (69KB)
│
└── 📁 build/                           # Production build (served by Nginx)
```

---

# NETWORK & SERVICES

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INTERNET (:443)                                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      NGINX REVERSE PROXY                                 │
│                    /etc/nginx/sites-available/default                    │
├─────────────────────────────────────────────────────────────────────────┤
│  SSL: TLS 1.2/1.3  │  HTTP/2  │  HSTS  │  Security Headers             │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  STATIC FILES   │     │   API ROUTES    │     │   AUTH ROUTES   │
│  /              │     │   /api/*        │     │   /token        │
│  /static/*      │     │   /generate*    │     │   /register/    │
│                 │     │   /download*    │     │                 │
│  React Build    │     │                 │     │                 │
│  :443 (Nginx)   │     │  :8070 FastAPI  │     │  :8001 Auth     │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────────────┐
│                        SERVICE ENDPOINTS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ Main API :8070  │  │ Chat API :8090  │  │ Genome API:8080 │         │
│  │ genfrominterface│  │ chatbot_service │  │ genome_server   │         │
│  │ .py (631KB)     │  │ .py             │  │ .py             │         │
│  │                 │  │                 │  │                 │         │
│  │ • /api/generate │  │ • /api/chat     │  │ • /genome/api/  │         │
│  │ • /api/render-  │  │ • /api/chat/    │  │ • /transform    │         │
│  │   chords        │  │   health        │  │                 │         │
│  │ • /api/audio-   │  │                 │  │                 │         │
│  │   to-midi       │  │ OpenAI GPT-4    │  │ MIDI DNA        │         │
│  │ • /api/drum-    │  │                 │  │ Extraction      │         │
│  │   sampler/*     │  │                 │  │                 │         │
│  │ • /separate-    │  │                 │  │                 │         │
│  │   stems         │  │                 │  │                 │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ Labels :8095    │  │ Monitor :8096   │  │ Celery Workers  │         │
│  │ Audio labeling  │  │ Data monitor    │  │ Async tasks     │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Port Mapping Reference

| Port | Service | File Location | Purpose |
|------|---------|---------------|---------|
| **443** | Nginx | `/etc/nginx/sites-available/default` | HTTPS, static files, reverse proxy |
| **8001** | Auth Server | (external) | User registration, login, JWT |
| **8070** | Main API | `/home/arlo/Data/genfrominterface.py` | All generation endpoints |
| **8080** | Genome Server | `/midi_generator/tools/web_interface/genome_server.py` | MIDI DNA extraction |
| **8090** | Chat Service | `/var/www/html/chatbot_service.py` | GPT-4 chatbot |
| **8095** | Labels API | (external) | Audio labeling |
| **8096** | Monitor API | (external) | Data monitoring |

## Nginx Route Mapping

| Route Pattern | Upstream | Timeout |
|---------------|----------|---------|
| `/` | Static: `/var/www/html/doseedo-react/build` | - |
| `/api/generate-do` | `http://localhost:8070` | 1000s |
| `/api/generate-ace-step` | `http://localhost:8070` | 1000s |
| `/api/generate-melody` | `http://localhost:8070` | 300s |
| `/api/render-chords` | `http://localhost:8070` | 1000s |
| `/api/chat` | `http://localhost:8090` | 120s |
| `/separate-stems` | `http://localhost:8070` | 600s |
| `/token` | `http://localhost:8001` | 60s |
| `/register/` | `http://localhost:8001` | 60s |
| `/genome/*` | `http://localhost:8080` | 300s |
| `/media/audio/` | Alias: `/home/arlo/ScoreAI/audiofiles/` | - |
| `/temp_videos/` | Alias: `/home/arlo/ScoreAI/temp_videos/` | - |

---

# BACKEND STRUCTURE

## Location: `/home/arlo/Data/`

```
home/arlo/Data/                         # Main Backend (2.4MB)
│
├── 📄 genfrominterface.py              # ⭐ MAIN API SERVER (631KB, 13,500 lines)
│                                       #    All /api/* endpoints
│
├── 📄 trainer_performerCN2.py          # ML Pipeline (current)
├── 📄 trainer_performerCN.py           # ML Pipeline (backup)
├── 📄 trainer_performer.py             # ML Pipeline (legacy)
│
├── 📄 dataloader.py                    # APPROVED_GROUPS, APPROVED_SUBGROUPS
├── 📄 generation_trajectory_logger.py  # Debug logging
├── 📄 generation_feedback_logger.py    # User feedback
│
├── 📄 melody_generator_proper.py       # Melody generation
├── 📄 melody_advanced.py               # Advanced melody
├── 📄 harmony_advanced.py              # Harmony generation
├── 📄 chord_progression_generator.py   # Chord progressions
│
├── 📄 film_scoring_engine.py           # Film scoring
├── 📄 drums.py                         # Drum generation
├── 📄 lyrics.py                        # Lyrics handling
│
├── 📄 encode.py                        # Audio encoding
├── 📄 decode.py                        # Audio decoding
├── 📄 dcae.py                          # Deep audio codec
├── 📄 conditioning_encoder.py          # Conditioning
│
├── 📄 ace_step_wrapper.py              # ACE-Step wrapper
├── 📄 ace_step_noise_wrapper.py        # ACE-Step noise
├── 📄 generate_ace_step_detailed.py    # ACE-Step detailed
│
├── 📄 unified_preprocess.py            # Preprocessing
├── 📄 unified_labeler.py               # Labeling
├── 📄 unified_validator.py             # Validation
│
├── 📄 run_fastapi_server.py            # Server runner
├── 📄 render_omnisphere.py             # Omnisphere rendering
├── 📄 gcs_storage.py                   # Cloud storage
├── 📄 normalize.py                     # Audio normalization
│
├── 📁 dø/                              # DoTrainComponents
│   ├── __init__.py
│   └── do/
│
├── 📁 soundfonts/                      # SoundFont instruments
│   ├── Piano.sf2
│   ├── violin.sf2
│   ├── trumpet.sf2
│   └── ... (15+ instruments)
│
└── 📁 mute_translator/                 # Mute translation module
    ├── models.py
    ├── train_translator.py
    └── inference.py
```

## Location: `/home/arlo/ScoreAI/`

```
home/arlo/ScoreAI/                      # Video Processing Service
│
├── 📄 main.py                          # Video processing server
├── 📁 audiofiles/                      # Generated audio (served at /media/audio/)
├── 📁 temp_videos/                     # Temp video storage (served at /temp_videos/)
└── 📁 temp_exports/                    # Export temp files
```

## Location: `/midi_generator/`

```
midi_generator/                         # ML Music Generation (203MB, 349 .py files)
│
├── 📁 api/                             # REST API
│   ├── server.py                       # MIDI DNA API
│   ├── unified_api.py                  # Unified generation API
│   ├── synthesis_api.py                # Synthesis API
│   └── big_band_api.py                 # Big band API
│
├── 📁 core/                            # Core Music Modules (30+ files)
│   ├── component_system.py
│   ├── instrument_library.py
│   ├── instrumentation_specialist.py
│   ├── modal_harmony.py
│   ├── neo_riemannian.py
│   ├── microtonality.py
│   ├── multi_genre_arranger.py
│   └── ensemble_registry.py
│
├── 📁 models/                          # ML Models
│   ├── scaled_hierarchical_mtl.py      # Hierarchical MTL
│   ├── registry_manager.py             # Model registry (62KB)
│   ├── loss_functions.py
│   ├── features_to_midi.py
│   └── rule_based_midi.py
│
├── 📁 training/                        # Training Scripts
│   ├── model_trainer.py                # Main trainer (77KB)
│   ├── synthetic_data_generator.py     # Data gen (85KB)
│   └── hierarchical_mtl/               # MTL training
│
├── 📁 orchestration/                   # Orchestration
│   └── expansion_orchestrator.py       # (33KB)
│
├── 📁 learning/                        # Learning System
│   ├── modular_discovery_pipeline.py   # DNA extraction
│   ├── discovered_patterns/
│   └── template_library/
│
├── 📁 generators/                      # Generators
├── 📁 genres/                          # Genre-specific (35+ genres)
├── 📁 algorithms/                      # Algorithms
├── 📁 analysis/                        # Analysis tools
├── 📁 synthesis/                       # Synthesis
├── 📁 transformation/                  # Transformations
├── 📁 transforms/                      # Transform library
├── 📁 experts/                         # Expert systems
├── 📁 constraints/                     # Constraints
├── 📁 validation/                      # Validation
├── 📁 optimization/                    # Optimization
├── 📁 processing/                      # Processing
├── 📁 integration/                     # Integration
├── 📁 interface/                       # Web interface
├── 📁 llm/                             # LLM integration
├── 📁 monitoring/                      # Monitoring
├── 📁 tracking/                        # Tracking
├── 📁 storage/                         # Storage
├── 📁 utils/                           # Utilities
│
├── 📁 tools/                           # Tools
│   ├── web_interface/
│   │   ├── genome_server.py            # Genome API (:8080)
│   │   └── hierarchy_server.py
│   └── big_band/
│
├── 📁 deployment/                      # Deployment
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── scripts/deploy.sh
│
├── 📁 parameters/                      # Parameters
│   ├── hierarchical_parameters.json
│   └── registry.json
│
├── 📁 midi_corpus/                     # MIDI Dataset
├── 📁 checkpoints/                     # Model Checkpoints
├── 📁 data/                            # Data files
├── 📁 examples/                        # Examples
├── 📁 docs/                            # Documentation
│
├── 📁 tests/                           # Tests
│   └── integration/
│       ├── test_end_to_end.py
│       ├── test_training_pipeline.py
│       └── test_feature_extraction.py
│
├── 📄 __init__.py
├── 📄 modular_blueprint.py             # Module definitions (29KB)
├── 📄 labeled_dataset.json             # Dataset (3.6MB)
└── 📄 README.md
```

---

# WEB AUDIO PLUGINS

## Location: `/web-audio-plugins/`

```
web-audio-plugins/                      # Audio Effects Library (1.7MB)
│
├── 📄 index.js                         # Plugin registry
├── 📄 register-all.js                  # Auto-registration
├── 📄 test-all-plugins.html            # Test page
├── 📄 README.md
│
├── 📁 core/                            # Fundamental nodes
│   └── *.js
│
├── 📁 vintage/                         # Vintage Emulations
│   ├── SSL-bus-compressor.js
│   ├── neve-preamp.js
│   ├── 1176-compressor.js
│   ├── LA2A-compressor.js
│   └── tape-saturation.js
│
├── 📁 creative/                        # Creative Effects
│   ├── grain-processor.js
│   ├── beat-repeat.js
│   ├── vinyl-distortion.js
│   └── erosion.js
│
├── 📁 dynamics/                        # Dynamics
│   ├── compressor.js
│   ├── gate.js
│   └── limiter.js
│
├── 📁 delay/                           # Delay Effects
├── 📁 reverb/                          # Reverb Effects
├── 📁 distortion/                      # Distortion Effects
├── 📁 eq/                              # Equalizers
├── 📁 filters/                         # Filter Designs
├── 📁 modulation/                      # Modulation (LFO, chorus, flanger)
├── 📁 modulation-matrix/               # Routing System
├── 📁 spectral/                        # FFT-based Effects
├── 📁 utility/                         # Gain, Pan, Stereo
├── 📁 analysis/                        # Spectral Analysis
├── 📁 worklets/                        # AudioWorklet Processors
└── 📁 examples/                        # Usage Examples
```

---

# DATA & FEATURE STORE LOCATIONS

## Data Storage Paths

| Path | Type | Description | Served At |
|------|------|-------------|-----------|
| `/home/arlo/Data/soundfonts/` | SoundFonts | Instrument samples (.sf2) | Internal |
| `/home/arlo/ScoreAI/audiofiles/` | Audio | Generated audio files | `/media/audio/` |
| `/home/arlo/ScoreAI/temp_videos/` | Video | Temporary video files | `/temp_videos/` |
| `/home/arlo/ScoreAI/temp_exports/` | Temp | Export temp files | Internal |
| `/mnt/msdd/` | Storage | Large file storage | Internal |
| `/mnt/msdd/generation_debug/` | Logs | Debug trajectories | Internal |

## Feature & Model Storage

| Path | Content |
|------|---------|
| `/midi_generator/checkpoints/` | Model checkpoints (.ckpt) |
| `/midi_generator/labeled_dataset.json` | Training dataset (3.6MB) |
| `/midi_generator/learning/discovered_patterns/` | Learned patterns |
| `/midi_generator/learning/template_library/` | MIDI templates |
| `/midi_generator/parameters/` | Parameter configurations |
| `/output/semantic_encoders/` | Semantic representations |

## Configuration Files

| File | Purpose |
|------|---------|
| `/integration_manifest.json` | Integration config (26KB) |
| `/parameters/schema.json` | Parameter schema |
| `/parameters/core/` | Core parameters |
| `/parameters/examples/` | Example configs |
| `/etc/nginx/sites-available/default` | Nginx config |
| `/var/www/html/doseedo-react/package.json` | Frontend deps |
| `/requirements.txt` | Python deps |

---

# ROOT LEVEL FILES

```
doseedo/Do/                             # Repository Root
│
├── 📄 .gitignore                       # Git ignore rules
├── 📄 requirements.txt                 # Python dependencies
├── 📄 requirements_agent2.txt          # Agent dependencies
├── 📄 integration_manifest.json        # Integration config (26KB)
│
├── 📄 FRONTEND_BACKEND_REVIEW.md       # Architecture review
├── 📄 TECH_STACK.md                    # Tech stack documentation
├── 📄 GITHUB_STRUCTURE.md              # This document
├── 📄 ARCHITECTURE_REVIEW_SUMMARY.txt  # Review summary (12KB)
├── 📄 FILM_SCORING_ARCHITECTURE.txt    # Film scoring docs (19KB)
│
├── 📄 run_pipeline.py                  # Main ML pipeline runner
├── 📄 launch_tensorboard.sh            # TensorBoard launcher
├── 📄 unify_libraries.sh               # Library unification
│
├── 📄 verify_balanced_extraction.py    # Extraction verification
├── 📄 compare_versions.py              # Version comparison
├── 📄 compare_all_extractors.py        # Extractor comparison
├── 📄 add_tensorboard_logging.py       # TensorBoard setup
├── 📄 quick_train_fix.py               # Training fixes
├── 📄 simple_csv_logger.py             # CSV logging
│
├── 📄 parameter_validation_report.json # Validation report
├── 📄 merge_order.txt                  # Merge instructions
├── 📄 all_claude_branches.txt          # Branch list
└── 📄 swing_final.mid                  # Example MIDI
```

---

# DOCUMENTATION FILES

## Frontend Documentation

| File | Description |
|------|-------------|
| `var/www/html/doseedo-react/README.md` | Main frontend docs |
| `var/www/html/doseedo-react/QUICK_START.md` | Quick start guide |
| `var/www/html/doseedo-react/CONVERSION_GUIDE.md` | HTML→React conversion |
| `var/www/html/doseedo-react/DRUMSAMPLER_QUICKSTART.md` | DrumSampler guide |
| `var/www/html/doseedo-react/HUGGINGFACE_SETUP.md` | HuggingFace setup |
| `var/www/html/doseedo-react/DAW_CONVERSION_COMPLETE.md` | DAW conversion notes |

## Backend Documentation

| File | Description |
|------|-------------|
| `docs/API_REFERENCE.md` | API reference (52KB) |
| `docs/SEMANTIC_FEATURE_DISCOVERY.md` | Semantic features (78KB) |
| `docs/TROUBLESHOOTING.md` | Troubleshooting guide |
| `docs/BENCHMARKS.md` | Performance benchmarks |
| `midi_generator/README.md` | MIDI generator docs |
| `midi_generator/ORCHESTRATION_README.md` | Orchestration (15KB) |
| `midi_generator/MUSICAL_PROGRAM_SYNTHESIS_README.md` | Synthesis (14KB) |

---

# GIT HISTORY & BRANCHES

## Recent Commits

```
e96f91c Add detailed tech stack documentation
bedd0e8 Add comprehensive frontend/backend architecture review
39e3b55 Merge branch 'main' of github.com:doseedo/Do
41cf54c trumpey
d127cc1 Merge PR #101: claude/review-tools-page
77c3de7 Merge PR #100: claude/review-audio-effects
2e45cd6 Add Vocal Harmonizer tool
93eff11 Add multi-mode MIDI melody generation
777a997 Merge PR #99: claude/review-audio-effects
8d101f3 Integrate web-audio-plugins library
```

## Branch Naming Convention

```
claude/<feature>-<session-id>

Examples:
- claude/review-frontend-backend-z1IAJ
- claude/review-tools-page-01A6GE11Z9NrhbARMhiTV36u
- claude/review-audio-effects-01LmLPxgjG8pzZaNVhfBoq23
- claude/implement-ppm-model-01AgTSHtNyPY7oicsPMxoRoh
```

---

# SUMMARY STATISTICS

| Category | Count/Size |
|----------|------------|
| **Frontend Components** | 75+ React components |
| **Frontend Services** | 15 API service files |
| **Frontend Hooks** | 10 custom hooks |
| **Frontend CSS Modules** | 42 .module.css files |
| **Backend Python Files** | 349 .py files (midi_generator) |
| **Backend JSON Configs** | 18 .json files |
| **Web Audio Plugins** | 50+ effect plugins |
| **Documentation Files** | 30+ .md files |
| **Total Frontend Size** | 2.4MB |
| **Total Backend Size** | 203MB (midi_generator) |
| **Main API File** | 631KB (13,500 lines) |

---

*Generated: 2025-12-26*
*Repository: github.com/doseedo/Do*
