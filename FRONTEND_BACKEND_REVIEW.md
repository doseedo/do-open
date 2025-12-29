# Doseedo Platform Review
## Milanote-Style Frontend & Backend Architecture Analysis

---

# OVERVIEW

```
+--------------------------------------------------+
|                    DOSEEDO                        |
|        AI-Powered Music Production Platform       |
+--------------------------------------------------+
|                                                   |
|   Frontend (React SPA)  <-->  Backend (Python)    |
|   /var/www/html/doseedo-react   /home/arlo/Data   |
|                                                   |
+--------------------------------------------------+
```

---

# FRONTEND ARCHITECTURE

## React Application Structure

**Location:** `/var/www/html/doseedo-react`

### Entry Points
| File | Purpose |
|------|---------|
| `index.html` | HTML entry point |
| `src/index.js` | React DOM rendering |
| `src/App.js` | Main application router (~619 lines) |

---

## PAGES / VIEWS

```
+------------------+     +------------------+     +------------------+
|      HOME        |     |    DASHBOARD     |     |      STUDIO      |
|   /dashboard     |     |    /projects     |     |     /studio      |
+------------------+     +------------------+     +------------------+
| Landing page     |     | Project cards    |     | Full DAW view    |
| Navigation       |     | Create new       |     | Generation panel |
| Welcome content  |     | Load existing    |     | Timeline         |
+------------------+     +------------------+     +------------------+

+------------------+     +------------------+     +------------------+
|     SEARCH       |     |     PROFILE      |     |      TOOLS       |
|    /search       |     |    /profile      |     |     /tools       |
+------------------+     +------------------+     +------------------+
| Session search   |     | User settings    |     | Vocal Harmonizer |
| MIDI browser     |     | Account info     |     | Additional tools |
| Public sessions  |     | Logout           |     |                  |
+------------------+     +------------------+     +------------------+

+------------------+
|    WHAT'S NEW    |
|   /whats-new     |
+------------------+
| Feature updates  |
| Changelog        |
+------------------+
```

---

## COMPONENTS BREAKDOWN

### Core Layout Components

```
src/components/
├── Navbar/
│   └── Navbar.js              # Top navigation bar
├── Sidebar/
│   ├── LeftSidebar/
│   │   ├── LeftSidebar.js     # Main navigation sidebar
│   │   ├── SidebarLink.js     # Navigation links
│   │   └── SidebarSection.js  # Section grouping
│   └── RightSidebar/
│       ├── TrackInfoSidebar.js # Track/bus properties panel
│       └── BusInfoSidebar.js   # Bus-level controls
├── ResizeBar/
│   ├── ResizeBar.js           # Horizontal panel resizer
│   └── VerticalResizeBar.js   # Vertical panel resizer
└── LiquidGlassFilters/
    └── LiquidGlassFilters.js  # SVG glass morphism filters
```

### DAW Components (Digital Audio Workstation)

```
src/components/DAW/
├── DAWOptimized.js        # Main DAW container (CSS Grid layout)
├── Timeline.js            # Time ruler + waveform display
├── TimelineGrid.js        # Beat/bar grid overlay
├── TimelineTick.js        # Time markers
├── TimelineWrapper.js     # Scroll container
├── TransportControls.js   # Play/Pause/Stop/Record
├── BusRow.js              # Audio bus (group of tracks)
├── TrackBox.js            # Individual track display
├── TrackItem.js           # Track in list view
├── TrackList.js           # Track listing
├── TrackContainer.js      # Track wrapper
├── TrackBus.js            # Bus-track connection
├── DraggableTrack.js      # Drag-enabled track
├── OptimizedTrack.js      # Performance-optimized track
├── ChordTrack.js          # Chord progression overlay
├── SceneMarkers.js        # Video scene markers
├── MIDITrackVisualization.js # MIDI piano roll
├── CompositeMIDIView.js   # Combined MIDI view
├── PlayheadCursor.js      # Playback position indicator
├── MasterTrack.js         # Master output controls
├── MasterFXPanels.js      # Master effects panels
├── LevelMeter.js          # Audio level meter
├── PanKnob.js             # Stereo pan control
├── ReverbSlider.js        # Reverb send control
├── ZoomControls.js        # Timeline zoom
├── TempoControls.js       # BPM/tempo settings
├── MoreControls.js        # Additional options
├── StemsSidebar.js        # Stem separation panel
├── PlaceholderWaveform.js # Loading placeholder
└── Downloads.js           # Export/download options
```

### Generation & AI Components

```
src/components/GenerationPanel/
├── GenerationPanel.js           # Original generation UI
└── GenerationPanelOptimized.js  # Optimized with CSS Modules
    │
    ├── InstrumentSelection      # Piano/Guitar/Bass/Strings/Brass/Winds
    ├── GenerationSettings       # Steps, CFG, seed, etc.
    ├── AudioSettings            # Tempo, noise level, t0
    └── DrumSampler integration  # Random drum patterns

src/components/ChatWindow/
├── ChatWindow.js         # AI chat interface
└── systemPrompt.js       # GPT-4 system prompt

src/components/ChordWindow/
└── ChordWindow.js        # Chord progression editor
```

### Content Views

```
src/components/
├── VideoUpload/
│   └── VideoUploadOptimized.js  # Video upload + scene detection
├── MIDIChart/
│   └── MIDIChart.js             # MIDI piano roll editor
├── AudioWaveform/
│   └── AudioWaveform.js         # Audio waveform display
├── ImageViewer/
│   └── ImageViewer.js           # Album art generation
├── FXView/
│   ├── FXView.js                # Effects chain view
│   ├── FXPanel.js               # Individual effect panel
│   └── PluginSlot.js            # Effect slot in chain
└── ModeSelector/
    └── ModeSelector.js          # Video/MIDI/Audio/Image/FX tabs
```

### Utility Components

```
src/components/
├── Dashboard/
│   ├── Dashboard.js      # Projects dashboard
│   └── ProjectCard.js    # Session card
├── DrumSampler/
│   └── DrumSampler.js    # Drum pattern generator
├── MIDIBrowser/
│   └── MIDIBrowser.js    # Browse MIDI files
├── AutomationWindow/
│   └── AutomationWindow.js # Parameter automation
├── ThemeEditor/
│   └── ThemeEditor.js    # Real-time theme customization
├── GlassCard/
│   ├── GlassCard.js      # Glass morphism card
│   ├── GlassCardWrapper.js
│   ├── shader-utils.js
│   └── utils.js
├── GlassButton/
│   └── GlassButtonWrapper.js
├── GlassWrapper/
│   └── GlassWrapper.js
├── Home/
│   └── Home.js           # Home/landing page
├── Search/
│   └── Search.js         # Search functionality
├── UserInfo/
│   └── UserInfo.js       # User profile page
├── Tools/
│   ├── Tools.js          # Tools page
│   └── VocalHarmonizer.js # Voice harmonization tool
├── WhatsNew/
│   └── WhatsNew.js       # Changelog page
├── MySessions/
│   └── MySessions.js     # User sessions list
└── common/
    ├── Button.js         # Reusable button
    └── Slider.js         # Reusable slider
```

---

## STATE MANAGEMENT

### Context & Reducer Pattern

**File:** `src/context/AppContext.js`

```javascript
// Key State Sections
initialState = {
  // Project
  projectName, isAuthenticated,

  // Audio/Playback
  audioTracks, currentTrack, isPlaying, playheadPosition,

  // Generation Parameters
  generationParams: {
    instrumentGroup, instrumentSubgroup, generationKey,
    midiTarget, seed, steps, cfgWeight, t0, noiseLevel,
    monophonicMode, arrangeMode, fattenMode...
  },

  // DAW State (Bus-based architecture)
  buses: [{ id, type, name, tracks, gain, pan, mute, solo }],
  selectedTrack, selectedBus, copiedTrack,
  bpm, masterGain, masterFX, fxSlots,

  // Video State
  video: { videoId, sceneChanges, sceneTempos, audioUrl },

  // UI State
  zoomLevel, trackHeight, totalDuration,
  chordTrack, automationWindow,

  // Undo/Redo
  history: { past, future }
}
```

### Custom Hooks

```
src/hooks/
├── useAudioPlayback.js     # Web Audio API playback control
├── useAudioRecorder.js     # Recording functionality
├── useGeneration.js        # Generation API integration
├── useWaveform.js          # Waveform rendering
├── useWaveSurfer.js        # WaveSurfer.js integration
├── useTimeline.js          # Timeline calculations
├── useMetronome.js         # Metronome with tempo sync
├── useKeyboardControls.js  # Keyboard shortcuts
└── useVideoProcessing.js   # Video processing hooks
```

---

## SERVICES (API Layer)

### Service Files Overview

| Service File | Purpose | Backend Endpoints |
|-------------|---------|-------------------|
| `generationAPI.js` | Audio/MIDI generation | `/api/generate-do`, `/api/generate-ace-step`, `/api/audio-to-midi` |
| `midiGenerationAPI.js` | MIDI melody/chord generation | `/api/generate-melody`, `/api/render-chords` |
| `chordAPI.js` | Chord rendering | `/api/render-chords` |
| `drumSamplerAPI.js` | Drum pattern generation | `/api/drum-sampler/randomize`, `/api/drum-sampler/render` |
| `videoAPI.js` | Video processing | `/uploadvideo/`, `/task-status/`, `/exportAudio/` |
| `chatAPI.js` | AI chat assistant | `/api/chat`, `/api/chat/health` |
| `sessionAPI.js` | Session management | `/api/sessions`, `/api/sessions/search` |
| `authService.js` | Authentication | `/register/`, `/token/`, `/logout` |
| `huggingfaceAPI.js` | HuggingFace models | HF Inference API |
| `gcsUploadService.js` | Cloud storage | `/api/upload/gcs` |
| `sessionService.js` | Local session storage | localStorage |
| `sessionExportService.js` | Export sessions | - |
| `tunaFX.js` | Tuna.js effects | - |
| `pluginFX.js` | Plugin effects | - |
| `saveService.js` | Save functionality | - |

---

## FRONTEND API ENDPOINTS USED

### Core Generation Endpoints

```
POST /api/generate-do              # Main audio generation (monophonic voices)
GET  /api/generate-do/task/{id}    # Poll generation status

POST /api/generate-ace-step        # ACE-Step music generation
GET  /api/generate-ace-step/task/{id}

POST /api/generate-melody          # MIDI melody generation (basic/genre/context)
POST /api/render-chords            # Chord progression to MIDI

POST /api/audio-to-midi            # Audio transcription to MIDI
```

### Drum & Samples

```
POST /api/drum-sampler/randomize   # Random drum pattern
POST /api/drum-sampler/render      # Render drum MIDI to audio
POST /generate-drums               # Orchestral drums
POST /generate-risers              # Riser/transition sounds
```

### File Management

```
GET  /api/list-midi-files          # List available MIDI files
GET  /api/get-midi-file/{name}     # Download MIDI file
GET  /api/get-midi-info/{name}     # MIDI metadata
POST /api/upload-audio             # Upload audio file
POST /api/download-with-fx         # Download with effects applied
```

### Video Processing

```
POST https://doseedo.com/uploadvideo/     # Video upload + scene detection
GET  https://doseedo.com/task-status/{id} # Poll processing status
POST https://doseedo.com/exportAudio/     # Export audio to video
GET  https://doseedo.com/export/status/{id}
```

### Session & Auth

```
POST /api/sessions                 # Create session
GET  /api/sessions                 # List user sessions
GET  /api/sessions/{id}            # Get session
PATCH /api/sessions/{id}           # Update session
DELETE /api/sessions/{id}          # Delete session
POST /register/                    # User registration
POST /token/                       # Login
POST /logout                       # Logout
```

### Other

```
POST /api/chat                     # AI chat (GPT-4)
GET  /api/chat/health              # Chat service health
POST /api/generation-feedback      # User feedback on generations
POST /api/generate-track-image     # Album art generation
POST /api/vocal-harmonizer         # Vocal harmonization
POST /separate-stems               # Audio stem separation
```

---

# BACKEND ARCHITECTURE

## Main Server

**Primary Backend:** `/home/arlo/Data/genfrominterface.py` (~631KB, 13,000+ lines)

This is the monolithic FastAPI server handling all generation endpoints.

### Backend Endpoints (Active)

```python
# Main Generation
@app.post("/api/generate-do")           # Line 10555 - Main audio generation
@app.post("/api/generate-ace-step")     # Line 13112 - ACE-Step generation
@app.get("/api/generate-do/task/{id}")  # Line 11969 - Task polling

# MIDI Operations
@app.post("/api/generate-melody")       # Line 12997 - Melody generation
@app.post("/api/render-chords")         # Line 13332 - Chord rendering
@app.get("/api/list-midi-files")        # Line 12117 - List MIDI files
@app.get("/api/get-midi-file/{fn}")     # Line 12140 - Get MIDI file
@app.post("/api/audio-to-midi")         # Line 12161 - Audio transcription

# Drum Sampler
@app.post("/api/drum-sampler/randomize") # Line 12896
@app.post("/api/drum-sampler/render")    # Line 12931
@app.post("/generate-drums")             # Line 11711
@app.post("/generate-risers")            # Line 11655

# Audio Processing
@app.post("/api/download-with-fx")       # Line 12434 - Apply FX
@app.post("/api/upload-audio")           # Line 11592
@app.post("/separate-stems")             # Line 11414

# Supplementary
@app.post("/api/generate-track-image")   # Line 12744
@app.post("/api/generation-feedback")    # Line 13440
@app.get("/api/generation-feedback/stats") # Line 13492

# Downloads
@app.get("/download/{pid}/{fn}")         # Line 12109
@app.get("/download-ace-step/{pid}/{fn}")# Line 13322
@app.get("/download-chord-midi/{pid}/{fn}") # Line 13421
@app.get("/download-drums/{pid}/{fn}")   # Line 12988
@app.get("/download-image/{pid}/{fn}")   # Line 12830
```

---

## Supporting Services

### Chat Service

**File:** `/var/www/html/chatbot_service.py`
**Port:** 8090

```python
@app.post("/api/chat")          # GPT-4 chat endpoint
@app.get("/api/chat/health")    # Health check
```

Features:
- OpenAI GPT-4 integration
- DAW context awareness (BPM, key, tracks)
- Conversation history
- Music production assistant

### MIDI DNA API

**File:** `/midi_generator/api/server.py`
**Port:** 8000

```python
@app.post("/extract_dna")       # Extract 120-param DNA from MIDI
@app.post("/generate")          # Generate MIDI from DNA
@app.post("/edit")              # Edit MIDI via DNA params
@app.get("/parameters")         # List available parameters
@app.get("/health")             # Health check
```

---

## Backend Scripts Used by Frontend

### Actively Used Scripts

```
/home/arlo/Data/
├── genfrominterface.py      # Main generation server (ALL /api/* endpoints)
├── trainer_performerCN2.py  # ML Pipeline for audio generation
├── dataloader.py            # APPROVED_GROUPS, APPROVED_SUBGROUPS
└── generation_trajectory_logger.py  # Debug logging

/var/www/html/
└── chatbot_service.py       # AI chat service

/midi_generator/
├── api/server.py            # MIDI DNA API
├── learning/modular_discovery_pipeline.py  # DNA extraction
└── (various processing modules called by genfrominterface)
```

### Key Dependencies in genfrominterface.py

```python
from trainer_performerCN2 import Pipeline  # Audio generation
from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS
import pretty_midi                          # MIDI processing
import mido                                 # MIDI I/O
import torchaudio                          # Audio processing
import gradio as gr                        # Web interface (legacy)
```

---

## UNUSED Backend Scripts

The following directories/scripts in `midi_generator/` are **NOT directly called** by the frontend:

```
midi_generator/
├── 1_approaches/           # Experimental approaches (unused)
├── 3_analysis/             # Analysis tools (unused)
├── algorithms/             # Algorithmic composition (unused)
├── analysis/               # Music analysis (unused)
├── constraints/            # Constraint satisfaction (unused)
├── experts/                # Expert systems (unused)
├── feature_selection/      # Feature selection (unused)
├── genres/                 # Genre-specific modules (partially used)
├── integration/            # Integration modules (unused)
├── llm/                    # LLM integration (unused)
├── modular_blueprint.py    # Blueprint (unused)
├── monitoring/             # Monitoring (unused)
├── multi_genre/            # Multi-genre (unused)
├── optimization/           # Optimization (unused)
├── orchestration/          # Orchestration (unused)
├── processing/             # Processing (unused)
├── synthesis/              # Synthesis (unused)
├── tools/                  # Additional tools (unused)
├── tracking/               # Tracking (unused)
├── training/               # Training scripts (offline only)
├── transformation/         # Transformation (unused)
├── transforms/             # Transforms (unused)
├── validation/             # Validation (unused)
└── core/                   # Core modules (some used internally)
```

**Note:** Many of these modules may be used indirectly through the `Pipeline` class or for offline training, but they are not directly called by frontend API requests.

---

# DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │ Generation   │    │    DAW       │    │    Video     │         │
│   │    Panel     │    │  Component   │    │   Upload     │         │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘         │
│          │                   │                   │                  │
│          ▼                   ▼                   ▼                  │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                    Services Layer                        │      │
│   │  generationAPI | videoAPI | sessionAPI | chatAPI | ...  │      │
│   └─────────────────────────────────────────────────────────┘      │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                        HTTP / HTTPS
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          BACKEND (Python)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │             genfrominterface.py (Main Server)             │     │
│   │                                                           │     │
│   │   /api/generate-do     → Pipeline.generate()              │     │
│   │   /api/generate-melody → ProperMelodyGenerator            │     │
│   │   /api/render-chords   → ChordRenderer                    │     │
│   │   /api/audio-to-midi   → BasicPitch                       │     │
│   │   /api/drum-sampler/*  → FluidSynth                       │     │
│   │   /separate-stems      → Demucs                           │     │
│   └──────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │                    ML Models & Tools                      │     │
│   │                                                           │     │
│   │   trainer_performerCN2.Pipeline (Audio Generation)        │     │
│   │   ACE-Step Pipeline (Music Generation)                    │     │
│   │   BasicPitch (Audio to MIDI)                              │     │
│   │   Demucs (Stem Separation)                                │     │
│   │   FluidSynth (MIDI to Audio)                              │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

# TECHNOLOGY STACK

## Frontend

| Category | Technology |
|----------|------------|
| Framework | React 18.2.0 |
| Routing | React Router DOM 6.30.2 |
| State | Context API + useReducer |
| Styling | CSS Modules + CSS Variables + Glass Morphism |
| Audio | Web Audio API, WaveSurfer.js, Tone.js, Tuna.js |
| MIDI | @tonejs/midi, soundfont-player |
| HTTP | fetch API, axios |
| AI | @huggingface/inference, @gradio/client |
| UI | react-draggable, react-glass-ui, interactjs |
| Build | Create React App (react-scripts) |

## Backend

| Category | Technology |
|----------|------------|
| Framework | FastAPI |
| ML | PyTorch, torchaudio |
| MIDI | mido, pretty_midi |
| Audio | scipy, numpy, FluidSynth, rubberband |
| AI | OpenAI (GPT-4), HuggingFace |
| Task Queue | Celery (implied by task polling) |
| Audio Separation | Demucs |
| Transcription | BasicPitch |

---

# KEY FEATURES

## Audio Generation

- **Monophonic Voice Separation**: Generate individual instrument voices
- **Instrument Groups**: Piano, Guitar, Bass, Strings, Brass, Winds
- **Subgroups**: Solo instruments + ensembles
- **Arrange Mode**: Automatic voice assignment by register
- **Fatten Mode**: Generate variations
- **ACE-Step**: AI music generation with lyrics/prompt

## MIDI Capabilities

- **Melody Generation**: Basic, Genre-specific, Context-aware modes
- **Chord Rendering**: Random voicings, rhythms, styles
- **Audio-to-MIDI**: Transcription using BasicPitch
- **Piano Roll Editor**: Visual MIDI editing
- **Chord Track**: Overlay chord symbols

## Video Integration

- **Scene Detection**: Automatic scene change detection
- **Tempo Mapping**: Compute optimal tempo per scene
- **Audio Export**: Merge generated audio back to video

## DAW Features

- **Bus-based Architecture**: Group tracks by type
- **Transport Controls**: Play, pause, seek
- **Zoom Controls**: Horizontal + vertical zoom
- **Master FX**: Reverb, EQ, FX chain
- **Metronome**: Tempo-synced click
- **Keyboard Shortcuts**: Play (Space), Copy (Cmd+C), Paste (Cmd+V), Undo (Cmd+Z)

---

# ASSESSMENT

## Strengths

1. **Rich Feature Set**: Comprehensive music production capabilities
2. **Modern React**: Hooks, Context, CSS Modules, performance optimization
3. **Real-time Audio**: Web Audio API with proper gain staging
4. **AI Integration**: GPT-4 chat, HuggingFace models, custom ML
5. **Glass Morphism UI**: Consistent visual design

## Areas for Improvement

1. **Backend Monolith**: 631KB single file is hard to maintain
2. **Unused Code**: Many modules in midi_generator not used
3. **Type Safety**: No TypeScript in frontend
4. **Error Handling**: Could be more robust in some services
5. **Testing**: Limited test coverage visible

## Recommendations

1. Split `genfrominterface.py` into modular FastAPI routers
2. Add TypeScript to frontend for type safety
3. Remove or archive unused backend modules
4. Add comprehensive error boundaries in React
5. Implement end-to-end testing

---

*Generated: 2025-12-26*
*Review covers: doseedo-react frontend + genfrominterface.py backend*
