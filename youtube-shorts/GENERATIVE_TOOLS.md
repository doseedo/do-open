# Generative Models & Tools Available on This VM

Everything available at `/home/arlo` for making marble/plinko animations interact with pitch and music.

---

## Audio Generation Models

### ACE-Step (3.5B parameter neural audio synthesis)
- **Location**: `/home/arlo/Data/ACE-Step`
- **Checkpoints**: `/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/`
- **Conda env**: `ace_step`
- **What it does**: Generates high-fidelity audio from text prompts, lyrics, instrument conditioning
- **Key files**:
  - `/home/arlo/Data/ace_step_wrapper.py` — simple wrapper
  - `/home/arlo/Data/ace_step_wrapper_with_noise.py` — wrapper with noise injection
  - `/home/arlo/Data/ace_step_pipeline_custom.py` — custom pipeline
  - `/home/arlo/Data/generate_per_syllable_ace_step.py` — per-syllable lyrics generation
- **API endpoints**: `/api/generate-ace-step`, `/api/generate-ace-step-simple`, `/generate-ace-step-detailed`
- **Output**: 44.1kHz audio
- **Use for shorts**: Generate background music matching a mood/genre, or generate audio that can be analyzed for pitch to drive animations

### AudioCraft / MusicGen (Meta)
- **Location**: Installed via pip (`audiocraft==1.1.0`)
- **Training script**: `/home/arlo/Data/train.py`
- **What it does**: Text-to-music generation, melody-conditioned generation
- **Output**: 32kHz audio
- **Use for shorts**: Generate short music clips from text descriptions like "upbeat marimba melody" or "playful xylophone tune"

### dø (Sparse Synthesizer / Modulo)
- **Location**: `/home/arlo/Data/dø/do/`
- **Pipeline**: `/home/arlo/Data/dø/do/pipeline_do.py`
- **API endpoint**: `/api/generate-do`
- **What it does**: Generates audio from sparse symbolic representations using DCAE backend
- **Use for shorts**: Generate sounds from symbolic note patterns that map to marble events

### DCAE (Discrete Content AutoEncoder)
- **Location**: Integrated in ACE-Step and dø
- **Reference**: `/home/arlo/Data/dcae.py`
- **What it does**: Encodes audio → latent tokens, decodes latents → audio
- **Specs**: 44100 Hz, latent shape `[B, 8, 16, T]` (~86.1 fps), 128-dim flattened
- **Use for shorts**: Real-time-ish audio reconstruction from latent manipulations

### Latent Flow Models (Consistency-based generation)
- **Checkpoints**:
  - `/mnt/models/latentflow/latent_flow_consistency/`
  - `/mnt/models/latentflow/latent_flow_weighted/`
  - `/mnt/models/latentflow/latent_flow_conditioned/`
- **Inference**: `/home/arlo/test_latent_flow_inference.py`
- **What it does**: Fast generation via consistency models in DCAE latent space
- **Use for shorts**: Quick audio generation from latent space sampling

---

## MIDI & Symbolic Generation

### Melody Generator
- **Files**:
  - `/home/arlo/Data/melody_generator.py` — basic
  - `/home/arlo/Data/melody_generator_advanced.py` — advanced features
  - `/home/arlo/Data/melody_generator_proper.py` — production version
- **API endpoint**: `/api/generate-melody`
- **Use for shorts**: Generate melodies where each note triggers a marble drop or peg hit

### Chord Progression Generator
- **Files**:
  - `/home/arlo/Data/chord_progression_generator.py`
  - `/home/arlo/do-repo/harmonymodule/chord_progression_generator.py`
- **Class**: `ChordProgressionGenerator`
- **API endpoint**: `/api/render-chords`
- **Features**: Voice leading, diatonic awareness, inversions
- **Use for shorts**: Generate chord progressions, map chord changes to color shifts or scene transitions

### Harmony & Voice Generation
- **Files**:
  - `/home/arlo/do-repo/harmonymodule/melody_harmonizer.py`
  - `/home/arlo/do-repo/harmonymodule/melody_harmonizer_improved.py`
  - `/home/arlo/do-repo/harmonymodule/vocal_harmonizer.py`
  - `/home/arlo/Data/generate_improved_voices.py`
- **API endpoint**: `/api/vocal-harmonizer`
- **Use for shorts**: Layer harmonies, map multiple voices to multiple marble colors

### Drum Pattern Generator
- **Location**: `/home/arlo/do-repo/harmonymodule/drum_sampler_simple.py`
- **API endpoints**: `/api/drum-sampler/randomize`, `/api/drum-sampler/render`
- **Use for shorts**: Generate drum patterns — kicks trigger big marble drops, hi-hats trigger small bounces

### Riser Generator
- **API endpoint**: `/generate-risers`
- **Use for shorts**: Build-up sounds to create tension before a big plinko drop

### MIDI Rendering to Audio (Soundfonts & VST)
- **Soundfonts**: `/home/arlo/Data/soundfonts/` (piano, guitar, strings, brass, woodwinds, FluidR3_GM.sf2)
- **VST3 rendering**: `/home/arlo/Data/render_omnisphere.py`
- **Harmony render**: `/home/arlo/do-repo/harmonymodule/render.py`
- **Use for shorts**: Render MIDI to audio with specific instrument timbres (marimba, xylophone, kalimba = perfect for marble sounds)

---

## Audio Analysis & Feature Extraction

### Pitch Detection — Basic Pitch (Spotify)
- **Conda env**: `basicpitch-gpu`
- **Files**: `/home/arlo/Data/encodepitch.py`, `/home/arlo/Data/checkpoint_preview_single.py`
- **Output**: Piano roll, MIDI notes, pitch estimates
- **Use for shorts**: Extract pitch from generated audio → drive marble Y-position or peg layout

### Continuous Pitch Tracking — TorchCREPE
- **Usage**: `/home/arlo/Data/batch_conditioning_process.py`
- **Output**: Fundamental frequency (F0) + periodicity over time
- **Use for shorts**: Smooth pitch contour → animate marble trajectory curves

### Audio-to-MIDI
- **File**: `/home/arlo/Data/extract_and_generate_midi_vocals.py`
- **Use for shorts**: Convert any audio to MIDI events → each note = a marble/peg interaction event

### Instrument Classification
- **Files**:
  - `/home/arlo/Data/latent_instrument_classifier.py`
  - `/home/arlo/Data/latent_multilabel_classifier.py`
  - `/home/arlo/Data/binary_classifier.py`
  - `/home/arlo/Data/mix_temporal_classifier.py`
  - `/home/arlo/Data/run_mix_classifier_v2.py`, `v3.py`
- **Use for shorts**: Classify what instruments are present → assign different marble colors per instrument

---

## Stem Separation

### Demucs (Meta)
- **Models**: `htdemucs` (6-stem), `htdemucs_6s`
- **Stems**: Drums, Bass, Vocals, Guitar, Piano, Other
- **Files**:
  - `/home/arlo/Data/extract_demucs_latents.py`
  - `/home/arlo/Data/classify_demucs_stems.py`
- **API endpoint**: `/separate-stems` (supports 6-stem and 2-stem modes)
- **Use for shorts**: Separate a song into stems → each stem drives a different set of marbles (drums=red marbles, bass=blue, melody=green, etc.)

---

## Voice Conversion & Synthesis

### RVC (Retrieval-based Voice Conversion)
- **Location**: `/home/arlo/Data/RVC-WebUI/`
- **Wrapper**: `/home/arlo/Data/rvc_voice_converter.py`
- **Use for shorts**: Convert voice timbre, pitch-shift vocals

### OpenVoice V2 (Zero-shot Voice Conversion)
- **Location**: `/home/arlo/Data/OpenVoice/`
- **Wrapper**: `/home/arlo/Data/openvoice_converter.py`
- **Checkpoints**: `/home/arlo/Data/OpenVoice/checkpoints/`
- **Use for shorts**: Clone any voice without training, multilingual

### TTS / eSpeak Vocoders
- **Files**:
  - `/home/arlo/Data/espeak_from_audio.py`
  - `/home/arlo/Data/espeak_midi_vocoder.py` / `v2.py`
  - `/home/arlo/Data/espeak_world_vocoder.py` / `_aligned.py`
  - `/home/arlo/Data/midi_lyrics_tts_synthesis.py`
- **Use for shorts**: Generate robotic/vocoded speech synced to marble events

---

## Effects & Audio Transformation

### Inverse AFX (Audio Clarification)
- **Checkpoints**:
  - `/mnt/models/inverse_afx_checkpoints/` (multiple versions)
  - `/mnt/models/inverse_afx_v2/`
  - `/mnt/models/clarifier_checkpoints/` (instrument-specific)
- **API endpoint**: `/clarify-audio`
- **Use for shorts**: Clean up generated audio before final mix

### Pedalboard Effects (Spotify)
- **Files**: `/home/arlo/Data/complete_audio_processor.py`, `/home/arlo/Data/demo_builtin_effects.py`
- **Effects**: Reverb, delay, distortion, EQ, compression, etc.
- **API endpoint**: `/api/download-with-fx`
- **Use for shorts**: Add reverb/effects to marble bounce sounds

### Time/Pitch Manipulation (SOX + WORLD)
- **File**: `/home/arlo/Data/genfrominterface.py`
- **Functions**: `apply_tape_speed_sox()`, `apply_pitch_shift_sox()`, `apply_time_stretch_sox()`
- **Use for shorts**: Time-stretch or pitch-shift audio to match animation duration

---

## SoundSpace (Latent Sound Exploration)
- **Checkpoints**: `/home/arlo/soundspace_checkpoints/`
- **Data**: `/mnt/models/soundspace_data/`
- **API**: port 8096 (`/space/*`)
- **What it does**: Navigate a latent space of sounds, interpolate between timbres
- **Use for shorts**: Morph marble collision sounds through timbre space as they descend

---

## MIDI Analysis & Search

### Chord Extraction & Organization
- `/home/arlo/Data/midi_chord_extractor.py` — extract chords from MIDI
- `/home/arlo/Data/chord_audio_extractor.py` — extract chord audio
- `/home/arlo/Data/chord_organizer.py` — organize chord database
- `/home/arlo/Data/browse_organized_chords.py` — browse chords

### Audio/MIDI Search
- `/home/arlo/Data/note_search_interface.py`
- `/home/arlo/Data/midi_audio_search_interface.py`
- `/home/arlo/Data/enhanced_note_search_interface.py`
- `/home/arlo/Data/note_search_engine.py`
- **Use for shorts**: Find existing audio samples matching specific pitch patterns

---

## Training Infrastructure (for custom models)

### Training Pipelines
- `/home/arlo/Data/trainer_performerCN2.py` — latest ACE-Step trainer
- `/home/arlo/Data/train.py` — MusicGen-based training
- `/home/arlo/Data/dataloader.py` — main training data loader

### Conditioning System
- `/home/arlo/Data/conditioning_encoder.py` — extract conditioning features
- `/home/arlo/Data/conditioning_encodervox.py` — voice-specific conditioning
- **Conditioning types**: Piano rolls, amplitude envelopes, pitch contours, speaker embeddings, instrument tokens

### Cached Data
- `/mnt/models/latent_cache/` — 389k+ precomputed DCAE latents
- `/mnt/models/batch_latents/` — batch latent data
- `/mnt/models/batch_conditioning/` — batch conditioning data

---

## Conda Environments
- `ace_step` — ACE-Step model (DCAE, generation)
- `basicpitch-gpu` — Basic Pitch (ONNX pitch detection)

## Key Frameworks (installed)
- PyTorch 2.1.2 + torchaudio
- HuggingFace Transformers 4.36.2
- audiocraft 1.1.0 (MusicGen/AudioGen)
- demucs 4.0.1 (stem separation)
- basic-pitch 0.2.6 (pitch detection)
- librosa 0.10.1 (audio analysis)
- pretty_midi 0.2.10 (MIDI manipulation)
- mido 1.3.2 (MIDI I/O)
- encodec 0.1.1 (audio codec)
- pedalboard (audio effects)
- torchcrepe (pitch tracking)

---

## Ideas: Marble/Plinko + Music Integration

Here are concrete ways to wire these tools into the animation system:

1. **Note-triggered drops**: Generate a melody → each MIDI note onset drops a marble, pitch maps to X-position (low notes = left, high = right)
2. **Stem-driven multi-marble**: Separate a song with Demucs → drums drive red marbles, bass drives blue, melody drives green, each stem's amplitude controls drop timing
3. **Pitch-reactive pegs**: Use TorchCREPE to extract F0 contour → peg positions shift vertically with pitch over time
4. **Collision sounds**: Each marble-peg collision triggers a synthesized note (soundfont xylophone/marimba) at the pitch corresponding to the peg's Y-position
5. **Chord color mapping**: Extract chord progressions → change marble/background palette on chord changes
6. **Generated soundtrack**: Use ACE-Step or MusicGen to generate a 40s track → analyze it with Basic Pitch → use the note events to choreograph marble drops
7. **Drum-synced plinko**: Generate drum patterns → kick = big marble, snare = medium, hi-hat = small marble, each with corresponding collision sounds
8. **Latent space morphing**: Use SoundSpace to smoothly morph collision sounds from top (bright/high) to bottom (deep/low) as marbles descend
