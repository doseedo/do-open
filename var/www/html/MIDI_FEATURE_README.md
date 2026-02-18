# MIDI Voice Generation Feature

This document describes the new MIDI upload and voice generation functionality added to `gen_from_web2.py`.

## Features Added

### 1. MIDI File Upload and Processing
- **File Support**: Accepts `.mid` and `.midi` files
- **MIDI Parsing**: Extracts note events from all tracks and channels
- **Voice Separation**: Intelligently separates MIDI notes into multiple voices (1-8 voices)

### 2. Voice Leading Optimization
- **Chord Analysis**: Groups notes occurring within small time windows into chords
- **Voice Leading**: Assigns notes to voices to minimize voice leading distance
- **Range Awareness**: Uses appropriate pitch ranges for different voices (SATB defaults)

### 3. Audio Conditioning Cache
- **Smart Caching**: Caches extracted conditioning data based on file hash
- **Persistent Storage**: Saves cache in `./conditioning_cache/` directory
- **Automatic Validation**: Verifies cached files still exist before reuse

### 4. FluidSynth Audio Rendering
- **MIDI to Audio**: Renders each voice to audio using FluidSynth
- **Fallback Synthesis**: Basic sine wave synthesis if no soundfont found
- **Configurable Sample Rate**: Renders at 24kHz for EnCodec compatibility

### 5. Separate Voice Generation
- **Independent Processing**: Each voice generates separately with unique seeds
- **MIDI Piano Roll**: Uses MIDI-derived piano roll instead of audio-extracted
- **Audio Conditioning**: Uses uploaded audio file for other conditioning streams

### 6. Playback and Combination
- **Individual Voices**: Can generate and save each voice separately
- **Voice Combination**: Option to combine all voices into single output
- **Automatic Mixing**: Normalizes combined audio to prevent clipping

## UI Changes

### New Tab: "🎹 MIDI Voice Generation"
- **MIDI Upload**: File upload widget for MIDI files
- **Audio Conditioning**: Audio upload for style conditioning
- **Voice Controls**:
  - Number of voices (1-8)
  - Separate voice generation toggle
  - Combine outputs toggle
- **All Generation Parameters**: Same controls as audio-only generation

### Existing Tab: "🎧 Audio-only Generation"
- **Unchanged**: Original functionality preserved
- **Same Interface**: All existing controls work as before

## How It Works

### 1. MIDI Processing Pipeline
```
MIDI File → Parse Tracks → Extract Notes → Group Chords → Assign to Voices
```

### 2. Voice Generation Pipeline
```
Voice Notes → Piano Roll → FluidSynth Audio → Extract Conditioning → Generate
```

### 3. Caching System
```
Audio File → Hash → Check Cache → Extract/Load Conditioning → Cache Result
```

## Usage Examples

### MIDI-Only Generation (No Audio Required)
1. Upload a MIDI file
2. Leave audio conditioning empty
3. Select instrument group/subgroup
4. Click "🎼 Generate from MIDI"

### MIDI + Audio Conditioning
1. Upload a MIDI file
2. Upload an audio file for conditioning style
3. Select instrument group/subgroup
4. Click "🎼 Generate from MIDI"

### Multi-Voice Generation (MIDI-Only)
1. Upload MIDI file (no audio needed)
2. Set "Number of Voices" to 4
3. Enable "Separate Voice Generation"
4. Enable "Combine Voice Outputs"
5. Generate to get all 4 voices mixed together

### Multi-Voice with Audio Conditioning
1. Upload MIDI and audio files
2. Set "Number of Voices" to 4
3. Enable "Separate Voice Generation"
4. Enable "Combine Voice Outputs"
5. Generate to get all 4 voices mixed together

### Single Voice with MIDI Piano Roll
1. Upload MIDI file (audio optional)
2. Disable "Separate Voice Generation"
3. Generate to use first voice's piano roll

## Technical Details

### Dependencies Added
- `mido`: MIDI file parsing
- `fluidsynth`: Audio synthesis
- `scipy`: Audio I/O utilities
- `hashlib`, `pickle`: Caching system

### Key Functions
- `parse_midi_file()`: MIDI parsing and note extraction
- `separate_voices()`: Voice separation with voice leading
- `process_midi_upload()`: Complete MIDI processing pipeline
- `generate_voice_separately()`: Single voice generation
- `generate_all_voices()`: Multi-voice generation
- `combine_voice_outputs()`: Audio mixing and combination

### Performance Optimizations
- **Conditioning Cache**: Avoids re-extracting same audio files
- **Parallel Processing**: Can process multiple voices
- **Memory Efficient**: Processes voices separately to reduce memory usage

## Future Enhancements

Potential improvements for future versions:
- **Soundfont Selection**: UI control for different instrument soundfonts
- **Voice-Specific Instruments**: Different instruments per voice
- **MIDI Export**: Export generated voices back to MIDI
- **Real-time Preview**: Preview voice separation before generation
- **Advanced Voice Leading**: More sophisticated voice leading rules