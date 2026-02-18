# 🔊 MIDI Audio Search Interface Guide

## Overview

The new MIDI Audio Search Interface solves the problem of unreadable chord progression output by creating **playable audio snippets** for each detected chord occurrence.

## What's Different

### ❌ Old Way (Text Only)
```
F#unknown -> C#unknown -> Dunknown -> Dunknown -> Dunknown -> Dunknown...
```
- Overwhelming text output
- No way to hear the actual chords
- Hard to understand musical context

### ✅ New Way (Audio Snippets)
```
🎵 Audio Snippet 1: E GTR - D#major (Position 36, 11.9% through song)
🔊 [Audio Player] ► Play 3.2s snippet
```
- **Playable audio snippets** of detected chord sections
- **Precise timing** - know exactly when chords occur
- **Musical context** - hear surrounding notes
- **Easy browsing** between different chord occurrences

## How to Use

### 1. Launch Audio Interface
```bash
python /home/arlo/Data/launch_audio_interface.py
```
**Access at:** http://localhost:7862

### 2. Search for Chords
1. Enter chord name (e.g., "major", "minor", "Cmajor")
2. Set max files to process (1-20)
3. Click "🔍 Search & Create Audio Snippets"

### 3. Listen to Results
- **Audio Player**: Plays current snippet automatically
- **Playlist Dropdown**: Switch between different chord occurrences
- **Snippet Info**: Shows timing, instrument, session details

## What You Get

### Audio Snippets
- **2-4 second audio clips** of detected chord sections
- **High-quality synthesis** using FluidSynth or pretty_midi
- **Context included** - hear notes before and after the chord
- **Multiple examples** from different files and positions

### Detailed Information
For each snippet:
- **Original file name** and session
- **Exact position** in the chord progression
- **Percentage** through the original song
- **Instrument** that played the chord
- **Duration** of the snippet

### Example Output
```
🎹 E GTR.01_03.mid - D#major
📍 Position 36 (11.9% through song)
🎼 E GTR - 05.02.2025_Simona_Grounded
⏱️ 3.2s
```

## Interface Features

### Smart Processing
- **Limited file processing** to avoid overwhelming results
- **Best examples first** - prioritizes files with multiple chord occurrences
- **Quick generation** - audio snippets created in seconds

### Easy Navigation
- **Playlist dropdown** to switch between snippets
- **Auto-play** when selecting new snippets
- **Visual feedback** showing current snippet info

### File Organization
Audio snippets saved to:
```
/home/arlo/Data/chord_audio_snippets/
└── major/                     # Chord name
    ├── playlist.json          # Snippet metadata
    ├── major_player.html      # Standalone HTML player
    ├── filename_pos036_D#major.wav  # Audio snippet
    ├── filename_pos036_D#major.mid  # MIDI snippet
    └── filename_pos036_D#major.json # Snippet metadata
```

## Available Interfaces

### 1. Main Search Interface (Text Results)
- **Launch:** `python /home/arlo/Data/midi_search_interface.py`
- **Access:** http://localhost:7861
- **Use for:** Quick text-based searches, file organization

### 2. Audio Search Interface (Audio Snippets)
- **Launch:** `python /home/arlo/Data/launch_audio_interface.py`
- **Access:** http://localhost:7862
- **Use for:** Listening to actual chord occurrences

## Example Workflow

### Search for "major" chords:

1. **Launch audio interface**
   ```bash
   python /home/arlo/Data/launch_audio_interface.py
   ```

2. **Search and generate snippets**
   - Enter "major" in chord field
   - Set max files to 5
   - Click search button
   - Wait for audio generation (~30 seconds)

3. **Browse and listen**
   - First snippet plays automatically
   - Use playlist dropdown to hear other examples:
     - "1. E GTR.01_03 - D#major"
     - "2. E GTR.04_06 - Emajor"
     - "3. GHOST.01_03 - Cmajor"

4. **Understand context**
   - See exact position in song
   - Know which instrument played it
   - Hear musical context around the chord

## Benefits

✅ **Actually hear** the chords instead of reading text
✅ **Quick generation** - snippets ready in seconds
✅ **Multiple examples** from different songs/instruments
✅ **Precise timing** - know exactly when chords occur
✅ **Musical context** - hear surrounding notes
✅ **Easy browsing** between different occurrences
✅ **Organized output** - files saved for future reference

## Commands Quick Reference

| Task | Command |
|------|---------|
| Launch audio interface | `python /home/arlo/Data/launch_audio_interface.py` |
| Launch text interface | `python /home/arlo/Data/midi_search_interface.py` |
| Generate snippets directly | `python /home/arlo/Data/chord_audio_extractor.py "major"` |
| Browse organized files | `python /home/arlo/Data/browse_organized_chords.py "major"` |

## Interfaces Available

- **Audio Interface:** http://localhost:7862 (🔊 Playable snippets)
- **Text Interface:** http://localhost:7861 (📝 Text results & organization)

---

Now you can **hear** your chord progressions instead of reading endless text! 🎵✨