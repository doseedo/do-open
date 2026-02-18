# 🎵 MIDI Note Search Interface - Complete Guide

## The Solution You Asked For

**Problem**: Inaccurate chord detection giving unreadable results like "F#unknown -> C#unknown -> Dunknown..."

**Solution**: Direct note search that finds actual MIDI notes (C, E, G) and creates audio snippets where they occur together.

## Launch Interface

```bash
python /home/arlo/Data/launch_note_search.py
```

**Access at**: http://localhost:7863

## How It Works

### 1. Select Notes
- ✅ **Check boxes** for notes you want to find
- ✅ **Example**: C, E, G (finds C major chord notes)
- ✅ **Flexible**: Select any combination (single notes, intervals, chords)

### 2. Search Actual MIDI Data
- ✅ **Analyzes actual note content** in MIDI files
- ✅ **Finds time windows** where selected notes play together
- ✅ **No reliance on inaccurate chord labels**

### 3. Get Audio Snippets
- ✅ **2-4 second audio clips** of the exact moments
- ✅ **Shows which notes were found** (may include additional notes)
- ✅ **Precise timing** within the original files

## Example Searches

### C Major Chord
**Select**: C, E, G
**Finds**: Sections where all three notes play together
**Result**: Audio snippets of actual C major chords

### Single Note
**Select**: C
**Finds**: Any section with C notes
**Result**: All occurrences of C across your collection

### Interval Study
**Select**: C, G
**Finds**: Perfect fifth intervals
**Result**: Audio examples of C-G combinations

### Complex Chords
**Select**: G, B, D, F
**Finds**: G7 chord occurrences
**Result**: Audio snippets of dominant seventh chords

## Interface Features

### Note Selection
- 📋 **Checkbox interface** for all 12 notes
- 🎯 **Default selection**: C, E, G (C major)
- ⚡ **Instant search** after selection

### Smart Search
- 🔍 **Analyzes 2-second windows** throughout each file
- 🎵 **Finds simultaneous note occurrences**
- 📊 **Limits results** to best examples

### Audio Playback
- 🔊 **Instant playback** of found sections
- 📋 **Playlist dropdown** to browse different examples
- 📍 **Timing information** showing when notes occur

### Detailed Results
For each snippet:
```
🎹 Original file name
🎵 Notes found: C, E, G, A, D
🎼 Instrument - Session
⏱️ 3.0s (5.5s - 8.5s in original)
```

## What Makes This Accurate

### ❌ Old Chord Detection Problems
- Relied on algorithmic chord guessing
- Produced labels like "Cunknown", "F#unknown"
- No way to verify accuracy

### ✅ New Note Search Benefits
- **Direct MIDI note analysis** - sees actual notes played
- **Time-based windows** - finds when notes occur together
- **Audio verification** - hear the actual results
- **Flexible search** - any note combination

## File Organization

Results saved to:
```
/home/arlo/Data/note_search_results/
└── C_E_G/                          # Selected notes
    ├── playlist.json                # Search results metadata
    ├── filename_section01_5.5s.wav  # Audio snippet
    ├── filename_section01_5.5s.mid  # MIDI snippet
    └── filename_section01_5.5s.json # Snippet details
```

## Real Example Result

**Search for**: C, E, G
**Found in**: 121.62_32.mid
**Section 1**: 5.5s - 8.5s
**Notes found**: C, E, G, A, D (includes target notes plus extras)
**Audio**: 3-second snippet of actual chord

## Available Interfaces Summary

| Interface | Purpose | Port | Command |
|-----------|---------|------|---------|
| **Note Search** | Find actual notes | 7863 | `python launch_note_search.py` |
| Audio Search | Chord audio snippets | 7862 | `python launch_audio_interface.py` |
| Text Search | Text results & organization | 7861 | `python midi_search_interface.py` |

## Quick Start

1. **Launch**: `python /home/arlo/Data/launch_note_search.py`
2. **Open**: http://localhost:7863
3. **Select notes**: Check C, E, G (or any combination)
4. **Click search**: Wait for results (~30 seconds)
5. **Listen**: Play audio snippets of found note combinations
6. **Browse**: Use dropdown to hear different examples

## Benefits Over Previous System

✅ **Accurate**: Searches actual MIDI note data
✅ **Verifiable**: Listen to what was actually found
✅ **Flexible**: Any note combination possible
✅ **Fast**: Results in 30 seconds
✅ **Musical**: Hear harmonic context
✅ **Precise**: Exact timing information

---

**Now you can find and hear actual note combinations in your MIDI collection!** 🎵✨