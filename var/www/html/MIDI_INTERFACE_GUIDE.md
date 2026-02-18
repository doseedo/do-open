# 🎹 MIDI Search Interface - Usage Guide

## Quick Start
1. Ensure you have /home/arlo/Data/midi_analysis.json (run `python /home/arlo/Data/midianal.py` first)
2. Install requirements: `pip install gradio pandas numpy matplotlib seaborn pretty-midi music21`
3. Launch interface: `python /home/arlo/Data/midi_search_interface.py`
4. Open browser to: http://localhost:7860

## Search Features

### 🎵 Search by Note
- Enter note names: C, F#, Bb, etc.
- Toggle exact match for precise results
- Results show note occurrence count

### 🎶 Search by Chord
- Enter chord names: "C major", "Am", "F# minor"
- Finds files containing chord progressions
- Shows timing of chord occurrences

### 🎼 Search by Key
- Enter key signatures: "C", "Am", "F# major"
- Uses automatic key detection algorithm
- Based on pitch content analysis

### 🎚️ Pitch Range Search
- Use MIDI note numbers (60 = Middle C)
- Find files within specific ranges
- Useful for instrument-specific searches

### 📊 Statistics
- Collection overview and distributions
- Most common keys and notes
- Total files and note counts

## Example Searches
- Notes: "C", "F#", "Bb"
- Chords: "C major", "Am", "G7", "Dm"
- Keys: "C major", "A minor", "F# major"
- Ranges: 60-72 (middle octave), 36-48 (bass range)

## Features Overview

✨ **Smart Search Capabilities**
- 🎵 Individual note detection across all MIDI files
- 🎶 Chord progression analysis and identification
- 🎼 Automatic key signature detection
- 🎚️ Flexible pitch range filtering
- 📊 Comprehensive collection statistics

🔍 **Advanced Analysis**
- Harmonic content analysis using music theory
- Real-time chord identification from note combinations
- Key detection based on pitch distribution patterns
- Instrument-specific pitch range categorization

🎯 **User-Friendly Interface**
- Clean web-based Gradio interface
- Tabbed organization for different search types
- Real-time search results with detailed information
- Musical context for each result (key, duration, instruments)

## Technical Details
- Uses pretty-midi for MIDI parsing
- music21 for harmonic analysis
- Gradio for web interface
- Automatic key detection via pitch analysis
- Chord identification from harmonic content

## Installation & Launch

```bash
# Install dependencies
pip install -r /home/arlo/Data/midi_interface_requirements.txt

# Quick launch (checks requirements automatically)
python /home/arlo/Data/launch_midi_interface.py

# Manual launch
python /home/arlo/Data/midi_search_interface.py
```

## Files Created
- `midi_search_interface.py` - Main interface script
- `launch_midi_interface.py` - Launcher with dependency checks
- `midi_interface_requirements.txt` - Required packages
- `MIDI_INTERFACE_GUIDE.md` - This guide

Your MIDI search interface is ready! 🎹✨