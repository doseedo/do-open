# 🎹 MIDI Chord Organization System

## Overview

The MIDI search interface now includes a powerful chord organization feature that creates manageable folder structures instead of overwhelming chord progression lists.

## How It Works

### 1. Regular Search
- Use the main search interface at http://localhost:7861
- Go to "🎶 Search by Chord" tab
- Enter chord name (e.g., "major", "minor", "Cmajor")
- Click "Search Chords" for regular list results

### 2. Organized Results
- Same tab, but click "Search & Organize Files"
- Creates organized folder structure at `/home/arlo/Data/chord_organized/`
- Groups files by recording session
- Provides detailed analysis for each file

## Output Structure

```
chord_organized/
└── major/                          # Chord name
    ├── INDEX.txt                   # Overview of all results
    └── session_SessionName/        # Grouped by session
        ├── filename.mid            # Original MIDI file
        ├── filename_info.json      # Detailed metadata
        └── filename_summary.txt    # Human-readable analysis
```

## What You Get

### INDEX.txt
- Overview of all found files
- Summary by session
- File counts and durations
- Directory structure map

### Summary Files (filename_summary.txt)
- **Target chord occurrences**: Exact positions where chord appears
- **Timing information**: Percentage through the song
- **Context**: Surrounding chords for each occurrence
- **Full progression**: First 100 chords of the song
- **Metadata**: Session, instrument, duration details

### Info Files (filename_info.json)
- Complete technical metadata
- All chord positions with timestamps
- Original file paths
- Session information

## Browsing Results

### Command Line Browser
```bash
# Browse specific chord results
python /home/arlo/Data/browse_organized_chords.py major

# List all organized chord results
python /home/arlo/Data/browse_organized_chords.py --list
```

### Direct File Access
```bash
# Navigate to results
cd /home/arlo/Data/chord_organized/major/

# Read analysis
cat session_*/filename_summary.txt

# Play MIDI file
fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 session_*/filename.mid
```

## Example Workflow

1. **Search for "major" chords**:
   - Interface finds 31,667 files
   - Click "Search & Organize Files"
   - System processes top 20 files

2. **Results created**:
   ```
   chord_organized/major/
   ├── INDEX.txt
   └── session_05.02.2025_Simona_Grounded/
       ├── E GTR.01_03.mid
       ├── E GTR.01_03_summary.txt
       └── E GTR.01_03_info.json
   ```

3. **Read analysis**:
   ```
   Target Chord: major
   File: E GTR.01_03.mid
   Duration: 602.60 seconds

   CHORD OCCURRENCES:
   Position  20: D#major (6.6% through song)
   Position  36: D#major (11.9% through song)
   Position 155: Emajor (51.3% through song)
   ```

4. **Listen to specific sections**:
   - Use timing info to jump to chord positions
   - Play full file or use audio editing software to extract sections

## Benefits

✅ **Manageable Results**: No more overwhelming chord progression strings
✅ **Session Organization**: Files grouped by recording session
✅ **Precise Timing**: Know exactly when chords occur
✅ **Context Awareness**: See surrounding chords for musical context
✅ **Easy Browsing**: Simple file structure for exploration
✅ **Multiple Formats**: JSON for data, TXT for humans

## Commands Summary

| Task | Command |
|------|---------|
| Launch interface | `python /home/arlo/Data/midi_search_interface.py` |
| Organize chord results | Click "Search & Organize Files" in interface |
| Browse results | `python /home/arlo/Data/browse_organized_chords.py <chord>` |
| List all organized | `python /home/arlo/Data/browse_organized_chords.py --list` |
| Direct organization | `python /home/arlo/Data/chord_organizer.py <chord>` |

## Interface Access

**Web Interface**: http://localhost:7861
**Results Directory**: `/home/arlo/Data/chord_organized/`

---

Your MIDI chord search is now organized and manageable! 🎵✨