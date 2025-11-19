# Big Band Generator V2 - Setup & Troubleshooting

## Quick Start

```bash
# 1. Navigate to the midi_generator directory
cd /path/to/Do/midi_generator

# 2. Install dependencies
pip3 install mido matplotlib

# 3. Test imports (optional but recommended)
python3 test_imports.py

# 4. Generate your first arrangement
python3 generate_big_band_v2.py my_arrangement.mid 140 0
```

## Common Issues

### Issue 1: ImportError - relative import beyond top-level package

**Error message:**
```
ImportError: attempted relative import beyond top-level package
```

**Solution:**
This error has been fixed in the latest version. Make sure you have the latest code:
```bash
git pull origin claude/analyze-midi-generator-01Uf1NpkChcni2fpeNNQZ2QM
```

The script now handles imports correctly without needing relative imports from other modules.

### Issue 2: ModuleNotFoundError: No module named 'mido'

**Error message:**
```
ModuleNotFoundError: No module named 'mido'
```

**Solution:**
Install the mido library:
```bash
pip3 install mido
```

Or if using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows

pip install mido matplotlib
```

### Issue 3: ImportError for genres/jazz or genres/funk_soul

**Error message:**
```
ImportError: Failed to import required modules
```

**Solution:**
Make sure you're running the script from the `midi_generator` directory:

```bash
# Wrong (will fail):
cd /path/to/Do
python3 midi_generator/generate_big_band_v2.py

# Correct (will work):
cd /path/to/Do/midi_generator
python3 generate_big_band_v2.py
```

### Issue 4: Script runs but produces no output or crashes

**Debugging steps:**

1. **Test imports first:**
```bash
python3 test_imports.py
```

2. **Check Python version:**
```bash
python3 --version  # Should be 3.7 or higher
```

3. **Verify file structure:**
```bash
ls -la genres/
ls -la algorithms/
```

You should see:
- `genres/jazz.py`
- `genres/funk_soul.py`
- `algorithms/` directory

## System Requirements

### Python Version
- **Minimum**: Python 3.7
- **Recommended**: Python 3.9 or higher

### Dependencies

**Required:**
- `mido` - MIDI file reading/writing
  ```bash
  pip3 install mido
  ```

**Optional:**
- `matplotlib` - For visualization (not used by generator but useful for analysis)
  ```bash
  pip3 install matplotlib
  ```

## File Structure

Make sure your directory structure looks like this:

```
midi_generator/
├── generate_big_band_v2.py    # Main generator script
├── test_imports.py             # Import testing script
├── genres/
│   ├── __init__.py
│   ├── jazz.py                # Jazz generation module
│   ├── funk_soul.py           # Funk/soul + Tower of Power horns
│   └── ...
├── algorithms/
│   ├── __init__.py
│   ├── groove_library.py
│   ├── rhythm_engine.py
│   └── ...
└── ...
```

## Testing Your Installation

### Step 1: Test imports
```bash
cd midi_generator
python3 test_imports.py
```

You should see:
```
Testing Big Band Generator V2 Dependencies...
============================================================

1. Testing mido library...
   ✓ mido installed successfully

2. Testing path setup...
   ✓ Working directory: /path/to/midi_generator

3. Testing genres/jazz.py...
   ✓ Jazz module imports successfully

4. Testing genres/funk_soul.py...
   ✓ Funk/Soul module imports successfully

5. Testing generator instantiation...
   ✓ JazzGenerator created successfully

6. Testing funk generator instantiation...
   ✓ FunkSoulGenerator created successfully

============================================================
✅ ALL TESTS PASSED!
============================================================
```

### Step 2: Generate a test arrangement
```bash
python3 generate_big_band_v2.py test.mid 140 0
```

You should see output like:
```
======================================================================
PROFESSIONAL BIG BAND ARRANGEMENT GENERATOR V2
======================================================================

🎺 Generating Professional Big Band Arrangement
   Tempo: 140 BPM
   Key: C
======================================================================

   🎵 INTRO
   🎵 HEAD
   🎵 SAX_SOLI
   🎵 SOLO
   🎵 SHOUT_CHORUS
   🎵 HEAD
   🎵 OUTRO

======================================================================
✅ Arrangement complete: 84 bars

💾 Exporting to MIDI: test.mid
   ✓ Lead Alto Sax: XX notes
   ✓ Alto Sax 1: XX notes
   ...
```

### Step 3: Verify MIDI file
```bash
ls -lh test.mid
```

The file should be created (typically 10-50 KB).

## Platform-Specific Notes

### macOS
```bash
# If you get SSL certificate errors:
pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org mido

# If using Homebrew Python:
/usr/local/bin/python3 -m pip install mido
```

### Linux
```bash
# May need to install Python pip first:
sudo apt-get install python3-pip  # Ubuntu/Debian
sudo yum install python3-pip       # CentOS/RHEL

# Then install mido:
pip3 install mido
```

### Windows
```powershell
# Use Command Prompt or PowerShell:
py -3 -m pip install mido

# Run the script:
py -3 generate_big_band_v2.py
```

## Performance Tips

### Generation Speed
- **Normal**: 5-10 seconds for full 84-bar arrangement
- **Slow**: If it takes longer, check:
  - Python version (3.9+ is faster)
  - Available RAM (should have 100MB+ free)
  - Disk space for output

### Output File Size
- **Expected**: 10-50 KB
- **Large files**: If >1 MB, there may be duplicate notes

## Advanced Usage

### Custom Arrangement Structure

Edit the `sections` list in `generate_big_band_v2.py` around line 115:

```python
sections = [
    SectionConfig(ArrangementSection.INTRO, 1, "moderate", "none"),
    SectionConfig(ArrangementSection.HEAD, 2, "sparse", "melody"),  # Change to 2 choruses
    SectionConfig(ArrangementSection.SOLO, 2, "sparse", "background", "trumpet"),  # Trumpet solo
    SectionConfig(ArrangementSection.SHOUT_CHORUS, 1, "heavy", "soli"),
    SectionConfig(ArrangementSection.OUTRO, 1, "heavy", "soli"),
]
```

### Different Solo Instruments

Options for `solo_instrument` parameter:
- `"trumpet"` - Trumpet solo
- `"tenor_sax"` - Tenor sax solo
- `"alto_sax"` - Alto sax solo
- `"trombone"` - Trombone solo
- `"piano"` - Piano solo

### Brass Activity Levels

Options for `brass_activity`:
- `"none"` - No brass (0%)
- `"sparse"` - Occasional hits (10-20%)
- `"moderate"` - Regular patterns (40-50%)
- `"heavy"` - Constant brass (80-90%)

### Sax Activity Types

Options for `sax_activity`:
- `"none"` - Saxes rest
- `"melody"` - Lead alto only (solo)
- `"soli"` - Full 5-part sax section
- `"background"` - Sparse pads

## Getting Help

If you're still having issues:

1. **Check the error message carefully** - It usually tells you what's missing

2. **Run the test script:**
   ```bash
   python3 test_imports.py
   ```

3. **Verify you're in the right directory:**
   ```bash
   pwd  # Should end with /midi_generator
   ```

4. **Check Python version:**
   ```bash
   python3 --version
   ```

5. **Try with full paths:**
   ```bash
   /usr/bin/python3 /full/path/to/generate_big_band_v2.py
   ```

## Still Not Working?

Create an issue with:
1. Your operating system
2. Python version (`python3 --version`)
3. Full error message
4. Output of `python3 test_imports.py`

## Success Checklist

- [ ] Python 3.7+ installed
- [ ] mido library installed (`pip3 install mido`)
- [ ] In the `midi_generator` directory
- [ ] `test_imports.py` passes all tests
- [ ] Can generate test.mid successfully
- [ ] MIDI file opens in your DAW/notation software

If all checkboxes are checked, you're ready to create professional big band arrangements! 🎺🎷
