# VST Audio Processor - Complete Setup

## Summary

I've set up a complete audio processing system for you with VST plugin support and built-in effects. The system is **fully functional right now** with built-in effects, and can be extended to work with VST plugins.

## Important Note About Your VintageVerb Plugin

Your VintageVerb plugin at:
```
/home/arlo/Data/plugin/ValhallaDSP.ValhallaVintageVerb.v2.0.2-macOS-CASHMERE copy
```

**This is a macOS plugin and will NOT work on Linux.** However, I've created:
1. A working audio processor with built-in reverb that produces similar results
2. Scripts to help you use Windows VST plugins via yabridge
3. Presets that simulate various reverb and effects chains

## Files Created

### Main Scripts
1. **complete_audio_processor.py** - Complete audio processor with presets and VST support
2. **vst_processor.py** - VST plugin loader and processor
3. **demo_builtin_effects.py** - Simple demo with built-in reverb
4. **DEMO.sh** - Automated demo script
5. **setup_yabridge.sh** - Script to install yabridge for Windows VST support

### Documentation
- **VST_SETUP_GUIDE.md** - Complete setup guide
- **README_VST_AUDIO_PROCESSOR.md** - This file

### Test Files
- **test_input.wav** - Test audio (440 Hz tone)
- **test_output_reverb.wav** - Processed with reverb
- **demo_*.wav** - Various processed examples

## Quick Start

### 1. Process Audio with Built-in Presets (Works Now!)

```bash
# List available presets
python3 complete_audio_processor.py --list-presets

# Process with vintage reverb (similar to VintageVerb)
python3 complete_audio_processor.py -i your_audio.wav -o output.wav --preset vintage_reverb

# Other presets
python3 complete_audio_processor.py -i your_audio.wav -o output.wav --preset hall_reverb
python3 complete_audio_processor.py -i your_audio.wav -o output.wav --preset dream_space
python3 complete_audio_processor.py -i your_audio.wav -o output.wav --preset tape_echo
python3 complete_audio_processor.py -i your_audio.wav -o output.wav --preset lo_fi
python3 complete_audio_processor.py -i your_audio.wav -o output.wav --preset warm_compress
```

### 2. Run the Demo

```bash
./DEMO.sh
```

This creates 6 different processed versions of a test audio file.

### 3. Use Your Own Audio

```bash
# Replace 'your_song.wav' with your audio file
python3 complete_audio_processor.py -i your_song.wav -o your_song_reverb.wav --preset vintage_reverb
```

## Available Presets

### vintage_reverb
Simulates classic reverb units like VintageVerb:
- Large room reverb
- Subtle chorus for warmth
- Gentle compression

### hall_reverb
Large concert hall reverb:
- Very large room size
- Lower damping for brightness
- Limiter to control peaks

### tape_echo
Vintage tape delay effect:
- Dotted eighth note delay
- Bitcrushing for tape warmth
- High-pass filter

### dream_space
Ambient soundscape effect:
- Large reverb
- Chorus and delay
- Creates ethereal spaces

### lo_fi
Vintage/degraded sound:
- 8-bit bit crushing
- Low-pass filter
- Retro character

### warm_compress
Professional mastering compression:
- Gentle compression
- Makeup gain
- Limiting

## Using Real VST Plugins

### For Windows VST Plugins (via yabridge):

1. **Install yabridge:**
   ```bash
   ./setup_yabridge.sh
   source ~/.bashrc
   ```

2. **Get Windows VST plugins:**
   - Download Windows version of VintageVerb or other plugins
   - Place in `~/vst-plugins-windows/`

3. **Configure yabridge:**
   ```bash
   yabridgectl add ~/vst-plugins-windows
   yabridgectl sync
   ```

4. **Use with processor:**
   ```bash
   python3 vst_processor.py -i input.wav -o output.wav \
       -p ~/.vst3/yabridge/YourPlugin.vst3
   ```

### For Linux Native VST3 Plugins:

```bash
# Find VST3 plugins
python3 vst_processor.py --find-plugins

# Show plugin parameters
python3 vst_processor.py -p /path/to/plugin.vst3 --show-params

# Process audio
python3 vst_processor.py -i input.wav -o output.wav \
    -p /path/to/plugin.vst3 --param mix=0.5
```

## Python API Usage

You can also use these tools in your own Python scripts:

```python
from pedalboard import Pedalboard, Reverb, Delay, Chorus
from pedalboard.io import AudioFile

# Create effect chain
board = Pedalboard([
    Reverb(room_size=0.8, damping=0.7, wet_level=0.3),
    Chorus(rate_hz=0.5, depth=0.15, mix=0.2),
])

# Process audio
with AudioFile('input.wav') as f:
    audio = f.read(f.frames)
    processed = board(audio, f.samplerate)

# Save result
with AudioFile('output.wav', 'w', f.samplerate, audio.shape[0]) as f:
    f.write(processed)
```

### With VST Plugins:

```python
from pedalboard import load_plugin
from pedalboard.io import AudioFile

# Load VST3 plugin
vst = load_plugin('/path/to/plugin.vst3')

# Set parameters
vst.mix = 0.5
vst.decay = 2.0

# Process
with AudioFile('input.wav') as f:
    audio = f.read(f.frames)
    processed = vst(audio, f.samplerate)

with AudioFile('output.wav', 'w', f.samplerate, audio.shape[0]) as f:
    f.write(processed)
```

## Supported Formats

### Input/Output Audio:
- WAV
- AIFF
- FLAC
- MP3 (with appropriate codecs)
- OGG

### VST Plugins:
- ✅ VST3 (on Linux, macOS, Windows)
- ✅ Audio Units (macOS only)
- ❌ VST2 (.so files - not supported by pedalboard)
- ❌ macOS plugins on Linux

## Built-in Effects Available

Pedalboard includes many high-quality built-in effects:

- **Reverb** - Room/hall reverb
- **Delay** - Digital delay/echo
- **Chorus** - Chorus effect
- **Phaser** - Phaser effect
- **Distortion** - Distortion/overdrive
- **Compressor** - Dynamic range compression
- **Limiter** - Peak limiting
- **Gain** - Volume adjustment
- **Bitcrush** - Bit depth reduction
- **LadderFilter** - Moog-style filter
- **HighpassFilter** - High-pass filter
- **LowpassFilter** - Low-pass filter
- And more...

## Performance Tips

1. **Sample Rate**: Higher sample rates (96kHz) provide better quality but slower processing
2. **Buffer Size**: Process in chunks for very long files
3. **Effect Order**: Order matters - usually: EQ → Compression → Reverb → Limiting
4. **Parameter Ranges**: Most parameters are 0-1 or in appropriate units (Hz, dB, ms)

## Troubleshooting

### "Failed to load plugin"
- Make sure it's a VST3 plugin (.vst3)
- Linux cannot load macOS or Windows plugins directly
- Use yabridge for Windows plugins

### "Module not found: pedalboard"
```bash
pip install pedalboard
```

### Audio sounds distorted
- Reduce wet/dry mix
- Lower gain values
- Add a limiter at the end of the chain

### VST plugin has no effect
- Check parameter values with `--show-params`
- Some plugins need specific parameter names
- Try adjusting mix/dry/wet parameters

## Next Steps

1. **Immediate**: Use the built-in presets on your audio
2. **Short-term**: Install free Linux VST3 plugins
3. **Long-term**: Set up yabridge for Windows VST plugins

## Resources

- Pedalboard Documentation: https://spotify.github.io/pedalboard/
- Yabridge: https://github.com/robbert-vdh/yabridge
- Free VST Plugins:
  - Dragonfly Reverb: https://github.com/michaelwillis/dragonfly-reverb
  - Airwindows: https://github.com/airwindows/airwindows
  - Vital Synth: https://vital.audio/

## Examples

```bash
# Process a vocal track
python3 complete_audio_processor.py -i vocals.wav -o vocals_verb.wav --preset vintage_reverb

# Process a guitar track
python3 complete_audio_processor.py -i guitar.wav -o guitar_echo.wav --preset tape_echo

# Create ambient textures
python3 complete_audio_processor.py -i synth.wav -o synth_space.wav --preset dream_space

# Add professional compression
python3 complete_audio_processor.py -i mix.wav -o mix_compressed.wav --preset warm_compress
```

## Status

✅ Audio processing system - **WORKING**
✅ Built-in effects - **WORKING**
✅ Preset system - **WORKING**
✅ Test audio generation - **WORKING**
⚠️  VST3 plugin loading - **READY** (needs VST3 plugins)
⚠️  Windows VST support - **READY** (needs yabridge installation)
❌ macOS VintageVerb - **NOT COMPATIBLE** (need Windows version + yabridge)

---

**Created:** October 25, 2025
**System:** Linux (Debian)
**Python:** 3.9
**Library:** Pedalboard 0.9.19
