#!/bin/bash
# Audio Processing Demo
# Shows all features of the VST processor

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        VST Plugin Audio Processor - Demo                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Create test audio
echo "📝 Step 1: Creating test audio..."
python3 complete_audio_processor.py --create-test demo_input.wav
echo ""

# Process with different presets
echo "🎛️  Step 2: Processing with different presets..."
echo ""

echo "   Preset 1: Vintage Reverb (simulates VintageVerb)..."
python3 complete_audio_processor.py -i demo_input.wav -o demo_vintage_reverb.wav --preset vintage_reverb

echo "   Preset 2: Hall Reverb..."
python3 complete_audio_processor.py -i demo_input.wav -o demo_hall_reverb.wav --preset hall_reverb

echo "   Preset 3: Tape Echo..."
python3 complete_audio_processor.py -i demo_input.wav -o demo_tape_echo.wav --preset tape_echo

echo "   Preset 4: Dream Space..."
python3 complete_audio_processor.py -i demo_input.wav -o demo_dream_space.wav --preset dream_space

echo "   Preset 5: Lo-Fi..."
python3 complete_audio_processor.py -i demo_input.wav -o demo_lofi.wav --preset lo_fi

echo "   Preset 6: Warm Compress..."
python3 complete_audio_processor.py -i demo_input.wav -o demo_warm_compress.wav --preset warm_compress

echo ""
echo "✅ All processing complete!"
echo ""
echo "📁 Generated files:"
ls -lh demo_*.wav
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  DEMO COMPLETE                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "You can now listen to the processed files:"
echo "  - demo_input.wav (original)"
echo "  - demo_vintage_reverb.wav"
echo "  - demo_hall_reverb.wav"
echo "  - demo_tape_echo.wav"
echo "  - demo_dream_space.wav"
echo "  - demo_lofi.wav"
echo "  - demo_warm_compress.wav"
echo ""
echo "For VST plugin support, see VST_SETUP_GUIDE.md"
