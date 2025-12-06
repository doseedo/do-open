import sys
sys.path.insert(0, '.')
from scripts.run_factored_pipeline import load_midi_factored
import glob

# Load one file and check octaves
files = glob.glob('/home/arlo/do-repo/midi_generator/midi_corpus/big_band/*.mid')[:1]
tracks = load_midi_factored(files[0])
if tracks:
    track = tracks[0]
    print(f'Track has {len(track)} notes')
    print(f'pitch_classes[:10]: {list(track.pitch_classes[:10])}')
    print(f'octaves[:10]: {list(track.octaves[:10])}')
    # Check if there are any different octaves for same pitch class
    for i in range(min(50, len(track) - 1)):
        if track.pitch_classes[i] == track.pitch_classes[i+1]:
            print(f'Same pitch class at {i},{i+1}: pc={track.pitch_classes[i]}, octaves={track.octaves[i]},{track.octaves[i+1]}')
            if track.octaves[i] != track.octaves[i+1]:
                print('  -> DIFFERENT octaves!')
