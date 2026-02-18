import sys
import torch
from pathlib import Path
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio

# ARGS
audio_path = Path(sys.argv[1])
session_rel = Path(sys.argv[2])
OUTPUT_ROOT = Path("/home/arlo/Data/encodec_tokens")

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)
model.eval()

# Load audio
try:
    waveform, sr = torchaudio.load(audio_path)
    if sr != 24000:
        waveform = convert_audio(waveform, sr, model.sample_rate, model.channels)

    with torch.no_grad():
        tokens = model.encode(waveform.unsqueeze(0).to(device))

    out_dir = OUTPUT_ROOT / session_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (audio_path.stem + ".pt")
    torch.save(tokens, out_path)
    print(f"✅ Saved: {out_path}")
except Exception as e:
    print(f"❌ Worker error: {audio_path}: {e}")
finally:
    del waveform, tokens, model
    torch.cuda.empty_cache()
