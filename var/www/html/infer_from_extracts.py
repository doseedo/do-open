#!/usr/bin/env python3
import os, json, argparse, numpy as np, torch, torchaudio
from pathlib import Path
import tempfile, sys, subprocess, traceback

# import your modules
from trainer_performer import Pipeline
from conditioning_encoder import PerformanceConditionEncoder  # used inside Pipeline

# Additional imports for extraction
import librosa
import pretty_midi
from encodec import EncodecModel
from encodec.utils import convert_audio
from basic_pitch.inference import predict as basicpitch_predict
import basic_pitch

# match training grid
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000, 320
SLOW_HZ = DCAE_SR / DCAE_HOP  # ~10.7666 Hz

# Extraction configuration
SAMPLE_RATE = 44100
HOP_LENGTH = 4096
N_FFT = 8192
ENC_TARGET_SR = 24000
ENC_BANDWIDTH = 6.0

APPROVED_GROUPS = ["piano","guitar","bass","strings","brass","winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano","keys","undefined"],
    "guitar":  ["acoustic_guitar","electric_guitar","plucked","undefined"],
    "bass":    ["electric_bass","upright_bass","undefined"],
    "strings": ["violin","viola","cello","undefined"],
    "brass":   ["trumpet","trombone","french_horn","tuba","undefined"],
    "winds":   ["bassoon","clarinet","flute","oboe","sax"],
}

def make_rbend_mask(pr: np.ndarray|None, rframe: np.ndarray, amp: np.ndarray,
                    amp_thr=0.06, smooth_k=5) -> torch.Tensor:
    T = int(len(rframe))
    mask = np.ones(T, dtype=bool)

    if pr is not None and pr.size > 0:
        prb = (pr > 0)
        active = prb.sum(axis=0)
        mono = active <= 1
        # allow perfect 5ths / octaves when exactly two notes
        two = np.where(active == 2)[0]
        for t in two:
            notes = np.where(prb[:, t])[0]
            if len(notes) == 2:
                d = abs(int(notes[0]) - int(notes[1]))
                if min(abs(d-12), abs(d-7)) <= 1:
                    mono[t] = True
        if smooth_k > 1:
            pad = smooth_k // 2
            x = mono.astype(np.float32)
            x = np.pad(x, (pad, pad), mode="edge")
            filt = np.ones(smooth_k, dtype=np.float32)/smooth_k
            mono = (np.convolve(x, filt, mode="valid") > 0.5)
        mask &= mono

    mask &= (rframe > 0)
    bleed = (amp > amp_thr) & (rframe == 0)
    mask &= (~bleed)
    mask &= (amp > 0.01)

    if smooth_k > 1:
        pad = smooth_k // 2
        x = mask.astype(np.float32)
        x = np.pad(x, (pad, pad), mode="edge")
        filt = np.ones(smooth_k, dtype=np.float32)/smooth_k
        mask = (np.convolve(x, filt, mode="valid") > 0.5)

    return torch.from_numpy(mask.astype(np.float32))

# ---- add this helper ----
def _load_encodec_tokens(path):
    """Return int tensor [8, T_fast] from a variety of saved Encodec formats."""
    import torch
    obj = torch.load(path, map_location="cpu")

    # 1) Direct tensor
    if isinstance(obj, torch.Tensor):
        enc = obj
    # 2) dict with 'codes' (common in Encodec)
    elif isinstance(obj, dict) and "codes" in obj and isinstance(obj["codes"], torch.Tensor):
        enc = obj["codes"]
    # 3) nested list/tuple: find the first tensor inside
    elif isinstance(obj, (list, tuple)):
        def _first_tensor(x):
            if isinstance(x, torch.Tensor): return x
            if isinstance(x, (list, tuple)):
                for it in x:
                    t = _first_tensor(it)
                    if t is not None: return t
            if isinstance(x, dict):
                for v in x.values():
                    t = _first_tensor(v)
                    if t is not None: return t
            return None
        enc = _first_tensor(obj)
        if enc is None:
            raise RuntimeError(f"No tensor found in {path}")
    else:
        raise RuntimeError(f"Unsupported encodec format in {path}: {type(obj)}")

    # Shapes we might see:
    # [B, 8, T]  -> take batch 0
    # [8, T]     -> already good
    # [T, 8]     -> transpose
    if enc.dim() == 3 and enc.shape[0] == 1:
        enc = enc[0]
    if enc.dim() != 2:
        raise RuntimeError(f"Expected 2D or [1,8,T], got shape {tuple(enc.shape)}")
    if enc.shape[0] != 8 and enc.shape[1] == 8:
        enc = enc.transpose(0, 1)

    # make sure it’s integer-ish (codes)
    if enc.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        enc = enc.long()

    return enc  # [8, T_fast], long

def extract_audio_features(audio_path: Path, temp_dir: Path):
    """Extract all conditioning features from audio file"""
    
    # Create output paths
    stem = audio_path.stem
    out_encodec = temp_dir / f"{stem}.encodec.pt"
    out_mid = temp_dir / f"{stem}.mid"
    out_pr = temp_dir / f"{stem}.pianoroll.npy"
    out_amp = temp_dir / f"{stem}.amp.npy"
    out_rframe = temp_dir / f"{stem}.rframe.npy"
    out_rbend = temp_dir / f"{stem}.rbend.npy"
    
    print(f"Extracting features from: {audio_path}")
    
    # Load audio
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    
    # 1. Extract Encodec tokens
    print("  -> Extracting Encodec tokens...")
    save_encodec_tokens(wav, sr, out_encodec)
    
    # 2. Extract MIDI and piano roll
    print("  -> Extracting MIDI and piano roll...")
    extract_basicpitch_midi_pr(audio_path, out_mid, out_pr)
    
    # 3. Extract amplitude, rframe, rbend
    print("  -> Extracting amplitude and rhythm features...")
    extract_signal_features(wav, sr, out_amp, out_rframe, out_rbend)
    
    return {
        'encodec': out_encodec,
        'pianoroll': out_pr,
        'amp': out_amp,
        'rframe': out_rframe,
        'rbend': out_rbend
    }

def save_encodec_tokens(waveform: torch.Tensor, sr: int, out_path: Path):
    """Extract and save Encodec tokens"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(ENC_BANDWIDTH)
    model.to(device).eval()
    
    with torch.no_grad():
        wf24 = convert_audio(waveform, sr, ENC_TARGET_SR, 1)  # mono 24k
        tokens = model.encode(wf24.unsqueeze(0).to(device))
    
    torch.save(tokens, out_path)

def extract_basicpitch_midi_pr(audio_path: Path, out_mid: Path, out_pr: Path):
    """Extract MIDI and piano roll using BasicPitch"""
    onnx_model = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"
    
    # Run BasicPitch
    _, midi_pm, _ = basicpitch_predict(str(audio_path), model_or_model_path=str(onnx_model))
    midi_pm.write(str(out_mid))
    
    # Convert to piano roll
    pr_pm = pretty_midi.PrettyMIDI(str(out_mid))
    pr = pr_pm.get_piano_roll(fs=SAMPLE_RATE/HOP_LENGTH)  # (128, T)
    pr[pr > 0] = 1
    np.save(out_pr, pr.astype(np.uint8))

def extract_signal_features(wav: torch.Tensor, sr: int, out_amp: Path, out_rframe: Path, out_rbend: Path):
    """Extract amplitude, rhythm frame, and rhythm bend features"""
    y = wav[0]  # (T,)
    
    # RMS envelope @ hop=4096, win=N_FFT
    win = N_FFT
    hop = HOP_LENGTH
    if y.numel() < win:
        y = torch.nn.functional.pad(y, (0, win - y.numel()))
    frames = y.unfold(0, win, hop)  # (n_frames, win)
    amp = torch.sqrt((frames**2).mean(dim=1) + 1e-12)
    if amp.max() > 0:
        amp = amp / amp.max()
    
    # F0 with torchcrepe (CPU)
    import torchcrepe
    device = "cpu"
    y_for_crepe = y.unsqueeze(0).to(device)  # (1, T)
    
    with torch.inference_mode():
        f0, pd = torchcrepe.predict(
            y_for_crepe,
            sample_rate=sr,
            hop_length=int(HOP_LENGTH),
            fmin=65.41,      # C2
            fmax=2093.00,    # C7
            pad=True,
            model="tiny",    # memory-light
            batch_size=64,   # small batches
            device=device,
            return_periodicity=True
        )
        f0 = f0[0].cpu()           # Hz
        periodicity = pd[0].cpu()  # 0..1
    
    # Align to amp frames
    T = min(amp.shape[0], f0.shape[0])
    amp = amp[:T]
    f0 = f0[:T]
    periodicity = periodicity[:T]
    
    # Voiced mask: confidence + envelope gate
    vmask = ((periodicity > 0.5) & (amp > 0.02)).float()
    
    # rbend (semitones rel. A4), masked
    f0_safe = torch.where(f0 > 0, f0, torch.ones_like(f0))
    rbend = 12.0 * torch.log2(f0_safe / 440.0)
    rbend = torch.where(torch.isfinite(rbend), rbend, torch.zeros_like(rbend))
    rbend = rbend * vmask
    
    # Save
    np.save(out_amp, amp.cpu().numpy().astype("float32"))
    np.save(out_rframe, vmask.cpu().numpy().astype("float32"))
    np.save(out_rbend, rbend.cpu().numpy().astype("float32"))


@torch.no_grad()
def sample_from_noise(model: Pipeline,
                      pr: torch.Tensor, amp: torch.Tensor, rframe: torch.Tensor,
                      rbend: torch.Tensor, rb_mask: torch.Tensor,
                      enc_fast: torch.Tensor, gid: int, sgid: int,
                      steps=30, sr_out=48000, seed=0):
    device = next(model.parameters()).device
    B, C_lat, H_lat = 1, 8, 16
    T_slow = int(pr.shape[-1])

    # shape checks / moves
    pr  = pr.to(device).unsqueeze(0)          # [1,128,T]
    amp = amp.to(device).unsqueeze(0)         # [1,T]
    rframe = rframe.to(device).unsqueeze(0)   # [1,T]
    rbend  = rbend.to(device).unsqueeze(0)    # [1,T]
    rb_mask= rb_mask.to(device).unsqueeze(0)  # [1,T]

    # ensure enc_fast is [1,C_fast,T_fast] float
    if enc_fast.dim() == 2:
        enc_fast = enc_fast.unsqueeze(0)
    enc_fast = enc_fast.to(device=device, dtype=torch.float32)

    group_id    = torch.tensor([gid],  device=device, dtype=torch.long)
    subgroup_id = torch.tensor([sgid], device=device, dtype=torch.long)

    # control tokens
    tokens, mask = model.ctrl_enc(
        piano_roll=pr, amp=amp, rframe=rframe,
        rbend=rbend, rbend_mask=rb_mask,
        encodec_tokens=enc_fast,
        group_id=group_id, subgroup_id=subgroup_id,
    )

    # start from pure noise (NO ground-truth latents needed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    x = torch.randn(1, C_lat, H_lat, T_slow, device=device)

    T_train = 1000
    steps = max(1, int(steps))
    dt = 1.0 / float(steps)

    for i in range(steps, 0, -1):
        t_cont = torch.full((1,), i*dt, device=device)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train-1)

        tokens_adapt = tokens.to(dtype=x.dtype, device=device)
        cond_patch = model.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=model._adapter_gain_scale())
        cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

        x_in   = x + cond_patch
        v_pred = model._call_transformer_no_xattn(latents=x_in, t=t_idx)
        x      = x - dt * v_pred

    # decode with DCAE on CPU
    model.dcae.to("cpu")
    audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    x_cpu = x[:1].float().cpu()
    audio_lengths = torch.tensor([audio_len], dtype=torch.long)
    sr_pred, wav_pred = model.dcae.decode(x_cpu, audio_lengths=audio_lengths, sr=sr_out)
    model.dcae.to(device)
    return wav_pred[0].cpu(), int(sr_pred)

def main():
    ap = argparse.ArgumentParser()
    
    # Input options: either audio file OR pre-extracted features
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=str, help="Audio file to extract features from and process")
    input_group.add_argument("--extract_dir", type=str,
                    help="Folder with *.encodec.pt, *.pianoroll.npy, *.amp.npy, *.rframe.npy, *.rbend.npy")
    
    ap.add_argument("--components_dir", required=True,
                    help="ACEStep components dir (music_dcae_f8c8, transformer config, etc.)")
    ap.add_argument("--group", default="guitar")
    ap.add_argument("--subgroup", default="electric_guitar")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--sr_out", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Output wav path")
    args = ap.parse_args()

    # Handle audio input vs pre-extracted features
    if args.audio:
        # Audio file input - extract features on the fly
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create temporary directory for extracts
        temp_dir = Path(tempfile.mkdtemp(prefix="audio_extract_"))
        print(f"Using temporary directory: {temp_dir}")
        
        try:
            # Extract all features
            extract_paths = extract_audio_features(audio_path, temp_dir)
            
            pr_path = extract_paths['pianoroll']
            amp_path = extract_paths['amp']
            rframe_path = extract_paths['rframe']
            rbend_path = extract_paths['rbend']
            enc_path = extract_paths['encodec']
            
            # Set default output path if not provided
            if not args.out:
                args.out = str(audio_path.parent / f"{audio_path.stem}_generated.wav")
        
        except Exception as e:
            # Clean up temp directory on error
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    else:
        # Pre-extracted features directory
        d = Path(args.extract_dir)
        # infer common stem
        pr_path    = next(d.glob("*.pianoroll.npy"))
        amp_path   = next(d.glob("*.amp.npy"))
        rframe_path= next(d.glob("*.rframe.npy"))
        rbend_path = next(d.glob("*.rbend.npy"))
        enc_path   = next(d.glob("*.encodec.pt"))
        
        # Set default output path if not provided
        if not args.out:
            args.out = str(d / "preview.wav")

    # Load the extracted features
    pr_np  = np.load(pr_path)                       # [128, T_slow]
    amp_np = np.load(amp_path).astype(np.float32)   # [T]
    rf_np  = np.load(rframe_path).astype(np.float32)
    rb_np  = np.load(rbend_path).astype(np.float32)
    enc    = torch.load(enc_path, map_location="cpu")  # tensor or dict

    # get encodec tensor [C_fast,T_fast] robustly
    if isinstance(enc, list):
        # Handle list format from encodec.encode() output
        if len(enc) > 0 and hasattr(enc[0], 'codes'):
            enc = enc[0].codes.squeeze(0)  # Remove batch dimension
        else:
            raise RuntimeError("encodec list format not recognized")
    elif isinstance(enc, dict):
        for k in ("latents","codes","tokens","encodec","audio_tokens","data"):
            if k in enc and isinstance(enc[k], torch.Tensor):
                enc = enc[k]
                break
        if not isinstance(enc, torch.Tensor):
            raise RuntimeError("encodec pt did not contain a tensor")
    
    if not isinstance(enc, torch.Tensor):
        raise RuntimeError(f"encodec data type not supported: {type(enc)}")
        
    if enc.dim() == 3 and enc.shape[0] == 1:
        enc = enc.squeeze(0)
    if enc.dim() == 2 and enc.shape[0] != 8 and enc.shape[1] == 8:
        enc = enc.transpose(0, 1)
    # trim/pad channels to 8
    C = enc.shape[0]
    if C > 8:
        enc = enc[:8]
    elif C < 8:
        pad = torch.zeros(8 - C, enc.shape[1], dtype=enc.dtype)
        enc = torch.cat([enc, pad], dim=0)
    if not torch.is_floating_point(enc):
        enc = enc.long()

    # align lengths (make all slow-grid series the same T)
    T = max(pr_np.shape[1], len(amp_np), len(rf_np), len(rb_np))
    if pr_np.shape[1] != T:
        # nearest for PR to keep binary-ish
        import librosa
        pr_np = librosa.resample(pr_np.astype(float), orig_sr=pr_np.shape[1], target_sr=T,
                                 axis=1, res_type="nearest")
        pr_np = (pr_np > 0.5).astype(np.uint8)
    def _interp_1d(x, Tt, thresh=None):
        if len(x) == Tt: return x
        xi = np.linspace(0,1,len(x))
        xo = np.linspace(0,1,Tt)
        y = np.interp(xo, xi, x).astype(np.float32)
        if thresh is not None: y = (y > thresh).astype(np.float32)
        return y
    amp_np = _interp_1d(amp_np, T)
    rf_np  = _interp_1d(rf_np,  T, thresh=0.5)
    rb_np  = _interp_1d(rb_np,  T)

    # rbend mask
    rb_mask = make_rbend_mask(pr_np, rf_np, amp_np)

    # pack torch tensors
    pr   = torch.from_numpy(pr_np.astype(np.float32))    # [128,T]
    amp  = torch.from_numpy(amp_np)                      # [T]
    rfr  = torch.from_numpy(rf_np)                       # [T]
    rbd  = torch.from_numpy(rb_np)                       # [T]
    rbm  = rb_mask                                       # [T]
    encf = enc.float()                                   # [8,T_fast]

    # instrument ids
    g2id = {g:i for i,g in enumerate(APPROVED_GROUPS)}
    subs = sorted({sg for v in APPROVED_SUBGROUPS.values() for sg in v})
    sg2id= {sg:i for i,sg in enumerate(subs)}
    gid  = g2id.get(args.group.lower(), g2id["guitar"])
    sgid = sg2id.get(args.subgroup.lower(), sg2id.get("undefined", 0))

    # load model from specific checkpoint
    dummy_manifest = d / "_dummy_manifest.json"
    dummy_manifest.write_text("[]")
    
    # Use the specified checkpoint path
    checkpoint_path = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-08-31_03-14-24_all_groups_ft_v2_fixed_resume3k/checkpoints/epoch=6-step=3500.ckpt"
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model: Pipeline = Pipeline.load_from_checkpoint(
        checkpoint_path,
        checkpoint_dir=args.components_dir,
        manifest_json=str(dummy_manifest),
        batch_size=1,
        preview_steps=args.steps,
    )
    model.eval().freeze()

    wav, sr = sample_from_noise(
        model, pr, amp, rfr, rbd, rbm, encf,
        gid, sgid, steps=args.steps, sr_out=args.sr_out, seed=args.seed
    )

    # Save the generated audio
    torchaudio.save(args.out, wav, sr)

    # Print results
    print(f"\n✅ Generated audio saved to: {args.out}")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duration: {wav.shape[-1] / sr:.2f} seconds")
    print(f"   Steps: {args.steps}")
    print(f"   Instrument: {args.group}/{args.subgroup}")
    
    # Clean up temporary directory if we created one
    if args.audio:
        import shutil
        print(f"   Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
