#!/usr/bin/env python3
"""
Script to load a model from a Lightning checkpoint using trainer_performer.py
Uses built-in Pipeline preview methods for audio generation
"""

import sys
import os
import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
sys.path.append('/home/arlo/Data')

from trainer_performer import Pipeline, PerformerAIDataset, collate_latent_cond
import subprocess
import tempfile
import json

def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning") -> dict:
    """Extract conditioning from audio file using test_extract_local.py"""
    print(f"Extracting conditioning from: {audio_path}")
    
    # Construct paths 
    audio_path_obj = Path(audio_path)
    stem = audio_path_obj.stem
    safe_stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem)[:128] or "audio"
    out_dir = Path(output_dir) / safe_stem
    
    # Check if extraction files already exist
    required_files = [
        out_dir / f"{safe_stem}.pianoroll.npy",
        out_dir / f"{safe_stem}.amp.npy", 
        out_dir / f"{safe_stem}.rframe.npy",
        out_dir / f"{safe_stem}.rbend.npy",
        out_dir / f"{safe_stem}.encodec.pt"
    ]
    
    if all(f.exists() for f in required_files):
        print("✅ Using existing extracted conditioning files")
        return {
            "output_dir": str(out_dir),
            "stem": safe_stem
        }
    
    # Run the extraction script with timeout
    cmd = ["python", "test_extract_local.py", "--input", audio_path, "--output", output_dir]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            print(f"Extraction stderr: {result.stderr}")
            print(f"Extraction stdout: {result.stdout}")
            raise RuntimeError(f"Conditioning extraction failed: {result.stderr}")
        
        print("✅ Conditioning extracted successfully")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Conditioning extraction timed out after 5 minutes")
    
    return {
        "output_dir": str(out_dir),
        "stem": safe_stem
    }

def load_conditioning_arrays_from_extraction(extraction_info: dict):
    """Load conditioning data from extraction results and align to window_slow"""
    out_dir = Path(extraction_info["output_dir"])
    stem = extraction_info.get("stem") or out_dir.name
    
    print(f"Loading conditioning from: {out_dir}")
    
    # Load conditioning files
    piano_roll_path = out_dir / f"{stem}.pianoroll.npy"
    amp_path = out_dir / f"{stem}.amp.npy"
    rframe_path = out_dir / f"{stem}.rframe.npy"
    rbend_path = out_dir / f"{stem}.rbend.npy"
    encodec_path = out_dir / f"{stem}.encodec.pt"
    
    if not all(p.exists() for p in [piano_roll_path, amp_path, rframe_path, rbend_path, encodec_path]):
        raise FileNotFoundError(f"Missing conditioning files in {out_dir}")
    
    # Load the conditioning data
    piano_roll = np.load(piano_roll_path)  # (128, T)
    amp = np.load(amp_path)  # (T,)
    rframe = np.load(rframe_path)  # (T,)
    rbend = np.load(rbend_path)  # (T,)
    
    # Load encodec tokens
    encodec_data = torch.load(encodec_path, map_location="cpu")
    if isinstance(encodec_data, list) and len(encodec_data) > 0:
        first_item = encodec_data[0] 
        if isinstance(first_item, tuple):
            encodec_tokens = first_item[0] if first_item[0] is not None else first_item[1]
        else:
            encodec_tokens = first_item
    elif isinstance(encodec_data, tuple):
        encodec_tokens = encodec_data[0]
    else:
        encodec_tokens = encodec_data
    
    if not isinstance(encodec_tokens, torch.Tensor):
        encodec_tokens = torch.tensor(encodec_tokens)
    
    if len(encodec_tokens.shape) == 2:  # (n_q, T) - add batch dim
        encodec_tokens = encodec_tokens.unsqueeze(0)
    
    # Align all to window_slow=2584
    window_slow = 2584
    orig_T = min(piano_roll.shape[1], len(amp), len(rframe), len(rbend))
    encodec_T = encodec_tokens.shape[-1]
    
    print(f"Original lengths - conditioning: {orig_T}, encodec: {encodec_T}")
    print(f"Aligning to window_slow: {window_slow}")
    
    def crop_or_pad_1d(arr, target_len):
        if len(arr) > target_len:
            return arr[:target_len]
        elif len(arr) < target_len:
            pad_len = target_len - len(arr)
            return np.pad(arr, (0, pad_len), mode='constant', constant_values=0)
        return arr
    
    def crop_or_pad_2d(arr, target_len):
        if arr.shape[1] > target_len:
            return arr[:, :target_len]
        elif arr.shape[1] < target_len:
            pad_len = target_len - arr.shape[1]
            return np.pad(arr, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
        return arr
    
    # Apply alignment
    piano_roll = crop_or_pad_2d(piano_roll, window_slow)
    amp = crop_or_pad_1d(amp, window_slow)
    rframe = crop_or_pad_1d(rframe, window_slow)
    rbend = crop_or_pad_1d(rbend, window_slow)
    
    if encodec_T > window_slow:
        encodec_tokens = encodec_tokens[:, :, :window_slow]
    elif encodec_T < window_slow:
        pad_len = window_slow - encodec_T
        encodec_tokens = torch.nn.functional.pad(encodec_tokens, (0, pad_len))
    
    print(f"✅ Aligned conditioning: piano_roll {piano_roll.shape}, amp {amp.shape}, encodec {encodec_tokens.shape}")
    return piano_roll, amp, rframe, rbend, encodec_tokens

def load_model_from_checkpoint(audio_file: str = None):
    """Load model from checkpoint and setup for audio generation"""
    
    checkpoint_path = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-08-31_03-14-24_all_groups_ft_v2_fixed_resume3k/checkpoints/last.ckpt"
    checkpoint_dir = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"
    manifest_json = "./final_training_manifest_final.json"
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    try:
        model = Pipeline.load_from_checkpoint(
            checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            manifest_json=manifest_json,
        )
        
        print("✅ Model loaded successfully!")
        model.eval()
        
        # Setup preview batch
        if audio_file:
            # Extract and load conditioning from uploaded audio
            extraction_info = extract_conditioning_from_audio(audio_file)
            piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning_arrays_from_extraction(extraction_info)
            
            # Build batch exactly as Pipeline expects
            batch = {
                "latents": None,  # No GT for uploaded audio
                "encodec_tokens": encodec_tokens.long(),  # Must be long indices
                "conds": {
                    "piano_roll": torch.from_numpy(piano_roll).float().unsqueeze(0),  # (1, 128, T)
                    "amp": torch.from_numpy(amp).float().unsqueeze(0),  # (1, T)  
                    "rframe": torch.from_numpy(rframe).float().unsqueeze(0),  # (1, T)
                    "rbend": torch.from_numpy(rbend).float().unsqueeze(0),  # (1, T)
                    "rbend_mask": torch.from_numpy((rframe > 0.5).astype(np.float32)).unsqueeze(0),  # (1, T)
                },
                "instrument": {
                    "group_id": torch.tensor([0], dtype=torch.long),
                    "subgroup_id": torch.tensor([0], dtype=torch.long),
                },
            }
            
            model._preview_batch = batch
            model._wrote_gt = True  # Skip GT saving
            print(f"✅ Custom conditioning ready: {piano_roll.shape[1]} frames")
            
        else:
            print("No audio file provided - would setup default dataset preview")
            
        # Ensure correct device placement
        if hasattr(model, 'ctrl_enc'):
            model.ctrl_enc.to(model.device)
        if hasattr(model, 'dcae'):
            model.dcae.to("cpu")  # Keep on CPU to save VRAM
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load model and generate audio from uploaded conditioning")
    parser.add_argument("--audio", help="Path to audio file to extract conditioning from")
    args = parser.parse_args()
    
    model = load_model_from_checkpoint(audio_file=args.audio)
    
    if model is not None:
        print("\n=== Model Info ===")
        print(f"Model class: {model.__class__.__name__}")
        
        if hasattr(model, 'hparams'):
            print(f"Window slow: {model.hparams.window_slow}")
            print(f"Batch size: {model.hparams.batch_size}")
        
        # Check if preview functionality is available
        if hasattr(model, '_preview_batch') and model._preview_batch is not None:
            print("\n=== Audio Generation Ready ===")
            if args.audio:
                print("Built-in preview methods available:")
                print("- model._preview_euler_20(model._preview_batch, sr_out=32000)")
                print("- model._preview_euler_40(model._preview_batch, sr_out=32000)") 
                
                # Ask user to run a test
                try:
                    response = input("\\nGenerate test audio with Euler-20? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        print("\\n🎵 Generating audio...")
                        os.makedirs("./preview_results", exist_ok=True)
                        model._preview_euler_20(model._preview_batch, sr_out=32000)
                        print("✅ Check ./preview_results/euler_20.wav for generated audio!")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\\nSkipped generation.")
            else:
                print("Provide --audio flag to generate from your own audio")
        else:
            print("\\nPreview functionality not available.")
        
        print("\\n=== Model Ready ===")
        
    else:
        print("Failed to load model")

if __name__ == "__main__":
    main()