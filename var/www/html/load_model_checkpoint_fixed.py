#!/usr/bin/env python3
"""
Script to load a model from a Lightning checkpoint using trainer_performer.py
Includes preview functionality matching trainer_performer.py
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
    """Load conditioning data from extraction results into model format"""
    out_dir = Path(extraction_info["output_dir"])
    stem = extraction_info.get("stem") or out_dir.name
    
    print(f"Loading conditioning from: {out_dir}")
    print(f"Stem: {stem}")
    
    # Load conditioning files
    piano_roll_path = out_dir / f"{stem}.pianoroll.npy"
    amp_path = out_dir / f"{stem}.amp.npy"
    rframe_path = out_dir / f"{stem}.rframe.npy"
    rbend_path = out_dir / f"{stem}.rbend.npy"
    encodec_path = out_dir / f"{stem}.encodec.pt"
    
    print(f"Expected files:")
    for p in [piano_roll_path, amp_path, rframe_path, rbend_path, encodec_path]:
        print(f"  {p}: exists={p.exists()}")
    
    if not all(p.exists() for p in [piano_roll_path, amp_path, rframe_path, rbend_path, encodec_path]):
        raise FileNotFoundError(f"Missing conditioning files in {out_dir}")
    
    try:
        # Load the conditioning data
        print("Loading conditioning arrays...")
        piano_roll = np.load(piano_roll_path)  # (128, T)
        amp = np.load(amp_path)  # (T,)
        rframe = np.load(rframe_path)  # (T,)
        rbend = np.load(rbend_path)  # (T,)
        
        print("Loading encodec data...")
        encodec_data = torch.load(encodec_path, map_location="cpu")  # EncodecModel tokens
        print(f"Encodec data type: {type(encodec_data)}")
        
        # Handle different encodec formats
        if isinstance(encodec_data, list) and len(encodec_data) > 0:
            print(f"Encodec data is list with {len(encodec_data)} items")
            print(f"First item type: {type(encodec_data[0])}")
            
            # Handle list of EncodedFrames
            first_item = encodec_data[0] 
            if hasattr(first_item, 'codes'):
                encodec_tokens = first_item.codes
                print(f"Got codes from first item: {type(encodec_tokens)}")
            elif isinstance(first_item, tuple):
                print(f"First item is tuple with {len(first_item)} elements")
                for i, elem in enumerate(first_item):
                    print(f"  Element {i}: type={type(elem)}, value={'None' if elem is None else 'has_value'}")
                # Usually (codes, scale) - take codes which should be element 0
                encodec_tokens = first_item[0] if first_item[0] is not None else first_item[1]
                print(f"Selected encodec tokens: {type(encodec_tokens)}")
            else:
                encodec_tokens = first_item
                
        elif isinstance(encodec_data, tuple):
            # Encodec returns (codes, scale) tuple - take the codes
            encodec_tokens = encodec_data[0]  # Shape: (B, n_q, T)
            print(f"Encodec tokens from tuple: {type(encodec_tokens)}, shape: {encodec_tokens.shape if hasattr(encodec_tokens, 'shape') else 'no shape'}")
        elif hasattr(encodec_data, 'codes'):
            encodec_tokens = encodec_data.codes  # Shape: (B, n_q, T)
        else:
            encodec_tokens = encodec_data
        
        # Ensure it's a tensor
        if not isinstance(encodec_tokens, torch.Tensor):
            print(f"Converting {type(encodec_tokens)} to tensor")
            if encodec_tokens is None:
                raise ValueError("Encodec tokens are None - extraction may have failed")
            encodec_tokens = torch.tensor(encodec_tokens)
        
        print(f"Final encodec tokens: {type(encodec_tokens)}, shape: {encodec_tokens.shape}")
        
        # Ensure encodec tokens have correct batch dimension
        if len(encodec_tokens.shape) == 3:  # (B, n_q, T) - correct
            pass
        elif len(encodec_tokens.shape) == 2:  # (n_q, T) - add batch dim
            encodec_tokens = encodec_tokens.unsqueeze(0)
            print(f"Added batch dimension to encodec tokens: {encodec_tokens.shape}")
        else:
            print(f"Warning: Unexpected encodec tokens shape: {encodec_tokens.shape}")
        
    except Exception as e:
        print(f"Error loading conditioning data: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Get the expected window size from model hyperparameters (this should be 2584)
    window_slow = 2584  # Default from the loaded model
    print(f"Using model's window_slow: {window_slow}")
    
    # Get current lengths
    orig_T = min(piano_roll.shape[1], len(amp), len(rframe), len(rbend))
    encodec_T = encodec_tokens.shape[-1] if len(encodec_tokens.shape) >= 2 else 0
    print(f"Original lengths - conditioning: {orig_T}, encodec: {encodec_T}")
    
    # Crop or pad all conditioning to match window_slow
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
    
    # Apply cropping/padding
    piano_roll = crop_or_pad_2d(piano_roll, window_slow)
    amp = crop_or_pad_1d(amp, window_slow)
    rframe = crop_or_pad_1d(rframe, window_slow)
    rbend = crop_or_pad_1d(rbend, window_slow)
    
    # Handle encodec tokens (3D: [B, C, T])
    if encodec_T > window_slow:
        encodec_tokens = encodec_tokens[:, :, :window_slow]
        print(f"Cropped encodec from {encodec_T} to {window_slow}")
    elif encodec_T < window_slow:
        pad_len = window_slow - encodec_T
        encodec_tokens = torch.nn.functional.pad(encodec_tokens, (0, pad_len))
        print(f"Padded encodec from {encodec_T} to {window_slow}")
    
    print(f"Final lengths - piano_roll: {piano_roll.shape[1]}, amp: {len(amp)}, encodec: {encodec_tokens.shape[-1]}")
    
    print(f"✅ Loaded conditioning arrays: piano_roll {piano_roll.shape}, amp {amp.shape}, encodec {encodec_tokens.shape}")
    return piano_roll, amp, rframe, rbend, encodec_tokens

def load_model_from_checkpoint(audio_file: str = None):
    """Load model from the specified checkpoint path and set up preview functionality"""
    
    # Checkpoint path
    checkpoint_path = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-08-31_03-14-24_all_groups_ft_v2_fixed_resume3k/checkpoints/last.ckpt"
    
    # Required parameters for Pipeline.load_from_checkpoint()
    checkpoint_dir = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"
    manifest_json = "./final_training_manifest_final.json"
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Load the model from checkpoint
        model = Pipeline.load_from_checkpoint(
            checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            manifest_json=manifest_json,
        )
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Setup preview functionality
        if audio_file:
            # Extract conditioning from uploaded audio
            extraction_info = extract_conditioning_from_audio(audio_file)
            piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning_arrays_from_extraction(extraction_info)
            
            # Build the preview batch exactly how Pipeline expects
            batch = {
                "latents": None,
                "encodec_tokens": encodec_tokens.long(),  # [1, C_fast, T] - must be long indices
                "conds": {
                    "piano_roll": torch.from_numpy(piano_roll).float().unsqueeze(0),
                    "amp": torch.from_numpy(amp).float().unsqueeze(0),
                    "rframe": torch.from_numpy(rframe).float().unsqueeze(0),
                    "rbend": torch.from_numpy(rbend).float().unsqueeze(0),
                    "rbend_mask": torch.from_numpy((rframe > 0.5).astype(np.float32)).unsqueeze(0),
                },
                "instrument": {
                    "group_id": torch.tensor([0], dtype=torch.long),
                    "subgroup_id": torch.tensor([0], dtype=torch.long),
                },
            }
            
            model._preview_batch = batch
            model._wrote_gt = True  # Skip GT saving since we don't have ground truth
            
            print(f"Custom conditioning initialized with {piano_roll.shape[1]} frames")
        else:
            # Use default manifest-based preview would go here
            print("No audio file provided - would use default dataset")
        
        # Ensure ctrl_enc is on device
        if hasattr(model, 'ctrl_enc'):
            model.ctrl_enc.to(model.device)
            
        # Move dcae to CPU to save VRAM
        if hasattr(model, 'dcae'):
            model.dcae.to("cpu")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTip: Make sure to update checkpoint_dir and manifest_json paths")
        print("These should match the paths used during training")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load model checkpoint and optionally extract conditioning from audio")
    parser.add_argument("--audio", help="Path to audio file to extract conditioning from")
    args = parser.parse_args()
    
    model = load_model_from_checkpoint(audio_file=args.audio)
    
    if model is not None:
        print("\n=== Model Info ===")
        print(f"Model class: {model.__class__.__name__}")
        
        # Print some model hyperparameters if available
        if hasattr(model, 'hparams'):
            print("\n=== Hyperparameters ===")
            for key, value in model.hparams.items():
                print(f"{key}: {value}")
        
        # Check if preview functionality is available
        if hasattr(model, '_preview_batch') and model._preview_batch is not None:
            print("\n=== Preview Functionality Available ===")
            print("Available built-in preview methods:")
            if args.audio:
                print("- Use built-in Pipeline methods with your custom conditioning")
                print("\nTo generate audio, call:")
                print("model._preview_x0_direct_rf(model._preview_batch, t_scalar=0.05, sr_out=32000)")  
                print("model._preview_euler_20(model._preview_batch, sr_out=32000)")
                print("model._preview_euler_40(model._preview_batch, sr_out=32000)")
            else:
                print("- Standard dataset-based previews would be available")
            
            # Ask user if they want to run a test preview
            if args.audio:
                try:
                    response = input("\nRun a test preview (euler_20)? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        print("Running test preview...")
                        model._preview_euler_20(model._preview_batch, sr_out=32000)
                        print("Check ./preview_results/ for generated audio")
                except (EOFError, KeyboardInterrupt):
                    print("\nSkipping preview generation.")
        else:
            print("\nPreview functionality not available.")
        
        print("\n=== Ready for use ===")
        print("Model is loaded and ready for inference.")
        
    else:
        print("Failed to load model")

if __name__ == "__main__":
    main()