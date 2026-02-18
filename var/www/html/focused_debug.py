#!/usr/bin/env python3
"""
Focused Checkpoint Debugger
Allows selective testing of specific parameters to diagnose issues efficiently.
"""

import os
import glob
import torch
import argparse
import torchaudio
from datetime import datetime

# Import your training pipeline components
from trainer_performer import Pipeline, PerformerAIDataset, collate_latent_cond

def find_all_checkpoints(logs_dir, min_steps=1000):
    """Find all checkpoints that have reached at least min_steps"""
    checkpoints = []
    
    # Find all lightning_logs directories
    lightning_dirs = glob.glob(os.path.join(logs_dir, "lightning_logs", "*"))
    
    for lightning_dir in lightning_dirs:
        # Look for checkpoints in this directory
        checkpoint_dir = os.path.join(lightning_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            continue
            
        # Get all checkpoints
        all_ckpts = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        
        for ckpt_path in all_ckpts:
            # Extract step number from filename
            ckpt_name = os.path.basename(ckpt_path)
            if "step=" in ckpt_name:
                step_num = int(ckpt_name.split("step=")[1].split(".")[0])
                if step_num >= min_steps:
                    checkpoints.append((ckpt_path, step_num))
            elif "last.ckpt" in ckpt_name:
                # For last.ckpt, check if there are eval results with sufficient steps
                eval_dir = os.path.join(lightning_dir, "eval_results")
                if os.path.exists(eval_dir):
                    eval_steps = [int(f.split("_")[1]) for f in os.listdir(eval_dir) 
                                 if f.startswith("step_") and f.split("_")[1].isdigit()]
                    if eval_steps and max(eval_steps) >= min_steps:
                        checkpoints.append((ckpt_path, max(eval_steps)))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints

def modify_conditioning(batch, modification_type):
    """Modify conditioning signals for testing"""
    modified_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    if modification_type == "no_piano_roll":
        modified_batch["conds"]["piano_roll"] = torch.zeros_like(modified_batch["conds"]["piano_roll"])
    elif modification_type == "no_amp":
        modified_batch["conds"]["amp"] = torch.zeros_like(modified_batch["conds"]["amp"])
    elif modification_type == "no_rframe":
        modified_batch["conds"]["rframe"] = torch.zeros_like(modified_batch["conds"]["rframe"])
    elif modification_type == "no_rbend":
        modified_batch["conds"]["rbend"] = torch.zeros_like(modified_batch["conds"]["rbend"])
        modified_batch["conds"]["rbend_mask"] = torch.zeros_like(modified_batch["conds"]["rbend_mask"])
    elif modification_type == "no_encodec":
        modified_batch["encodec_tokens"] = torch.zeros_like(modified_batch["encodec_tokens"])
    elif modification_type == "random_instrument":
        # Randomize instrument group/subgroup
        B = modified_batch["instrument"]["group_id"].shape[0]
        modified_batch["instrument"]["group_id"] = torch.randint(0, 6, (B,), device=modified_batch["instrument"]["group_id"].device)
        modified_batch["instrument"]["subgroup_id"] = torch.randint(0, 16, (B,), device=modified_batch["instrument"]["subgroup_id"].device)
    
    return modified_batch

def generate_preview(model, batch, steps=30, sr_out=48000, 
                    conditioning_strength=1.0, start_noise_level=1.0, 
                    sampling_method="euler", seed=0):
    """Generate a preview with varied parameters"""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        x0 = batch["latents"]
        if x0 is None:
            return None
            
        x0 = x0.to(device)
        B, _, _, T_slow = x0.shape
        
        # Build control tokens
        tokens, mask = model.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"].to(device),
            amp=batch["conds"]["amp"].to(device),
            rframe=batch["conds"]["rframe"].to(device),
            rbend=batch["conds"]["rbend"].to(device),
            rbend_mask=batch["conds"]["rbend_mask"].to(device),
            encodec_tokens=batch["encodec_tokens"].to(device),
            group_id=batch["instrument"]["group_id"].to(device),
            subgroup_id=batch["instrument"]["subgroup_id"].to(device),
        )
        
        # Start from noise with specified level
        if start_noise_level < 1.0:
            # Start from partially noisy version of ground truth
            x = (1.0 - start_noise_level) * x0 + start_noise_level * torch.randn_like(x0)
            # Adjust steps based on noise level
            steps = int(steps * start_noise_level)
        else:
            # Start from pure noise
            x = torch.randn_like(x0)
        
        T_train = 1000
        dt = 1.0 / float(steps)
        
        for i in range(steps, 0, -1):
            t_cont = torch.full((B,), i * dt, device=device)
            t_idx = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)
            
            # Get conditioning patch with specified strength
            cond_patch = model.cond_adapter(tokens, T_out=x.shape[-1], 
                                          scale=conditioning_strength * model._adapter_gain_scale())
            cond_patch = cond_patch.to(device=device, dtype=x.dtype)
            
            x_in = x + cond_patch
            
            # Different sampling methods
            if sampling_method == "euler":
                v_pred = model._call_transformer_no_xattn(latents=x_in, t=t_idx)
                x = x - dt * v_pred
            elif sampling_method == "euler_plus":
                # Euler with smaller steps for stability
                v_pred = model._call_transformer_no_xattn(latents=x_in, t=t_idx)
                x = x - (dt * 0.5) * v_pred
                # Second evaluation
                v_pred2 = model._call_transformer_no_xattn(latents=x, t=t_idx-1)
                x = x - (dt * 0.5) * v_pred2
        
        # Move DCAE to CPU for decoding
        model.dcae.to("cpu")
        
        # Decode on CPU
        audio_len_out = int(round(T_slow * 4096 * (sr_out / 44100)))
        x_for_dcae = x[:1].to("cpu").float()
        audio_lengths = torch.tensor([audio_len_out], device="cpu", dtype=torch.long)
        sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)
        
        # Move DCAE back to original device
        model.dcae.to(device)
        
        return wav_pred[0].float().cpu(), sr_pred

def run_focused_experiments(checkpoint_path, step_count, manifest_json, output_dir, 
                           preview_index=0, target_param=None, param_values=None):
    """Run focused experiments for a specific parameter"""
    checkpoint_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
    exp_output_dir = os.path.join(output_dir, f"{checkpoint_name}_step{step_count}")
    os.makedirs(exp_output_dir, exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path} (step {step_count})")
    model = Pipeline.load_from_checkpoint(
        checkpoint_path,
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        manifest_json=manifest_json
    )
    
    model.eval()
    model.freeze()
    
    # Set up preview dataset
    ds_prev = PerformerAIDataset(
        json_path=manifest_json,
        conditioning_dropout={"piano_roll":0.0, "amp":0.0, "rbend":0.0, "rframe":0.0},
        use_trim=True,
        pre_roll_seconds=1.0,
        post_roll_seconds=0.0,
        keep_untrimmed_prob=0.0,
        amp_activity_thr=0.06,
        require_all_core=True,
        collapse_sparse_subgroups_to_any=False,
        static_window=True,
        window_slow=2584,  # Match your training window size (4 minutes)
        seed=0
    )
    
    idx = preview_index % len(ds_prev)
    item = ds_prev[idx]
    original_batch = collate_latent_cond([item])
    
    # Move batch to the same device as model
    device = next(model.parameters()).device
    for k, v in original_batch.items():
        if isinstance(v, torch.Tensor):
            original_batch[k] = v.to(device)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    v[k2] = v2.to(device)
    
    # Save ground truth first
    print("Saving ground truth...")
    model.dcae.to("cpu")
    x0 = original_batch["latents"][:1].to("cpu")
    T_slow = x0.shape[-1]
    
    # Pad ground truth to match training window size if needed
    target_window = 2584  # Match training window_slow
    if T_slow < target_window:
        print(f"Padding ground truth from {T_slow} to {target_window} frames")
        pad_amount = target_window - T_slow
        x0 = torch.nn.functional.pad(x0, (0, pad_amount))
        T_slow = target_window
    elif T_slow > target_window:
        print(f"Truncating ground truth from {T_slow} to {target_window} frames")  
        x0 = x0[..., :target_window]
        T_slow = target_window
        
    audio_len = int(round(T_slow * 4096 * (48000 / 44100)))
    audio_lengths = torch.full((1,), audio_len, device="cpu", dtype=torch.long)
    sr_gt, wav_gt = model.dcae.decode(x0, audio_lengths=audio_lengths, sr=48000)
    torchaudio.save(f"{exp_output_dir}/gt.wav", wav_gt[0].float().cpu(), sr_gt)
    model.dcae.to(device)
    
    # Default parameter values (match training specs)
    defaults = {
        "steps": 30,
        "conditioning_strength": 1.5,  # Match training scale: 1.5 * _adapter_gain_scale()
        "start_noise_level": 1.0,
        "sampling_method": "euler",
        "conditioning_type": "original",
        "seed": 0
    }
    
    # Parameter values to test
    if target_param and param_values:
        test_values = param_values
    else:
        # If no specific parameter, just test a few basics
        test_values = [defaults["steps"]]
        target_param = "steps"
    
    # Run experiments
    for i, value in enumerate(test_values):
        try:
            # Set up parameters based on target
            params = defaults.copy()
            
            if target_param == "steps":
                params["steps"] = value
                filename = f"steps_{value}.wav"
            elif target_param == "conditioning_strength":
                params["conditioning_strength"] = value
                filename = f"strength_{value}.wav"
            elif target_param == "start_noise_level":
                params["start_noise_level"] = value
                filename = f"noise_{value}.wav"
            elif target_param == "sampling_method":
                params["sampling_method"] = value
                filename = f"sampling_{value}.wav"
            elif target_param == "conditioning_type":
                params["conditioning_type"] = value
                filename = f"cond_{value}.wav"
            elif target_param == "seed":
                params["seed"] = value
                filename = f"seed_{value}.wav"
            else:
                filename = f"test_{i}.wav"
            
            # Modify conditioning if needed
            if target_param == "conditioning_type":
                batch = modify_conditioning(original_batch, value)
            else:
                batch = original_batch
            
            print(f"Testing {target_param} = {value}...")
            
            # Generate preview
            wav, sr = generate_preview(
                model, batch, 
                steps=params["steps"],
                conditioning_strength=params["conditioning_strength"],
                start_noise_level=params["start_noise_level"],
                sampling_method=params["sampling_method"],
                seed=params["seed"]
            )
            
            if wav is not None:
                torchaudio.save(f"{exp_output_dir}/{filename}", wav, sr)
                print(f"Saved: {filename}")
                
        except Exception as e:
            print(f"Error testing {target_param} = {value}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Completed experiments for {checkpoint_path}. Results in {exp_output_dir}")
    return exp_output_dir

def main():
    parser = argparse.ArgumentParser(description="Run focused preview experiments on checkpoints")
    parser.add_argument("--logs_dir", type=str, default="/mnt/msdd/exps/logs_v3",
                       help="Base directory containing lightning_logs")
    parser.add_argument("--manifest", type=str, default="./final_training_manifest_final.json",
                       help="Path to manifest JSON file")
    parser.add_argument("--preview_index", type=int, default=0,
                       help="Index of sample to use for previews")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: auto-generated)")
    parser.add_argument("--min_steps", type=int, default=1000,
                       help="Minimum number of steps for checkpoints to include")
    parser.add_argument("--target_param", type=str, default="steps",
                       help="Parameter to test: steps, conditioning_strength, start_noise_level, sampling_method, conditioning_type, seed")
    parser.add_argument("--param_values", type=str, default="",
                       help="Comma-separated values to test for the target parameter")
    
    args = parser.parse_args()
    
    # Parse parameter values
    if args.param_values:
        if args.target_param in ["steps", "seed"]:
            param_values = [int(x) for x in args.param_values.split(",")]
        elif args.target_param in ["conditioning_strength", "start_noise_level"]:
            param_values = [float(x) for x in args.param_values.split(",")]
        else:
            param_values = args.param_values.split(",")
    else:
        # Default values for each parameter type
        if args.target_param == "steps":
            param_values = [10, 20, 30, 40, 50]
        elif args.target_param == "conditioning_strength":
            param_values = [0.5, 1.0, 1.5, 2.0, 3.0]  # Include 1.5 (training default)
        elif args.target_param == "start_noise_level":
            param_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        elif args.target_param == "sampling_method":
            param_values = ["euler", "euler_plus"]
        elif args.target_param == "conditioning_type":
            param_values = ["original", "no_piano_roll", "no_amp", "no_rframe", "no_rbend", "no_encodec", "random_instrument"]
        elif args.target_param == "seed":
            param_values = [0, 42, 123]
        else:
            param_values = [30]  # Default to steps=30
    
    # Find all qualifying checkpoints
    checkpoints = find_all_checkpoints(args.logs_dir, args.min_steps)
    
    if not checkpoints:
        print("No qualifying checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} qualifying checkpoints:")
    for ckpt_path, steps in checkpoints:
        print(f"  - {ckpt_path} (step {steps})")
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"./focused_debug_{args.target_param}_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments for each checkpoint
    results = []
    for ckpt_path, steps in checkpoints:
        try:
            result_dir = run_focused_experiments(
                ckpt_path, steps, args.manifest, args.output_dir, 
                args.preview_index, args.target_param, param_values
            )
            results.append((ckpt_path, steps, result_dir))
        except Exception as e:
            print(f"Failed to process {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary
    summary_file = os.path.join(args.output_dir, "experiment_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Focused Preview Experiment Summary\n")
        f.write("==================================\n\n")
        
        f.write(f"Target Parameter: {args.target_param}\n")
        f.write(f"Tested Values: {param_values}\n\n")
        
        f.write("Checkpoints Processed:\n")
        for ckpt_path, steps, result_dir in results:
            f.write(f"- {ckpt_path} (step {steps}): {result_dir}\n")
    
    print(f"All experiments completed. Summary in {summary_file}")

if __name__ == "__main__":
    main()