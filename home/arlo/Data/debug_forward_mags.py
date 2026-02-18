#!/usr/bin/env python3
"""
Sanity-check the forward magnitudes (no decode)

This runs the exact forward used for generation and prints norms. 
It will tell us if ctrl_enc or cond_adapter are dead/zero.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append('/home/arlo/Data')
from trainer_performer import Pipeline

def main():
    ckpt = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-08-31_03-14-24_all_groups_ft_v2_fixed_resume3k/checkpoints/last.ckpt"
    checkpoint_dir = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"

    # Check if checkpoint has cond_adapter keys
    blob = torch.load(ckpt, map_location="cpu")
    sd = blob.get("state_dict", blob)
    print("HAS cond_adapter keys?:", any(k.startswith("cond_adapter") for k in sd))
    print("HAS ctrl_enc keys?:", any(k.startswith("ctrl_enc") for k in sd))
    
    # Check for alternative key patterns
    alt_patterns = ["adapter.", "model.cond_adapter.", "module.cond_adapter.", "net.cond_adapter."]
    for pattern in alt_patterns:
        if any(k.startswith(pattern) for k in sd):
            print(f"HAS {pattern}* keys: True")

    # Spin up model
    model = Pipeline(
        checkpoint_dir=checkpoint_dir, 
        manifest_json="./final_training_manifest_final.json"
    ).eval()
    
    # Apply key remapping if needed (same logic as main script)
    key_remaps = []
    model_keys = set(dict(model.named_parameters()).keys())
    ckpt_keys = set(sd.keys())
    
    prefixes_to_check = ["cond_adapter.", "ctrl_enc.", "transformers."]
    for prefix in prefixes_to_check:
        model_has = any(k.startswith(prefix) for k in model_keys)
        ckpt_has = any(k.startswith(prefix) for k in ckpt_keys)
        
        if model_has and not ckpt_has:
            # Try common prefix variations
            for candidate in [f"model.{prefix}", f"module.{prefix}", f"net.{prefix}", f"adapter." if prefix == "cond_adapter." else None]:
                if candidate and any(k.startswith(candidate) for k in ckpt_keys):
                    print(f"[remap] {candidate}* → {prefix}*")
                    for old_k in list(sd.keys()):
                        if old_k.startswith(candidate):
                            new_k = old_k.replace(candidate, prefix, 1)
                            sd[new_k] = sd.pop(old_k)
                            key_remaps.append((old_k, new_k))
                    break
    
    if key_remaps:
        print(f"[remap] Applied {len(key_remaps)} key remaps")
    
    # Load state dict
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    
    # Show critical missing keys
    critical_missing = [k for k in missing if any(k.startswith(p) for p in ["cond_adapter.", "ctrl_enc."])]
    if critical_missing:
        print("CRITICAL MISSING:", critical_missing[:10])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Load conditioning that your run used
    stem = "tptest"
    base = Path("extracted_conditioning") / stem
    
    pr = np.load(base / f"{stem}.pianoroll.npy")     # [128,T]
    amp = np.load(base / f"{stem}.amp.npy")          # [T]
    rfr = np.load(base / f"{stem}.rframe.npy")       # [T]
    rbd = np.load(base / f"{stem}.rbend.npy")        # [T]
    enc = torch.load(base / f"{stem}.encodec.pt", map_location="cpu")
    
    # Normalize encodec format (same logic as main script)
    if isinstance(enc, list) and len(enc) > 0:
        x = enc[0]
        if hasattr(x, "codes"):
            enc = x.codes
        elif isinstance(x, tuple):
            enc = x[0] if x[0] is not None else x[1]
        else:
            enc = x
    elif isinstance(enc, tuple):
        enc = enc[0]
    elif hasattr(enc, "codes"):
        enc = enc.codes
    
    if not isinstance(enc, torch.Tensor):
        enc = torch.tensor(enc)
    if enc.ndim == 2: 
        enc = enc.unsqueeze(0)
    
    pr_t = torch.from_numpy(pr).float().unsqueeze(0).to(device)   # [1,128,T]
    amp_t = torch.from_numpy(amp).float().unsqueeze(0).to(device)
    rfr_t = torch.from_numpy(rfr).float().unsqueeze(0).to(device)
    rbd_t = torch.from_numpy(rbd).float().unsqueeze(0).to(device)
    rbm_t = (rfr_t > 0.5)

    # ids (from your logs): group=4, subgroup=14
    g_id = torch.tensor([4], dtype=torch.long, device=device)
    s_id = torch.tensor([14], dtype=torch.long, device=device)

    print(f"\nInput shapes: pr={pr_t.shape}, amp={amp_t.shape}, rfr={rfr_t.shape}, rbd={rbd_t.shape}, enc={enc.shape}")
    print(f"EnCodec nonzero fraction: {(enc != 0).float().mean().item():.3f}")

    with torch.no_grad():
        # Test ctrl_enc
        try:
            toks, mask = model.ctrl_enc(
                pr_t, amp_t, rfr_t, rbd_t, rbm_t, 
                torch.as_tensor(enc).long().to(device), 
                g_id, s_id
            )
            print(f"\nctrl_enc SUCCESS:")
            print(f"  tokens shape: {toks.shape}")
            print(f"  tokens mean/std: {toks.float().mean().item():.6f} / {toks.float().std().item():.6f}")
            print(f"  tokens absmax: {toks.float().abs().max().item():.6f}")
        except Exception as e:
            print(f"\nctrl_enc FAILED: {e}")
            return

        # Test cond_adapter
        try:
            T_out = pr.shape[1]  # Use original length
            ca_dtype = toks.dtype
            if hasattr(model.cond_adapter, 'weight') and hasattr(model.cond_adapter.weight, 'dtype'):
                ca_dtype = model.cond_adapter.weight.dtype
            
            ca = model.cond_adapter(
                toks.to(ca_dtype), 
                T_out=T_out, 
                scale=1.0
            )
            ca = ca.to(device=toks.device, dtype=toks.dtype)
            
            print(f"\ncond_adapter SUCCESS:")
            print(f"  patch shape: {ca.shape}")
            print(f"  patch mean/std: {ca.float().mean().item():.6f} / {ca.float().std().item():.6f}")
            print(f"  patch absmax: {ca.float().abs().max().item():.6f}")
            
            # Diagnosis
            if ca.float().std().item() < 1e-6:
                print("  ⚠️  PATCH STD TOO SMALL - adapter weights likely not loaded!")
            elif ca.float().abs().max().item() < 1e-4:
                print("  ⚠️  PATCH VALUES TOO SMALL - may cause noise in decode")
            else:
                print("  ✅ PATCH VALUES LOOK REASONABLE")
                
        except Exception as e:
            print(f"\ncond_adapter FAILED: {e}")

if __name__ == "__main__":
    main()