#!/usr/bin/env python3
"""
Standalone FastAPI server for ACE-Step model
Run this script to start the FastAPI server that doseedo2.html/javascript2.js will call
"""

import sys
import os
import argparse
import json

# Add paths FIRST (before other imports)
sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

import torch
import uvicorn

# Import the app and model loading from genfrominterface
from genfrominterface import (
    app, MODEL, GROUP_NAMES, SUBGROUP_NAMES, MANIFEST_DATA,
    load_model_any_ckpt, APPROVED_GROUPS, APPROVED_SUBGROUPS
)

def initialize_model(checkpoint_path: str, checkpoint_dir: str, manifest_path: str):
    """
    DO NOT LOAD MODEL IN FASTAPI - ONLY CELERY SHOULD LOAD IT!
    This function is deprecated. Model loading happens in Celery worker via @worker_process_init
    """
    print("⚠️  WARNING: FastAPI should NOT load the model!")
    print("⚠️  Only Celery worker loads the model to save GPU memory.")
    print("⚠️  Make sure Celery worker is running with environment variables:")
    print("   export ACE_CHECKPOINT='...'")
    print("   export ACE_CHECKPOINT_DIR='...'")
    print("   export ACE_MANIFEST='...'")
    print("   celery -A genfrominterface.celery_app worker --loglevel=info")

if __name__ == "__main__":
    DEFAULT_CKPT = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-09-06_16-12-31_all_groups_ft_v3_capivotpitch_ctrl/checkpoints/last.ckpt"

    ap = argparse.ArgumentParser(description="Run FastAPI server for ACE-Step model")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to model checkpoint (DEPRECATED - use env vars for Celery)")
    ap.add_argument("--checkpoint_dir", required=False, help="Path to checkpoint directory (DEPRECATED)")
    ap.add_argument("--manifest", required=False, help="Path to manifest JSON file (DEPRECATED)")
    ap.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    ap.add_argument("--port", default=8000, type=int, help="Port to bind to")
    args = ap.parse_args()

    # DO NOT initialize model in FastAPI anymore!
    print("\n" + "="*60)
    print("🚀 Starting FastAPI server (NO model loaded here)")
    print("="*60)
    print(f"📍 Server: http://{args.host}:{args.port}")
    print(f"📝 Docs: http://{args.host}:{args.port}/docs")
    print("\n⚠️  IMPORTANT: Make sure Celery worker is running separately!")
    print("   It will load the model automatically.")
    print("="*60 + "\n")

    uvicorn.run(app, host=args.host, port=args.port)
