import os

# Video dimensions (9:16 vertical for Shorts)
WIDTH = 1080
HEIGHT = 1920

# Physics simulation rate (higher = smoother physics)
SIM_FPS = 60

# Video output frame rate
VIDEO_FPS = 30

# How many sim steps per rendered frame
SIM_STEPS_PER_FRAME = SIM_FPS // VIDEO_FPS  # 2

# Output directory
OUTPUT_DIR = "/home/arlo/gcs-bucket/youtube-shorts-queue"

# Audio settings
AUDIO_SAMPLE_RATE = 44100
TEMP_DIR = "/tmp/youtube-shorts-temp"
DEFAULT_SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

# ACE-Step generation model
ACESTEP_CKPT = "/mnt/models/epoch=102-step=60000.ckpt"
ACESTEP_CKPT_DIR = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"
ACESTEP_MANIFEST = "/home/arlo/gcs-bucket/Manifests/unified_manifest.json"

# Inverse patch z-space operations
INVERSE_PATCH_DIR = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs"
PITCHBIN_AXES = os.path.join(INVERSE_PATCH_DIR, "pitchbin_discovery", "pitchbin_axes.pt")
CONTRASTIVE_AXES = os.path.join(INVERSE_PATCH_DIR, "contrastive_discovery", "discovered_axes.pt")
OPERATION_TREE = os.path.join(INVERSE_PATCH_DIR, "phase2_operation_tree", "operation_tree.pt")
AUDIO_EDITOR = os.path.join(INVERSE_PATCH_DIR, "audio_domain_editor", "audio_domain_editor.pt")

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
