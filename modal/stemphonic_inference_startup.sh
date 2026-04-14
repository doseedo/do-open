#!/bin/bash
set -e

# Stemphonic production inference startup script
# Pulls model + scripts from gs://doseedo-production and starts the inference server

WORK_DIR=/opt/stemphonic
CKPT_DIR=$WORK_DIR/checkpoints
SCRIPTS_DIR=$WORK_DIR/scripts
ACESTEP_DIR=$WORK_DIR/ACE-Step-1.5

mkdir -p $CKPT_DIR $SCRIPTS_DIR

# Install NVIDIA drivers if not present
if ! nvidia-smi &>/dev/null; then
  /opt/deeplearning/install-driver.sh
fi

# Pull scripts and checkpoint from production bucket
gsutil -m cp gs://doseedo-production/stemphonic/scripts/* $SCRIPTS_DIR/
gsutil cp gs://doseedo-production/stemphonic/checkpoints/stage2d_step130000.pt $CKPT_DIR/

# Clone ACE-Step if not already present
if [ ! -d "$ACESTEP_DIR" ]; then
  git clone https://github.com/ace-step/ACE-Step-1.5.git $ACESTEP_DIR
  pip install -r $ACESTEP_DIR/requirements.txt
fi

# Install additional dependencies
pip install peft soundfile flask gunicorn

# Download ACE-Step base model weights (needed by handler)
cd $ACESTEP_DIR
python3 -c "from acestep.pipeline_ace_step import AceStepPipeline; AceStepPipeline.from_pretrained('ACE-Step/ACE-Step-v1-5-turbo')" || true

# Write a simple Flask inference server
cat > $WORK_DIR/server.py << 'PYEOF'
import os, sys, json, uuid, tempfile, torch, soundfile as sf
from flask import Flask, request, jsonify, send_file

sys.path.insert(0, "/opt/stemphonic/scripts")
sys.path.insert(0, "/opt/stemphonic/ACE-Step-1.5")

app = Flask(__name__)
handler = None

CKPT_PATH = "/opt/stemphonic/checkpoints/stage2d_step130000.pt"

def load_model():
    global handler
    from acestep.pipeline_ace_step import AceStepPipeline
    from peft import LoraConfig, get_peft_model
    import re

    handler = AceStepPipeline.from_pretrained("ACE-Step/ACE-Step-v1-5-turbo")

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("trainable_state_dict", ckpt)

    # Separate LoRA vs model keys
    lora_keys = {k: v for k, v in state.items() if "lora_" in k}
    model_keys = {k: v for k, v in state.items() if "lora_" not in k and not k.startswith(("midi_", "activity_", "resonance_", "pr_head", "pitch2h"))}

    # Apply LoRA if present
    if lora_keys:
        rank = ckpt.get("config", {}).get("lora_rank", 128)
        alpha = ckpt.get("config", {}).get("lora_alpha", 128)
        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
        )
        handler.transformer = get_peft_model(handler.transformer, lora_config)
        handler.transformer.load_state_dict(lora_keys, strict=False)
        handler.transformer = handler.transformer.merge_and_unload()

    # Load model weights (upper layers / decoder)
    if model_keys:
        handler.transformer.load_state_dict(model_keys, strict=False)

    handler.transformer.to("cuda").eval()
    print("Model loaded successfully", flush=True)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": handler is not None})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "a melodic hip hop beat with warm synths and punchy drums")
    lyrics = data.get("lyrics", "")
    duration = float(data.get("duration", 30.0))
    steps = int(data.get("steps", 50))
    cfg_scale = float(data.get("cfg_scale", 7.0))
    seed = int(data.get("seed", -1))
    if seed < 0:
        seed = torch.randint(0, 2**31, (1,)).item()

    result = handler(
        prompt=prompt,
        lyrics=lyrics if lyrics else None,
        duration=duration,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=torch.Generator("cuda").manual_seed(seed),
    )

    # Save to temp file and return
    audio = result.audios[0].cpu().numpy()
    sr = result.sample_rate
    out_path = os.path.join(tempfile.gettempdir(), f"stemphonic_{uuid.uuid4().hex[:8]}.wav")
    sf.write(out_path, audio.T if audio.ndim > 1 else audio, sr)

    return send_file(out_path, mimetype="audio/wav", as_attachment=True,
                     download_name=f"stemphonic_{seed}.wav")

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080)
PYEOF

# Start the server with gunicorn (single worker since GPU is single-threaded)
cd $WORK_DIR
python3 -c "exec(open('server.py').read().split(\"if __name__\")[0]); load_model(); print('Warmup complete')"
gunicorn -w 1 -b 0.0.0.0:8080 --timeout 300 server:app &

echo "Stemphonic inference server started on port 8080"
