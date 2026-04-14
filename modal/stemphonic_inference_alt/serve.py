import modal

# --- Image definition ---
image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel")
    .apt_install("git")
    .pip_install(
        "peft", "soundfile", "flask", "transformers", "accelerate",
        "safetensors",
    )
    .run_commands("git clone https://github.com/ace-step/ACE-Step-1.5.git /opt/acestep")
    .run_commands(
        "pip install --no-cache-dir -r /opt/acestep/requirements.txt "
        "--ignore-installed flash-attn || true"
    )
    .run_commands(
        "MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation"
    )
    .add_local_dir(
        "/mnt/data/system_home/arlo/do2/stemphonic_trainer",
        remote_path="/opt/stemphonic/scripts",
    )
)

app = modal.App("stemphonic-inference", image=image)

# Checkpoint stored in a Modal Volume for persistence across cold starts
volume = modal.Volume.from_name("stemphonic-checkpoints", create_if_missing=True)
CKPT_DIR = "/checkpoints"
CKPT_PATH = f"{CKPT_DIR}/stage2d_step130000.pt"


@app.cls(
    gpu="T4",
    timeout=600,
    scaledown_window=300,  # keep warm 5 min after last request
    volumes={CKPT_DIR: volume},
)
@modal.concurrent(max_inputs=1)
class StemphonicInference:
    @modal.enter()
    def load_model(self):
        import sys, os, torch

        sys.path.insert(0, "/opt/stemphonic/scripts")
        sys.path.insert(0, "/opt/acestep")

        # Checkpoint must be pre-uploaded to Modal volume via:
        #   modal volume put stemphonic-checkpoints /scratch/stage2d_step130000.pt stage2d_step130000.pt
        if not os.path.exists(CKPT_PATH):
            raise RuntimeError(
                f"Checkpoint not found at {CKPT_PATH}. "
                "Upload it with: modal volume put stemphonic-checkpoints <local_path> stage2d_step130000.pt"
            )

        from acestep.pipeline_ace_step import AceStepPipeline
        from peft import LoraConfig, get_peft_model

        self.handler = AceStepPipeline.from_pretrained("ACE-Step/ACE-Step-v1-5-turbo")

        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        state = ckpt.get("trainable_state_dict", ckpt)

        lora_keys = {k: v for k, v in state.items() if "lora_" in k}
        model_keys = {
            k: v for k, v in state.items()
            if "lora_" not in k
            and not k.startswith(("midi_", "activity_", "resonance_", "pr_head", "pitch2h"))
        }

        if lora_keys:
            rank = ckpt.get("config", {}).get("lora_rank", 128)
            alpha = ckpt.get("config", {}).get("lora_alpha", 128)
            lora_config = LoraConfig(
                r=rank, lora_alpha=alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0,
            )
            self.handler.transformer = get_peft_model(self.handler.transformer, lora_config)
            self.handler.transformer.load_state_dict(lora_keys, strict=False)
            self.handler.transformer = self.handler.transformer.merge_and_unload()

        if model_keys:
            self.handler.transformer.load_state_dict(model_keys, strict=False)

        self.handler.transformer.to("cuda").eval()
        print("Model loaded.", flush=True)

    @modal.method()
    def generate(
        self,
        prompt: str = "a melodic hip hop beat with warm synths and punchy drums",
        lyrics: str = "",
        duration: float = 30.0,
        steps: int = 50,
        cfg_scale: float = 7.0,
        seed: int = -1,
    ) -> bytes:
        import torch, io, soundfile as sf

        if seed < 0:
            seed = torch.randint(0, 2**31, (1,)).item()

        result = self.handler(
            prompt=prompt,
            lyrics=lyrics if lyrics else None,
            duration=min(duration, 120.0),
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
        )

        audio = result.audios[0].cpu().numpy()
        sr = result.sample_rate
        buf = io.BytesIO()
        sf.write(buf, audio.T if audio.ndim > 1 else audio, sr, format="WAV")
        return buf.getvalue()

    @modal.fastapi_endpoint(method="POST")
    def api_generate(self, request: dict):
        """HTTP endpoint: POST JSON with prompt, lyrics, duration, steps, cfg_scale, seed."""
        import base64
        wav_bytes = self.generate.local(
            prompt=request.get("prompt", "a melodic hip hop beat"),
            lyrics=request.get("lyrics", ""),
            duration=float(request.get("duration", 30.0)),
            steps=int(request.get("steps", 50)),
            cfg_scale=float(request.get("cfg_scale", 7.0)),
            seed=int(request.get("seed", -1)),
        )
        return {
            "audio_base64": base64.b64encode(wav_bytes).decode(),
            "format": "wav",
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "healthy", "model": "stemphonic-stage2d-130k"}


# --- Local test entrypoint ---
@app.local_entrypoint()
def main(prompt: str = "a melodic hip hop beat with warm synths and punchy drums"):
    model = StemphonicInference()
    wav = model.generate.remote(prompt=prompt)
    with open("output.wav", "wb") as f:
        f.write(wav)
    print(f"Saved output.wav ({len(wav)} bytes)")
