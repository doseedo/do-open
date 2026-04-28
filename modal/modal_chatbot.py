"""Doseedo chatbot inference container — Qwen3-14B-AWQ (text) + Moondream 2
(vision) co-resident on a single L4.

Two HTTP endpoints on the same class/container/GPU:

  serve  — @modal.web_server on port 8000, handed straight to vLLM's
           OpenAI-compatible server. URL:
             arlo--doseedo-chatbot-qwenchatbot-serve.modal.run
           Routes: /v1/models, /v1/chat/completions, /v1/completions,
                   /health (existing, unchanged).

  vision — @modal.asgi_app, a thin FastAPI wrapping Moondream 2 for
           reference-image analysis feeding the Plugin Creator pipeline.
           URL:
             arlo--doseedo-chatbot-qwenchatbot-vision.modal.run
           Routes: POST /v1/vision/analyze  {image_base64, prompt, task?}
                   GET  /health

Memory split on the L4 (24 GiB):
    vLLM weights + KV cache at gpu_memory_utilization=0.65  ≈ 15.6 GiB
    Moondream 2 FP16 weights + ViT + activations           ≈  4.0 GiB
    Headroom                                               ≈  4.4 GiB

We dropped vLLM's gpu_memory_utilization from 0.90 → 0.65 and
--max-num-seqs 16 → 4 to leave room. Practical impact: max concurrent
streaming sessions drops from 6–8 to 2–3, which is still well above the
single-user desktop-app load today.

Drop-in replacement for the Anthropic API path that the desktop app currently
uses (logic_engine/chat_server.py → _handle_turn). vLLM's OpenAI-compatible
/v1/chat/completions endpoint is consumed by logic_engine/llm_local.py's
existing OpenAI→Anthropic adapter, so the chat_server streaming loop and the
Electron WS event shape stay unchanged.

Why L4 (not T4): the SYSTEM_PROMPT + 110-tool schema in logic_engine/chat.py
is ~10.3 KT. vLLM's prefix caching keeps that KV cache resident across
requests, but the KV cache itself eats VRAM. L4's 24 GB is already the
minimum for text-only; adding Moondream eats a further ~4 GiB which is why
we've tightened vLLM's budget above.

This is a SECOND L4 independent of modal_stemphonic's L4. vLLM pins the whole
model to GPU memory and does not yield between requests, so the two workloads
can't share a container.

Deploy:
    modal deploy modal/modal_chatbot.py

Prereqs (one-time):
    modal secret create doseedo-chatbot-gate VLLM_API_KEY=$(openssl rand -hex 32)

Test (after deploy):
    curl -s https://<your-deploy>.modal.run/v1/models \
      -H "Authorization: Bearer <VLLM_API_KEY>"

    curl -s https://<your-deploy>.modal.run/v1/chat/completions \
      -H "Authorization: Bearer <VLLM_API_KEY>" \
      -H "Content-Type: application/json" \
      -d '{"model":"qwen3-14b","messages":[{"role":"user","content":"hi"}]}'

    # Vision (separate URL, same container):
    IMG=$(base64 -i some-plugin-screenshot.png)
    curl -s https://<vision-deploy>.modal.run/v1/vision/analyze \
      -H "Authorization: Bearer <VLLM_API_KEY>" \
      -H "Content-Type: application/json" \
      -d "{\"image_base64\":\"$IMG\",\"prompt\":\"List every visible control with its label.\"}"
"""

from __future__ import annotations

import subprocess

import modal

# ---------------------------------------------------------------------------
# Model + port config
# ---------------------------------------------------------------------------
#
# Qwen3-14B-AWQ: 14B params, AWQ 4-bit quantization (~8.5 GB VRAM for weights).
# Non-thinking mode — the 103-tool selection problem wants reliable multi-tool
# reasoning without reaching for a bigger GPU. 7B hallucinates tool names on
# ambiguous requests ("make the bass hit harder" → EQ? compressor? saturation?
# combination?). Qwen3-30B-A3B (MoE, 3B active) is theoretically faster but
# the full 30B expert weights must be resident in VRAM — AWQ puts it at
# ~18–20 GB, too tight on L4 once batch KV cache is added. Would need L40S
# or A100.
#
# Qwen3-14B-AWQ is Apache-2.0 and public on HF — no HF_TOKEN required to
# download. If we ever switch to a gated model, add a huggingface-secret.

MODEL_NAME = "Qwen/Qwen3-14B-AWQ"
SERVED_MODEL_NAME = "qwen3-14b"  # what clients pass as "model" in requests
VLLM_PORT = 8000

# Moondream 2 — 1.93B vision-language model (SigLIP encoder + custom decoder).
# Pinned to a specific revision so cold restores don't silently pick up a new
# `trust_remote_code` modeling file that changes the .query()/.detect() API
# shape. Bump deliberately after smoke-testing analyze() output.
VISION_MODEL_NAME = "vikhyatk/moondream2"
VISION_MODEL_REVISION = "2025-01-09"


# Pydantic body model for the vision endpoint. Must be defined at module
# scope (not inside the @modal.asgi_app method) so FastAPI's pydantic v2
# TypeAdapter can resolve the forward reference. Inside-a-function classes
# trigger PydanticUserError "class-not-fully-defined" at request time.
from typing import Optional  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402


class VisionAnalyzeBody(BaseModel):
    # Accept `image_base64` or alias `image`. `populate_by_name=True` lets
    # pydantic v2 fill either name.
    image_base64: Optional[str] = Field(default=None)
    image: Optional[str] = Field(default=None)
    prompt: Optional[str] = None
    task: str = "query"
    # `object` shadows a builtin — rename field, alias for wire compat.
    obj: Optional[str] = Field(default=None, alias="object")

    model_config = {"populate_by_name": True, "extra": "ignore"}

# HuggingFace cache — first container downloads ~8.5 GB, subsequent containers
# read from the shared volume so scale-up cold start is weight-load bound
# (~15–25 s on L4 PCIe) rather than network bound. Survives container
# recycles so scale-to-zero doesn't re-pay the download cost.
HF_CACHE_VOLUME = modal.Volume.from_name(
    "huggingface-cache-chatbot",
    create_if_missing=True,
)
HF_CACHE_PATH = "/root/.cache/huggingface"

# vLLM torch.compile cache — first cold start compiles Inductor graphs + captures
# CUDA graphs for each batch size (~3 min on L4). Persisting the cache on a
# volume lets later cold starts load the compiled artefacts in ~5–10 s instead
# of recompiling. Without this, every scale-up from zero re-pays the compile
# tax even though the model + GPU + vLLM version haven't changed.
VLLM_CACHE_VOLUME = modal.Volume.from_name(
    "vllm-compile-cache-chatbot",
    create_if_missing=True,
)
VLLM_CACHE_PATH = "/root/.cache/vllm"


# ---------------------------------------------------------------------------
# Image — use upstream vLLM's pre-built Docker image
# ---------------------------------------------------------------------------
#
# Why from_registry instead of pip_install: vLLM has a tangled dependency
# graph with transformers + tokenizers + flashinfer + torch, and pip-resolving
# any specific set ourselves runs into version cross-hatches that surface as
# runtime crashes (e.g. vLLM 0.8.5 hitting `Qwen2Tokenizer has no attribute
# all_special_tokens_extended`, or vLLM 0.9.x double-registering the `aimv2`
# config against newer transformers). The `vllm/vllm-openai` image publishes
# a stack that vLLM's CI actually tested together, so we inherit that instead
# of re-inventing the resolution.
#
# Tag choice: pinned to v0.10.0 — the first tag that cleanly supports
# Qwen3-14B-AWQ end-to-end (Qwen3 model_type registered, tokenizer caching
# path patched, no aimv2 double-registration, hermes tool-call parser stable).
# Bump to chase newer perf at your own risk; smoke-test tool calls after.
#
# HF_HUB_ENABLE_HF_TRANSFER=1 parallelizes the initial weight download over
# ~16 concurrent connections, dropping the first-container init from ~90 s to
# ~25 s on Modal's 10 Gbps egress. The base image ships hf_transfer already.

image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.10.0",
        add_python="3.11",
    )
    .entrypoint([])  # override docker ENTRYPOINT so Modal's runtime boots first
    # Moondream 2 add-ons:
    #   - pillow: image decode for analyze()
    #   - einops: required by moondream's custom modeling_*.py (trust_remote_code)
    #   - accelerate: needed by AutoModel.from_pretrained(..., device_map=)
    #   - fastapi: the vision ASGI endpoint (vllm-openai image ships fastapi
    #     transitively via vLLM, but pin it here so we don't rely on that.)
    #   - transformers + torch: the original version of this file assumed
    #     they came from the vllm/vllm-openai base image, but `add_python=3.11`
    #     installs a fresh Python with its own site-packages, so we must
    #     install transformers explicitly. torch 2.11 was pulled in via the
    #     accelerate/pillow dependency graph. Pin transformers ≥4.44 for the
    #     moondream modeling hooks.
    #   - pyvips + libvips system lib: moondream2's custom modeling file
    #     imports pyvips for its fast image-tiling preprocessor. Without it
    #     the HF Auto loader raises "This modeling file requires the
    #     following packages that were not found in your environment: pyvips"
    #     and setup_gpu_state() fails right after vLLM binds port 8000.
    .apt_install("libvips")
    .pip_install(
        "pillow",
        "einops",
        "accelerate",
        "fastapi",
        "transformers>=4.44,<5",
        "pyvips",
        # hf_transfer: required because we set HF_HUB_ENABLE_HF_TRANSFER=1
        # below. Without it, huggingface_hub raises
        # "Fast download using 'hf_transfer' is enabled but hf_transfer
        # package is not available" mid-runtime when Moondream's
        # from_pretrained lazy-loads config/modeling_*.py.
        "hf_transfer",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # vLLM worker init on Modal: spawn avoids the fork+CUDA init
            # deadlock that shows up as "Cannot re-initialize CUDA in forked
            # subprocess" during engine startup.
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HOME": HF_CACHE_PATH,
        }
    )
)

app = modal.App("doseedo-chatbot")


# ---------------------------------------------------------------------------
# Inference class — mirrors modal_stemphonic.Stemphonic structure
# ---------------------------------------------------------------------------
#
# min_containers=0 + enable_memory_snapshot=True:
#   Scale-to-zero matches the stemphonic container's philosophy — no
#   always-on baseline charge. First request in a cold window eats
#   image pull + GPU attach + vLLM subprocess boot (~35 s total without
#   snapshot). enable_memory_snapshot captures the post-import Python
#   process state during setup_cpu_state(), so cold restores skip
#   ~5–8 s of interpreter + module init. Weight load still happens on
#   the GPU (cannot be snapshotted), so full cold start lands around
#   ~25–30 s. Acceptable for a desktop app that already shows a "starting
#   Modal vLLM…" system message on WS connect.
#
# max_containers=1:
#   Matches stemphonic. Single-user desktop deployment for now; each L4
#   handles ~6–8 concurrent streaming sessions via vLLM batching anyway.
#   Bump once we have real multi-user load.
#
# scaledown_window=60*15:
#   15 min warm window, same as stemphonic. One active chat session keeps
#   it hot; past 15 min of idle we scale down and accept the cold start
#   penalty on the next request.
#
# @modal.concurrent(max_inputs=16):
#   Matches vLLM's --max-num-seqs 16 below. Without this, Modal would only
#   forward one HTTP request at a time to the container and vLLM's
#   continuous batching would be wasted.

@app.cls(
    image=image,
    gpu="L4",
    volumes={
        HF_CACHE_PATH: HF_CACHE_VOLUME,
        VLLM_CACHE_PATH: VLLM_CACHE_VOLUME,
    },
    secrets=[
        # VLLM_API_KEY — bearer token that clients (desktop app, curl probes)
        # must send as `Authorization: Bearer <key>`. vLLM's --api-key flag
        # enforces this at the HTTP layer, so an unauthenticated internet
        # request gets a 401 before touching the engine. Create with:
        #   modal secret create doseedo-chatbot-gate VLLM_API_KEY=$(openssl rand -hex 32)
        modal.Secret.from_name("doseedo-chatbot-gate"),
    ],
    timeout=60 * 10,
    scaledown_window=60 * 15,
    min_containers=0,
    max_containers=1,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=16)
class QwenChatbot:
    @modal.enter(snap=True)
    def setup_cpu_state(self):
        """Snapshot-phase init. CPU machine, no GPU attached.

        Deliberately minimal: we do NOT import vllm here because its import
        chain touches torch.cuda and can fail or spuriously allocate on a
        CPU-only host. The goal is just to bake a warm-interpreter snapshot
        so cold restores skip generic python bootstrap. vLLM starts fresh
        as a subprocess in setup_gpu_state().

        We DO pre-download moondream2's weights + modeling files into the
        HF cache volume here: it's a pure filesystem operation (no GPU),
        saves ~30 s on the first cold start per container where the volume
        isn't populated yet, and snapshot-captures the Python-side
        transformers imports we'll touch in setup_gpu_state().
        """
        import urllib.request  # noqa: F401
        import urllib.error  # noqa: F401

        # Pre-warm the HF cache for moondream so setup_gpu_state doesn't
        # block on network. Idempotent: no-op after first run because the
        # volume persists blobs across deploys.
        from huggingface_hub import snapshot_download
        snapshot_download(
            VISION_MODEL_NAME,
            revision=VISION_MODEL_REVISION,
            cache_dir=HF_CACHE_PATH,
        )

    @modal.enter(snap=False)
    def setup_gpu_state(self):
        """GPU phase. Launch `vllm serve`, wait for ready, then load Moondream.

        Ordering matters: vLLM greedily claims `gpu_memory_utilization`
        fraction of TOTAL VRAM at init regardless of what else is on the
        card, so Moondream must load AFTER vLLM is settled — it takes from
        the ~8 GiB vLLM left behind, not from its reservation.
        """
        import os
        import shlex
        import time
        import urllib.request
        import urllib.error

        # Flag set below and consulted by serve() so the decorator body
        # returns immediately on restore (subprocess is already running).
        self._vllm_ready = False
        self._moondream = None
        self._moondream_tokenizer = None

        # --enable-prefix-caching:
        #   THE critical flag. vLLM's equivalent of Anthropic's cache_control —
        #   every request that shares the SYSTEM_PROMPT + TOOLS prefix reuses
        #   its KV cache instead of recomputing prefill. Our prefix is ~10.3 KT
        #   and accounts for ~95% of any given request's input, so without
        #   this every turn pays ~10.3 KT of prefill (≈ 400 ms on L4). With
        #   the flag: ~0 ms for cached prefix, prefill only on the tail.
        #
        # --enable-auto-tool-choice + --tool-call-parser hermes:
        #   Qwen3 uses the Hermes tool-call format. vLLM's hermes parser lifts
        #   those spans out of the completion stream and emits them as OpenAI
        #   tool_calls deltas in SSE, which the OllamaClient adapter in
        #   logic_engine/llm_local.py (lines 318–360) already translates back
        #   into Anthropic content_block_start(tool_use) + input_json_delta.
        #
        # --max-model-len 28672:
        #   Was 32768; reduced to fit the KV cache inside
        #   gpu_memory_utilization=0.65. At 32768 vLLM computed
        #   5.00 GiB needed vs 4.67 GiB available and refused to
        #   start. The estimated max it gave was 30560 tokens — we
        #   round down to 28672 (28K) for margin + alignment to 2048.
        #   Still enough for SYSTEM_PROMPT (10.3 KT) + multi-turn
        #   context (~8 KT) + output budget (~10 KT).
        #
        # --gpu-memory-utilization 0.65 (was 0.90):
        #   Gives back ~6 GiB of VRAM so Moondream 2 can co-reside. vLLM
        #   weights are unchanged at ~9.4 GiB; the cut comes out of the KV
        #   cache budget, dropping us from ~66 KT of KV headroom to ~25 KT.
        #   At max-num-seqs=4 that's still ~6 KT per concurrent session,
        #   well above typical chat turn requirements.
        #
        # --max-num-seqs 4 (was 16):
        #   Matches the reduced KV cache. @modal.concurrent(max_inputs=16)
        #   upstream is deliberately higher — Moondream calls bypass vLLM,
        #   so the container can handle up to 16 total in-flight requests
        #   (vision + text) without Modal queueing.

        api_key = os.environ["VLLM_API_KEY"]

        cmd = (
            f"vllm serve {MODEL_NAME}"
            f" --host 0.0.0.0"
            f" --port {VLLM_PORT}"
            f" --served-model-name {SERVED_MODEL_NAME}"
            f" --max-model-len 28672"
            f" --enable-prefix-caching"
            f" --enable-auto-tool-choice"
            f" --tool-call-parser hermes"
            f" --gpu-memory-utilization 0.65"
            f" --max-num-seqs 4"
            f" --api-key {shlex.quote(api_key)}"
        )

        self._vllm_proc = subprocess.Popen(shlex.split(cmd))

        # Poll /health until vLLM is ready. The server only starts listening
        # after weights are loaded + torch.compile Inductor passes + CUDA
        # graph capture for each of the --max-num-seqs buckets, which takes
        # ~4 min on a cold first boot (no compile cache) and ~30 s when the
        # VLLM_CACHE_VOLUME has been populated. Deadline generous enough to
        # cover the first-boot case; persistent containers warm up fast.
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self._vllm_proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {self._vllm_proc.returncode} "
                    f"before binding port {VLLM_PORT}"
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{VLLM_PORT}/health",
                    timeout=2,
                ) as r:
                    if r.status == 200:
                        self._vllm_ready = True
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(1.0)
        else:
            raise RuntimeError(
                f"vllm did not become healthy within 10 min on port {VLLM_PORT}"
            )

        # ── Moondream 2 (vision) ──
        # FP16 load (no bitsandbytes): ~3.7 GiB weights, ~0.3 GiB ViT workspace,
        # activations ~0.5 GiB per inference. With vLLM already at ~15.6 GiB
        # reserved on a 24 GiB L4, we're at ~20 GiB used, ~4 GiB headroom.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._moondream_tokenizer = AutoTokenizer.from_pretrained(
            VISION_MODEL_NAME,
            revision=VISION_MODEL_REVISION,
            cache_dir=HF_CACHE_PATH,
        )
        self._moondream = AutoModelForCausalLM.from_pretrained(
            VISION_MODEL_NAME,
            revision=VISION_MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            cache_dir=HF_CACHE_PATH,
        )
        self._moondream.eval()

    @modal.exit()
    def teardown(self):
        """Terminate the vLLM subprocess on container recycle."""
        proc = getattr(self, "_vllm_proc", None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    @modal.web_server(port=VLLM_PORT, startup_timeout=60 * 10)
    def serve(self):
        """Modal proxies HTTP for this method to localhost:VLLM_PORT.

        setup_gpu_state() has already bound the port; this decorator body is
        effectively a no-op that tells Modal "the port-listener is live."
        """
        pass

    @modal.asgi_app()
    def vision(self):
        """Moondream 2 vision endpoint — separate URL, same container.

        Routes:
          GET  /health              — public, reports moondream ready state
          POST /v1/vision/analyze   — Bearer-gated. Body:
              {
                "image_base64": "...",       # required, raw base64 or data URL
                "prompt":  "describe this",  # used when task="query"
                "task":    "query" | "detect" | "point",  # default "query"
                "object":  "knob"            # used when task in {detect, point}
              }
            Response:
              query:  {"task":"query",  "answer": "..."}
              detect: {"task":"detect", "objects": [{x_min,y_min,x_max,y_max}, ...]}
              point:  {"task":"point",  "points":  [{x,y}, ...]}

        Auth: same VLLM_API_KEY bearer the text side uses, checked in-app.
        vLLM handles its own --api-key enforcement; we replicate the check
        here so clients have a single credential.
        """
        import base64
        import io
        import os

        from fastapi import FastAPI, HTTPException, Header, Body
        from fastapi.responses import JSONResponse
        from PIL import Image

        api = FastAPI(title="doseedo-chatbot-vision", version="1.0.0")
        expected_key = os.environ["VLLM_API_KEY"]

        def _require_auth(authorization: str) -> None:
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="missing bearer")
            if authorization[len("Bearer "):] != expected_key:
                raise HTTPException(status_code=401, detail="invalid key")

        def _decode_image(b64: str) -> Image.Image:
            # Accept raw base64 or "data:image/png;base64,..." data URLs.
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            try:
                raw = base64.b64decode(b64)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"invalid image: {e}")

        @api.get("/health")
        async def health():
            ok = self._moondream is not None
            return JSONResponse(
                {"vision_ready": ok, "model": VISION_MODEL_NAME},
                status_code=200 if ok else 503,
            )

        @api.post("/v1/vision/analyze")
        async def analyze(
            body: VisionAnalyzeBody = Body(...),
            authorization: str = Header(default=""),
        ):
            _require_auth(authorization)

            image_b64 = body.image_base64 or body.image
            if not image_b64:
                raise HTTPException(status_code=400, detail="missing image_base64")
            img = _decode_image(image_b64)

            task = (body.task or "query").lower()

            import torch

            with torch.inference_mode():
                if task == "query":
                    prompt = body.prompt or "Describe this image."
                    result = self._moondream.query(img, prompt)
                    answer = result.get("answer") if isinstance(result, dict) else str(result)
                    return {"task": "query", "answer": answer}

                if task == "detect":
                    if not body.obj:
                        raise HTTPException(status_code=400, detail="detect task requires 'object'")
                    result = self._moondream.detect(img, body.obj)
                    return {"task": "detect", "objects": result.get("objects", [])}

                if task == "point":
                    if not body.obj:
                        raise HTTPException(status_code=400, detail="point task requires 'object'")
                    result = self._moondream.point(img, body.obj)
                    return {"task": "point", "points": result.get("points", [])}

                raise HTTPException(
                    status_code=400,
                    detail=f"unknown task {task!r}; expected query|detect|point",
                )

        return api
