"""
Modal deployment for stemphonic_server.py (doseedo backend).

Migration target: replace my-a100-80gb-vm:8765 (backend10) with a Modal
A100-80GB container serving the existing Flask app unchanged. Zero edits
to stemphonic_server.py — we symlink all hardcoded /scratch/... paths to
image-baked + volume-mounted locations in @enter().

Deploy flow (run from my-a100-80gb-vm where weights live on /scratch):

  1. Push weights to the models volume (one-time, ~34 GB stemphonic + 14 GB
     ACE-Step + misc):

       modal volume put stemphonic-models /scratch/stemphonic/checkpoints      stemphonic-checkpoints
       modal volume put stemphonic-models /scratch/ACE-Step-1.5/checkpoints    ace-step-checkpoints
       modal volume put stemphonic-models /scratch/piper_voices                piper-voices
       modal volume put stemphonic-models /scratch/audio_separator_models      audio-separator-models
       modal volume put stemphonic-models /scratch/latent_editor_ckpts         latent-editor-ckpts
       modal volume put stemphonic-models /scratch/latent_soundfonts           latent-soundfonts
       modal volume put stemphonic-models /scratch/onnx                        onnx

     (There's no /scratch/stage2d_step130000.pt on the VM — the legacy
     symlink target doesn't exist. The default checkpoint is loaded via
     /scratch/stemphonic/checkpoints/stage2d-130k.pt instead, which ships
     inside the stemphonic-checkpoints volume entry.)

     Verify total size after upload matches expectations (~50 GB):
       modal volume ls stemphonic-models

  2. Deploy the service:

       modal deploy modal_stemphonic.py

  3. Smoke test against the raw Modal URL (NOT doseedo.com) while tailing
     logs in another terminal:

       modal app logs doseedo-stemphonic &
       curl https://arlo--doseedo-stemphonic-stemphonic-wsgi.modal.run/health

  4. Only after parity is proven against every critical endpoint, swap the
     GCLB backend via Internet NEG (LAUNCH_AUDIT.md migration step 5).

Operational knobs (tuned for launch-week cost and simplicity, NOT steady-state):

  - min_containers=0         scale to zero, no idle A100 burn
  - max_containers=1         force single container so Flask's in-process
                             /task/<id> polling keeps working without
                             distributed state. Revisit after moving task
                             state to modal.Dict or user_data volume.
  - scaledown_window=60*15   keep warm 15 min between reqs so a user
                             mid-session doesn't cold-start
  - enable_memory_snapshot   stemphonic_server.py is genuinely CPU-clean at
                             import time (handler=None, model=None, only
                             torch.load(..., map_location='cpu') calls on
                             preset files at module scope). We import the
                             whole server in snap=True and snapshot the
                             fully-initialized module. On cold restore the
                             only remaining cost is lazy model-to-GPU load,
                             which we trigger eagerly in snap=False.

Known limitations for launch version:

  - Checkpoint hot-swap via /api/generate-stemphonic/checkpoint only works
    for checkpoints already present on /models (7 of the 10 in the
    registry). The 3 missing ones (stage2c-latest, stage2c-best,
    stage2b-merged) would require google-cloud-sdk + a GCP secret for the
    runtime gsutil cp call. Not needed for launch; add later if users
    actually switch checkpoints.
  - No chat_agent_server.py migration yet. Second Modal app, separate file,
    same pattern. Do it after this one is green.
"""

import os
import platform
from pathlib import Path

import modal

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Build-time path resolution — Mac vs GPU server
# ---------------------------------------------------------------------------
# This file is deployed from two machines:
#   GPU server (my-a100-80gb-vm): /home/arlo/do2   + /scratch/ACE-Step-1.5
#   Mac (doseedo-desktop):        ~/Downloads/Do   + ~/Downloads/ACE-Step
#
# Soundfonts + latent checkpoints no longer need to be present on the deploy
# machine — they live on the `stemphonic-models` Modal volume and get
# symlinked into /scratch/... at container start (see @enter below).
_IS_MAC = platform.system() == "Darwin"

if _IS_MAC:
    _DO2      = Path.home() / "Downloads/Do"
    _ACE_STEP = Path.home() / "Downloads/ACE-Step"
    # On Mac the harmonymodule tree lives inside the Do repo.
    _HARMONYMODULE = _DO2 / "home/arlo/harmonymodule"
else:
    _DO2      = Path("/home/arlo/do2")
    _ACE_STEP = Path("/scratch/ACE-Step-1.5")
    _HARMONYMODULE = Path("/scratch/Do/harmonymodule")


# ---------------------------------------------------------------------------
# R2 (Cloudflare S3-compatible) helper — lazy-init client + upload + presign.
# ---------------------------------------------------------------------------
# Why here (module scope, not inside the class):
#   Modal images are monolithic — a single .py file defines both the build
#   graph (image.pip_install, secrets, etc.) AND the container runtime entry
#   points. Keeping R2 helpers at module scope means the exact same code
#   object runs inside the container (via setup_gpu_state importing us as
#   __main__) and is trivially testable from a local REPL when AWS_* or R2_*
#   env vars happen to be set.
#
# Env vars (set via the `doseedo-r2` Modal secret):
#   R2_ACCESS_KEY_ID      — Cloudflare R2 token access key
#   R2_SECRET_ACCESS_KEY  — Cloudflare R2 token secret
#   R2_ENDPOINT_URL       — https://<account>.r2.cloudflarestorage.com
#   R2_BUCKET             — bucket name (e.g. doseedo-media)
#
# If ANY of these is missing we log a warning and no-op. This keeps staging
# deployments without the secret from crashing generation on every request —
# they just skip persistence and continue to serve via send_file.
#
# Mirrors the client config used in auth-service/app/storage.py:_r2_client.
_r2_client_cache = {"client": None, "checked": False}


def _r2_enabled() -> bool:
    return bool(
        os.environ.get("R2_ACCESS_KEY_ID")
        and os.environ.get("R2_SECRET_ACCESS_KEY")
        and os.environ.get("R2_ENDPOINT_URL")
        and os.environ.get("R2_BUCKET")
    )


def _r2_client():
    """Lazy boto3 S3 client targeting R2. Returns None when unconfigured."""
    if _r2_client_cache["client"] is not None:
        return _r2_client_cache["client"]
    if _r2_client_cache["checked"]:
        # We already checked and R2 is unconfigured — don't spam warnings.
        return None
    _r2_client_cache["checked"] = True
    if not _r2_enabled():
        import logging
        logging.getLogger("stemphonic.r2").warning(
            "R2 env vars missing (need R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, "
            "R2_ENDPOINT_URL, R2_BUCKET) — generation persistence disabled"
        )
        return None
    try:
        import boto3
        from botocore.config import Config
        _r2_client_cache["client"] = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3},
            ),
        )
        return _r2_client_cache["client"]
    except Exception as e:
        import logging
        logging.getLogger("stemphonic.r2").exception(
            "R2 client init failed: %s — persistence disabled for this container", e
        )
        return None


def _r2_upload_file(local_path: str, key: str, content_type: str) -> str | None:
    """Upload a local file to R2 at `key` with the given Content-Type.

    Returns the key on success, None on failure or when R2 is unconfigured.
    Intended to be called inline from Flask after_request hooks — every
    failure is logged and swallowed so generation responses are never
    blocked on R2 availability.
    """
    client = _r2_client()
    if client is None:
        return None
    bucket = os.environ.get("R2_BUCKET")
    if not bucket:
        return None
    try:
        extra = {"ContentType": content_type} if content_type else {}
        client.upload_file(local_path, bucket, key, ExtraArgs=extra)
        return key
    except Exception as e:
        import logging
        logging.getLogger("stemphonic.r2").exception(
            "R2 upload failed key=%s local=%s: %s", key, local_path, e
        )
        return None


def _r2_generate_download_url(key: str, expires: int = 3600) -> str | None:
    """Presigned GET URL for `key`. Returns None when unconfigured/errored."""
    client = _r2_client()
    if client is None:
        return None
    bucket = os.environ.get("R2_BUCKET")
    if not bucket:
        return None
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
    except Exception as e:
        import logging
        logging.getLogger("stemphonic.r2").exception(
            "R2 presign failed key=%s: %s", key, e
        )
        return None

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
models_vol = modal.Volume.from_name("stemphonic-models", create_if_missing=True)
user_data_vol = modal.Volume.from_name("stemphonic-user-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# Torch pin strategy (per second opinion): the VM's pip freeze captures
# torch==2.11.0 + CUDA 13 nvidia-* wheels, which are too bleeding-edge for
# Modal's A100 host drivers. We filter those out of the requirements file
# and install torch==2.4.1 + cu121 explicitly, after the frozen deps, so
# the explicit install wins on any version conflict.
#
# /home/arlo/do2/stemphonic_requirements_modal.txt was produced by:
#   grep -vE '^(torch|torchaudio|torchvision|torchao|torchcodec|triton)==' \
#     stemphonic_requirements.txt | grep -v '^nvidia-' > stemphonic_requirements_modal.txt
def _ignore_patterns(*extra):
    # Modal's FilePatternMatcher treats bare patterns like "__pycache__" and
    # "*.pyc" as TOP-LEVEL-ONLY. To exclude these at any nesting depth we
    # have to add the "**/" prefixed variants explicitly. Verified empirically:
    #   m = FilePatternMatcher("__pycache__")
    #   m(Path("foo/__pycache__/bar.pyc")) -> False  # does NOT match
    #   m(Path("__pycache__/bar.pyc"))     -> True   # top level only
    # Without the ** variants, nested .pyc files end up copied, and if any
    # long-running Python process (e.g. a training job, a server) is importing
    # modules from the source tree it regenerates those .pyc files mid-copy,
    # triggering Modal's "file modified during build" error.
    base = [
        "__pycache__", "**/__pycache__", "**/__pycache__/**",
        "*.pyc", "**/*.pyc",
        ".git", "**/.git", "**/.git/**",
        ".venv", "**/.venv", "**/.venv/**",
        ".pytest_cache", "**/.pytest_cache", "**/.pytest_cache/**",
        "*.log", "**/*.log",
    ]
    return base + list(extra)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # NOTE: fluidsynth + fluid-soundfont-gm deliberately omitted.
        # The stemphonic_server.py MIDI→WAV render pipeline has moved to
        # the latent soundfont path (latent_soundfont.synth, reading per-
        # instrument .pt latents from /scratch/latent_soundfonts). The only
        # 'fluidsynth' reference left in the server is a stale comment on
        # line 735, and the INSTRUMENT_SOUNDFONTS sf2 dict (lines 117-137)
        # is never iterated against a fluidsynth subprocess. Dropping this
        # apt package saves ~400 MB of image size.
        "ffmpeg",
        "libsndfile1",
        "libsndfile1-dev",
        "libgl1",
        "libglib2.0-0",
        "git",
        "build-essential",
    )
    # madmom==0.16.1 ships only an sdist, and its setup.py does
    # `from Cython.Build import cythonize` at import time. pip's per-package
    # build isolation means each sdist build runs in a fresh venv, so even
    # though Cython==3.2.4 is pinned in the requirements file it is not
    # visible when madmom's setup.py executes. Fix: pre-install Cython +
    # numpy at the image level, then install madmom with build isolation
    # disabled. The main requirements pass below will see madmom already
    # satisfied and skip rebuilding it.
    .pip_install("Cython==3.2.4", "numpy==1.26.4")
    .pip_install("madmom==0.16.1", extra_options="--no-build-isolation")
    # boto3 — R2 (S3-compatible) client for persisting user-facing generation
    # outputs to Cloudflare R2 so they survive Modal container recycles.
    # Pinned to match auth-service/requirements.txt so behavior is identical
    # across services (presigned URL format, retry policy, etc.).
    .pip_install("boto3==1.35.80")
    # Frozen pip deps minus torch/nvidia — installed with --no-deps.
    #
    # WHY --no-deps: stemphonic_requirements_modal.txt is a full pip freeze,
    # which means every transitive dependency is ALREADY explicitly pinned as
    # its own line. Letting pip re-resolve transitives with its strict resolver
    # on this file surfaces a cascade of "package X declares it needs Y>=N
    # but the file pins Y==M" contradictions. These contradictions didn't
    # break the original local install (older/legacy resolver, install-order
    # quirks, or the declared constraint was stricter than the actual runtime
    # requirement) but modern pip refuses to proceed. Examples hit so far:
    #   - audio-separator==0.44.1 declares numpy>=2, file pins numpy==1.26.4
    #   - onnx==1.21.0 declares ml_dtypes>=0.5.0, file pins ml-dtypes==0.2.0
    # Since every transitive is already listed, --no-deps is safe: pip just
    # installs each line verbatim without re-resolving. Any missing transitive
    # would mean the local env was also broken, which it isn't.
    #
    # Side effect: beat-this's unpinned `torch>=2` constraint is also ignored,
    # so pip won't pull torch 2.11 during this pass. torch==2.4.1 gets
    # installed explicitly in the next step.
    #
    # audio-separator is commented out in the requirements file (line 14).
    # It was used by MDX23C-DrumSep; removed 2026-04-11 along with
    # basic-pitch and demucs in favour of the latent student models
    # (latent_demucs / latent_drumsep / latent_pitch / latent_visual).
    .pip_install_from_requirements(
        os.path.join(_HERE, "requirements.txt"),
        extra_options="--no-deps",
    )
    # Explicit torch install — wins any version conflict with transitive deps.
    #
    # WHY 2.5.1 (not 2.4.1, not 2.11.0): torch 2.4.1 ships a stale
    # torch/onnx/_internal/fx/op_validation.py that uses @_beartype.beartype
    # on a function with a forward-ref type hint `Sequence[onnxscript.values.ParamSchema]`.
    # onnxscript >= 0.6.0 removed the `ParamSchema` class, so the beartype
    # decorator crashes at module import time whenever anything transitively
    # imports torch.onnx (e.g. transformers.integrations.flex_attention runs
    # @torch.compiler.disable as a class decorator, which imports torch._dynamo
    # which imports torch.onnx.operators). Verified: torch 2.5.1 dropped the
    # entire _beartype.py + op_validation.py files from torch.onnx._internal.fx,
    # so the broken decoration is gone. Local env runs torch 2.11.0+cu130
    # (also works, different reason) but 2.11 requires CUDA 13 drivers which
    # may or may not be available on all Modal GPU classes. 2.5.1+cu121 is
    # the latest torch built for CUDA 12.1 and is the minimum that fixes the
    # onnxscript breakage without pulling in a CUDA 13 runtime dependency.
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # ACE-Step source — exclude the 14 GB checkpoints subdir (goes on volume)
    .add_local_dir(
        str(_ACE_STEP),
        remote_path="/opt/ACE-Step-1.5",
        ignore=_ignore_patterns("checkpoints", "checkpoints/**"),
        copy=True,
    )
    # do2 source (32 MB) — stemphonic_trainer, time-sig-editor, harmony, etc.
    .add_local_dir(
        str(_DO2),
        remote_path="/opt/do2",
        ignore=_ignore_patterns(),
        copy=True,
    )
    # harmonymodule (420 KB) — legacy /scratch/Do/harmonymodule import path.
    # On Mac this lives inside the Do git repo at home/arlo/harmonymodule;
    # on the GPU server it's /scratch/Do/harmonymodule.
    .add_local_dir(
        str(_HARMONYMODULE),
        remote_path="/opt/harmonymodule",
        ignore=_ignore_patterns(),
        copy=True,
    )
    # Soundfonts, latent checkpoints, and the PANNs label CSV all live in
    # the `stemphonic-models` Modal volume (uploaded once from GCS).
    # @enter symlinks `/scratch/soundfonts`, `/scratch/latent_*_ckpts/*`,
    # and `/root/panns_data/` into the volume so the container sees the
    # same filesystem layout that stemphonic_server.py expects — without
    # requiring any of those files on the deploy machine.
    .add_local_file(
        str(_DO2 / "stemphonic_server.py"),
        remote_path="/opt/do2/stemphonic_server.py",
        copy=True,
    )
    # TODO(post-2026-04-11 wipe): latent whisper/lyric student vocal (190 MB)
    # — same recovery status as drumsep/panns above. /api/transcribe-vocals
    # will 500 until re-baked. Generation path doesn't use it.
    # .add_local_file(
    #     "/scratch/latent_whisper_student/ckpts_vocal/student_final.pt",
    #     remote_path="/opt/latent_ckpts/latent_whisper/student_final.pt",
    #     copy=True,
    # )
)

app = modal.App("doseedo-stemphonic")


# ---------------------------------------------------------------------------
# Stemphonic class — the actual inference container
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="L4",                  # 24 GB VRAM; measured peak ~9 GB on 30s/50-step gen
    volumes={
        "/models": models_vol,
        "/user_data": user_data_vol,
    },
    secrets=[
        # Provides AUTH_SERVICE_URL, INTERNAL_SECRET, DISABLE_ALL_GENERATION.
        # Create with:
        #   modal secret create doseedo-gate \
        #     AUTH_SERVICE_URL=https://doseedo-auth-wd7h2yezlq-uc.a.run.app \
        #     INTERNAL_SECRET=<same value as auth-service INTERNAL_SECRET secret> \
        #     DISABLE_ALL_GENERATION=false
        modal.Secret.from_name("doseedo-gate"),
        # Provides R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL,
        # R2_BUCKET. Used by _r2_upload_file / _r2_generate_download_url in
        # the after_request hooks to persist generation outputs to Cloudflare
        # R2 so users can replay them cross-device after the Modal container
        # recycles. The same INTERNAL_SECRET from doseedo-gate is reused to
        # call the auth-service POST /api/generations registration endpoint.
        modal.Secret.from_name("doseedo-r2"),
    ],
    timeout=60 * 30,           # 30 min per request (long ACE-Step gens)
    scaledown_window=60 * 15,  # 15 min warm window
    min_containers=1,          # keep 1 warm so users never hit ~80s cold start
                               # (Cloud Run inference-proxy still scales to
                               # zero but its cold start is ~1-3s, not 80s)
    max_containers=1,          # single container for in-process /task polling
    enable_memory_snapshot=True,
)
class Stemphonic:
    @modal.enter(snap=True)
    def setup_cpu_state(self):
        """Snapshot-phase init. Runs on a CPU machine WITHOUT GPU attached.

        stemphonic_server.py is genuinely CPU-clean at module scope:
          - handler=None, module=None at line 195-196
          - load_model() is only called from __main__ guard (line 4966)
            or lazily from Flask routes
          - all module-scope torch.load() calls use map_location='cpu'

        So we can import the whole server here, and the snapshot will
        include the fully-initialized Flask app + all preset tensors
        loaded to CPU. Cold restore from snapshot is ~5s vs ~60s without.

        The first real request still triggers lazy model-to-GPU load, but
        we proactively do that in setup_gpu_state() below.
        """
        import os, sys, pathlib

        # ---- Build the symlink tree that makes stemphonic_server's
        # ---- hardcoded /scratch/... and /mnt/data/... paths resolve ----

        os.makedirs("/scratch", exist_ok=True)
        os.makedirs("/mnt/data/system_home/arlo", exist_ok=True)
        os.makedirs("/cache", exist_ok=True)
        os.makedirs("/user_data/latents", exist_ok=True)
        os.makedirs("/user_data/stemphonic_outputs", exist_ok=True)
        os.makedirs("/user_data/stemphonic_outputs/separations", exist_ok=True)
        os.makedirs("/user_data/stemphonic_outputs/scores", exist_ok=True)
        os.makedirs("/user_data/stemphonic_outputs/timbre_cache", exist_ok=True)

        def link(src, target):
            src_p = pathlib.Path(src)
            if src_p.is_symlink() or src_p.exists():
                return
            src_p.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(target, src)

        # Source code dirs → image-baked locations
        link("/mnt/data/system_home/arlo/do2", "/opt/do2")
        link("/home/arlo/do2", "/opt/do2")
        os.makedirs("/scratch/Do", exist_ok=True)
        link("/scratch/Do/harmonymodule", "/opt/harmonymodule")

        # ACE-Step: source baked (minus checkpoints), checkpoints on volume.
        # Can't just symlink the whole /scratch/ACE-Step-1.5 dir because we
        # need /scratch/ACE-Step-1.5/checkpoints to be a distinct volume
        # link. Build a mirror dir with per-child symlinks.
        os.makedirs("/scratch/ACE-Step-1.5", exist_ok=True)
        for child in pathlib.Path("/opt/ACE-Step-1.5").iterdir():
            link(f"/scratch/ACE-Step-1.5/{child.name}", str(child))
        link("/scratch/ACE-Step-1.5/checkpoints", "/models/ace-step-checkpoints")

        # Stemphonic: server expects /scratch/stemphonic/checkpoints.
        # Mirror with a real dir + symlink to volume.
        os.makedirs("/scratch/stemphonic", exist_ok=True)
        link("/scratch/stemphonic/checkpoints", "/models/stemphonic-checkpoints")

        # NOTE: no /scratch/stage2d_step130000.pt symlink — that legacy
        # single-file target doesn't exist on the VM either (the server's
        # _ensure_ckpt_downloaded has an os.path.exists() fallback so it's
        # harmless, and the default ckpt loads via
        # /scratch/stemphonic/checkpoints/stage2d-130k.pt instead).

        # Soundfonts: served from volume (latent-soundfonts), not baked.
        link("/scratch/soundfonts", "/models/latent-soundfonts")

        # Other weight dirs on volume
        link("/scratch/piper_voices",           "/models/piper-voices")
        # audio_separator_models removed 2026-04-11 — MDX23C-DrumSep
        # replaced by latent_drumsep student baked into image.
        link("/scratch/latent_editor_ckpts",    "/models/latent-editor-ckpts")
        link("/scratch/latent_soundfonts",      "/models/latent-soundfonts")
        link("/scratch/onnx",                   "/models/onnx")

        # Latent student model checkpoints — served from volume
        # (latent-ckpts/*), not baked. The runtime still reads them from
        # /scratch/latent_*_ckpts/*_final.pt, so we mirror the expected
        # layout with per-file symlinks into /models/latent-ckpts/.
        os.makedirs("/scratch/latent_pitch_ckpts",   exist_ok=True)
        os.makedirs("/scratch/latent_drumsep_ckpts", exist_ok=True)
        os.makedirs("/scratch/latent_visual_ckpts",  exist_ok=True)
        os.makedirs("/scratch/latent_demucs",        exist_ok=True)
        link("/scratch/latent_pitch_ckpts/pitch_final.pt",
             "/models/latent-ckpts/latent_pitch/pitch_final.pt")
        link("/scratch/latent_drumsep_ckpts/drumsep_final.pt",
             "/models/latent-ckpts/latent_drumsep/drumsep_final.pt")
        link("/scratch/latent_visual_ckpts/latent_visual_final.pt",
             "/models/latent-ckpts/latent_visual/latent_visual_final.pt")
        link("/scratch/latent_demucs/distill_final.pt",
             "/models/latent-ckpts/latent_demucs/distill_final.pt")
        os.makedirs("/scratch/latent_panns_student/ckpts", exist_ok=True)
        link("/scratch/latent_panns_student/ckpts/panns_final.pt",
             "/models/latent-ckpts/latent_panns/panns_final.pt")
        os.makedirs("/scratch/latent_whisper_student/ckpts_vocal", exist_ok=True)
        link("/scratch/latent_whisper_student/ckpts_vocal/student_final.pt",
             "/models/latent-ckpts/latent_whisper/student_final.pt")
        # panns_inference loads its label CSV from /root/panns_data/.
        os.makedirs("/root/panns_data", exist_ok=True)
        link("/root/panns_data/class_labels_indices.csv",
             "/models/latent-ckpts/panns_data/class_labels_indices.csv")

        # User-durable writable dirs (volume)
        link("/scratch/cache/latents",      "/user_data/latents")
        link("/scratch/stemphonic_outputs", "/user_data/stemphonic_outputs")

        # Ephemeral writable dirs (container-local, fine to lose between
        # cold starts — each is scratch space for a single request)
        for ephemeral in [
            "/scratch/cache/drumsep_onsets",
            "/scratch/cache/whisper_tmp",
            "/scratch/cache/chord_tmp",
            "/scratch/cache/extract_midi",
            "/scratch/cache/panns_tmp",
            "/scratch/cache/latent_tmp",
        ]:
            pathlib.Path(ephemeral).mkdir(parents=True, exist_ok=True)

        # ---- sys.path mirrors stemphonic_server.py ----
        sys.path.insert(0, "/scratch/ACE-Step-1.5")
        sys.path.insert(0, "/mnt/data/system_home/arlo/do2/stemphonic_trainer")
        sys.path.insert(0, "/mnt/data/system_home/arlo/do2")
        # latent_demucs.runtime does bareword `from distill_model import ...`
        # so the student dir must be on sys.path.
        sys.path.insert(0, "/mnt/data/system_home/arlo/do2/latent_demucs_student")
        # NOTE: latent_panns_student and latent_whisper_student removed from
        # sys.path — both had `student_model.py` and whichever loaded first
        # poisoned sys.modules for the other. Both runtimes now use importlib
        # to load their own student_model.py by filepath.
        sys.path.insert(0, "/scratch/Do/harmonymodule")

        # ---- THE key import. stemphonic_server.py at module scope:
        # ---- - creates Flask app
        # ---- - loads 6 preset .pt files to CPU
        # ---- - declares CKPT_REGISTRY, INSTRUMENT_SOUNDFONTS, etc.
        # ---- - does NOT touch CUDA
        # ---- All of this lands in the snapshot.
        import stemphonic_server  # noqa: F401
        self._server_module = stemphonic_server

    @modal.enter(snap=False)
    def setup_gpu_state(self):
        """Post-snapshot init. Runs on the GPU machine on every cold start.

        Eagerly loads the default checkpoint to GPU so the first real
        request doesn't pay the ~20s model-load cost. Also installs Flask
        before_request hooks for:
          - generation gate (quota enforcement via auth-service)
          - queue-depth canary (max_containers=1 serialization warning)
        """
        import logging, os, threading, urllib.error, urllib.request, json as _json

        stemphonic_server = self._server_module

        # Pre-load the default checkpoint to GPU. Server's load_model()
        # (line 684) calls load_checkpoint(DEFAULT_CKPT_ID='stage2d-130k')
        # and moves everything to cuda:bfloat16. Takes 15-30s.
        try:
            stemphonic_server.load_model()
            logging.info("stemphonic: default checkpoint loaded to GPU")
            # VRAM diagnostic — post-weights, pre-generation. Helps
            # decide whether a smaller GPU (L4 24GB, A100 40GB) would fit.
            try:
                import torch as _t
                _alloc = _t.cuda.memory_allocated() / 1e9
                _resv  = _t.cuda.memory_reserved() / 1e9
                _peak  = _t.cuda.max_memory_allocated() / 1e9
                _total = _t.cuda.get_device_properties(0).total_memory / 1e9
                logging.info(
                    "VRAM post-load: allocated=%.2f GB reserved=%.2f GB "
                    "peak=%.2f GB total=%.2f GB",
                    _alloc, _resv, _peak, _total,
                )
            except Exception as _e:
                logging.warning("VRAM probe failed: %s", _e)
        except Exception as e:
            # Don't crash the container if preload fails — let the first
            # request trigger lazy load and surface the real error there.
            logging.exception("stemphonic: eager load_model() failed: %s", e)

        # ---------------------------------------------------------------------------
        # Generation gate — quota enforcement
        # ---------------------------------------------------------------------------
        # Routes that consume one generation credit per call. Utility routes
        # (classify, extract-midi, detect-chords, encode, download, health) are
        # excluded — they don't burn GPU for inference at generation scale.
        _GATED_ROUTES = frozenset({
            "/api/generate-stemphonic",
            "/api/separate-stemphonic",
            "/api/repaint-meter",
            "/api/regen-stem-for-chord",
            "/api/generate-score-from-video",
            "/api/transcribe-vocals",
            "/separate-stems",
        })

        _AUTH_SERVICE_URL = os.environ.get("AUTH_SERVICE_URL", "").rstrip("/")
        _INTERNAL_SECRET  = os.environ.get("INTERNAL_SECRET", "")
        _DISABLE_ALL      = os.environ.get("DISABLE_ALL_GENERATION", "false").lower() == "true"
        _gate_log         = logging.getLogger("stemphonic.gate")

        @stemphonic_server.app.before_request
        def _generation_gate():
            from flask import request, jsonify, g

            if request.method != "POST":
                return
            if request.path not in _GATED_ROUTES:
                return

            # Kill switch — no auth-service call needed
            if _DISABLE_ALL:
                return jsonify({"error": "Generation is temporarily disabled"}), 503

            if not _AUTH_SERVICE_URL or not _INTERNAL_SECRET:
                _gate_log.warning(
                    "Generation gate not configured (AUTH_SERVICE_URL or "
                    "INTERNAL_SECRET missing) — failing open"
                )
                return

            # Forward whichever identity the caller sent:
            #   - Browser: Authorization: Bearer <jwt> or access_token cookie
            #   - Desktop: X-API-Key: dsk_… (or Authorization: Bearer dsk_…)
            # The auth-service gate (generation_gate.py) accepts all three
            # shapes. We accept the request here if ANY of them is present
            # and let auth-service decide validity.
            auth_header    = request.headers.get("Authorization", "")
            cookie_header  = request.headers.get("Cookie", "")
            access_token   = request.cookies.get("access_token", "")
            api_key_header = request.headers.get("X-API-Key", "")

            if not auth_header and not access_token and not api_key_header:
                return jsonify({"error": "Authentication required for generation"}), 401

            gate_headers = {
                "Content-Type": "application/json",
                "X-Internal-Secret": _INTERNAL_SECRET,
            }
            if auth_header:
                gate_headers["Authorization"] = auth_header
            if cookie_header:
                gate_headers["Cookie"] = cookie_header
            elif access_token:
                gate_headers["Cookie"] = f"access_token={access_token}"
            if api_key_header:
                gate_headers["X-API-Key"] = api_key_header

            body = _json.dumps({"endpoint": request.path}).encode()

            try:
                req = urllib.request.Request(
                    f"{_AUTH_SERVICE_URL}/internal/generation/consume",
                    data=body,
                    headers=gate_headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    gate_body = _json.loads(resp.read())
                    # 200 → allowed, continue normally. Stash the user_id
                    # on flask.g so the after_request persistence hook can
                    # associate generated files with this user when
                    # registering rows in the auth-service POST /api/generations.
                    try:
                        g.gate_user_id = int(gate_body.get("user_id"))
                    except (TypeError, ValueError):
                        # Shouldn't happen — auth-service always returns an int —
                        # but don't fail the request if it ever does.
                        g.gate_user_id = None
                    return

            except urllib.error.HTTPError as e:
                err_body = {}
                try:
                    err_body = _json.loads(e.read())
                except Exception:
                    pass
                if e.code == 429:
                    remaining = err_body.get("remaining", 0)
                    resets_at = err_body.get("resets_at", "tomorrow")
                    _gate_log.info("gate: blocked user — daily cap reached")
                    return jsonify({
                        "error": err_body.get("detail", "Daily generation limit reached"),
                        "remaining": remaining,
                        "resets_at": resets_at,
                    }), 429
                elif e.code == 401:
                    return jsonify({"error": "Authentication required for generation"}), 401
                elif e.code == 503:
                    return jsonify({"error": "Generation is temporarily disabled"}), 503
                else:
                    # Unexpected HTTP error from auth-service — fail-CLOSED for
                    # generation bucket. Auth-service should not return unexpected
                    # codes; if it does something is wrong, don't let generation through.
                    _gate_log.error(
                        "gate: unexpected HTTP %d from auth-service — failing closed", e.code
                    )
                    return jsonify({"error": "Generation service unavailable, please try again"}), 503

            except Exception as exc:
                # Network error (timeout, DNS, connection refused) — fail-CLOSED.
                # Analysis routes are not gated and always pass; only generation
                # routes reach here, so closing is correct.
                _gate_log.error(
                    "gate: auth-service unreachable (%s) — failing closed", exc
                )
                return jsonify({"error": "Generation service unavailable, please try again"}), 503

        # ---------------------------------------------------------------------------
        # Queue-depth canary. With max_containers=1, if a second request
        # arrives while the first is running, it queues at Modal's front
        # door. Log a WARN so we notice in `modal app logs` before a user
        # complains.
        # ---------------------------------------------------------------------------
        _inflight = [0]
        _lock = threading.Lock()
        _log = logging.getLogger("stemphonic.queue")

        @stemphonic_server.app.before_request
        def _track_inflight():
            with _lock:
                _inflight[0] += 1
                depth = _inflight[0]
            if depth > 1:
                _log.warning(
                    "REQUEST QUEUED: %d concurrent in-flight requests "
                    "(max_containers=1 is serializing). Raise max_containers "
                    "or move task state off in-process.",
                    depth,
                )

        @stemphonic_server.app.teardown_request
        def _untrack_inflight(_exc):
            with _lock:
                _inflight[0] = max(0, _inflight[0] - 1)
            # VRAM diagnostic per request — captures the peak activation
            # footprint so we can decide whether a smaller GPU fits.
            try:
                import torch as _t
                from flask import request as _req
                if _t.cuda.is_available() and _req.path.startswith("/api/"):
                    _alloc = _t.cuda.memory_allocated() / 1e9
                    _peak  = _t.cuda.max_memory_allocated() / 1e9
                    logging.info(
                        "VRAM after %s: alloc=%.2f GB peak=%.2f GB",
                        _req.path, _alloc, _peak,
                    )
                    # Reset peak so next request's peak is independent
                    _t.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # ---------------------------------------------------------------------------
        # R2 persistence + auth-service generation registration
        # ---------------------------------------------------------------------------
        # Today every generation output WAV/MIDI dies with the Modal container
        # (served inline via send_file from /scratch/stemphonic_outputs). This
        # hook mirrors each successful output to Cloudflare R2 and registers
        # a row in the Fly auth-service so users can replay generations from
        # any device.
        #
        # Architecture:
        #   1. Gate accepts → stash user_id on flask.g.gate_user_id (above).
        #   2. POST create-task endpoint returns {task_id} → before the response
        #      leaves, we record user_id ↔ task_id in _task_owners[task_id].
        #   3. Polling endpoint (or synchronous endpoint) response includes
        #      file URLs under /scratch/stemphonic_outputs/<task_id>/<filename>.
        #      When we observe status==completed we iterate those files, upload
        #      to R2 at generations/<user_id>/<task_id>/<filename>, then call
        #      POST /api/generations on auth-service to register each row, and
        #      inject an "r2" block into the response body.
        #   4. Per-task cache in _task_r2_cache avoids re-uploading on every
        #      subsequent poll once a task is fully persisted.
        #
        # Every failure path logs + swallows: the user's request NEVER blocks
        # on R2. Files stay on /user_data for ~7 days (cleanup_old_outputs),
        # so a failed persist cycle just means the user gets the original
        # send_file URLs with no r2 block and the frontend falls back to the
        # Modal download route.
        import json as _gj, urllib.error as _gue, urllib.request as _gur, time as _gt
        from threading import Lock as _GLock

        # Default: POST /api/generations on the Fly auth-service
        # (doseedo-api.fly.dev), which owns the generations table. Keep the
        # GENERATION_REGISTER_URL override for forward compat in case the
        # endpoint moves. The quota gate may still live on a different host
        # via AUTH_SERVICE_URL — these two do not need to agree.
        _GEN_REGISTER_URL = (
            os.environ.get("GENERATION_REGISTER_URL", "").rstrip("/")
            or "https://doseedo-api.fly.dev/api/generations"
        )
        # Map path → (kind label, create-task? True, polling? False).
        # Separate-stems and repaint-meter create multiple tasks per request
        # so their kind label is applied per task_id that pops out of the
        # response body. transcribe-vocals returns text-only (no audio) so
        # it does not participate in R2 persistence.
        _KIND_BY_CREATE_PATH = {
            "/api/generate-stemphonic":       "stemphonic",
            "/api/separate-stemphonic":       "separate-stems",
            "/separate-stems":                "separate-stems",
            "/api/repaint-meter":             "repaint",
            "/api/regen-stem-for-chord":      "regen-stem",
            "/api/generate-score-from-video": "video-score",
        }
        # For polling endpoints we need to know which route family they
        # belong to so the after_request hook can find the right task dict
        # and output dir on the server module.
        #   path_prefix → (task_dict_attr, output_dir, kind_label)
        # The task dict lives on stemphonic_server as either `tasks` (generate
        # family) or `DEMUCS_TASKS` (separate-stems family). OUTPUT_DIR
        # constants on the server module point at the on-disk dirs.
        _POLL_FAMILIES = [
            # (path_prefix, dict_attr, output_dir_attr, kind)
            ("/api/generate-stemphonic/task/", "tasks",        "OUTPUT_DIR",        "stemphonic"),
            ("/separate-stems/status/",        "DEMUCS_TASKS", "DEMUCS_OUTPUT_DIR", "separate-stems"),
        ]

        _task_owners: dict[str, dict] = {}  # task_id → {user_id, kind, created_at}
        _task_r2_cache: dict[str, list] = {}  # task_id → [{key, url, generation_id, filename}, ...]
        _task_lock = _GLock()
        _r2_log = logging.getLogger("stemphonic.r2")

        def _gc_task_owners():
            """Drop owner entries older than 24h to keep the map bounded."""
            cutoff = _gt.time() - 86400
            with _task_lock:
                stale = [tid for tid, meta in _task_owners.items() if meta.get("created_at", 0) < cutoff]
                for tid in stale:
                    _task_owners.pop(tid, None)
                    _task_r2_cache.pop(tid, None)

        def _register_generation(user_id, task_id, kind, r2_key, content_type,
                                 filename, duration_sec=None, metadata=None):
            """POST to auth-service /api/generations. Returns dict or None."""
            if not _GEN_REGISTER_URL or not _INTERNAL_SECRET:
                return None
            body = _gj.dumps({
                "user_id": int(user_id),
                "task_id": task_id,
                "kind": kind,
                "r2_key": r2_key,
                "content_type": content_type,
                "filename": filename,
                "duration_sec": duration_sec,
                "metadata": metadata or {},
            }).encode()
            try:
                req = _gur.Request(
                    _GEN_REGISTER_URL,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Internal-Secret": _INTERNAL_SECRET,
                    },
                    method="POST",
                )
                with _gur.urlopen(req, timeout=5) as resp:
                    return _gj.loads(resp.read())
            except _gue.HTTPError as e:
                _r2_log.error(
                    "auth-service /api/generations HTTP %d for task=%s key=%s",
                    e.code, task_id, r2_key,
                )
                return None
            except Exception as e:
                _r2_log.exception(
                    "auth-service /api/generations unreachable for task=%s: %s",
                    task_id, e,
                )
                return None

        def _guess_content_type(filename: str) -> str:
            fn = filename.lower()
            if fn.endswith(".wav"):  return "audio/wav"
            if fn.endswith(".mp3"):  return "audio/mpeg"
            if fn.endswith(".flac"): return "audio/flac"
            if fn.endswith(".ogg"):  return "audio/ogg"
            if fn.endswith(".mid") or fn.endswith(".midi"):
                return "audio/midi"
            if fn.endswith(".pt"):   return "application/octet-stream"
            return "application/octet-stream"

        def _persist_task_outputs(user_id, task_id, kind, output_dir):
            """Walk the on-disk output dir for a completed task, upload each
            file to R2, register each with auth-service, and return a list of
            {key, url, generation_id, filename} dicts. Caches per-task so we
            only do this once even if the polling endpoint is hit repeatedly.
            """
            import os as _os, pathlib as _pl
            with _task_lock:
                if task_id in _task_r2_cache:
                    return _task_r2_cache[task_id]

            task_dir = _pl.Path(output_dir) / task_id
            if not task_dir.exists():
                return []

            persisted = []
            for fp in task_dir.rglob("*"):
                if not fp.is_file():
                    continue
                # Skip intermediate / per-task private artifacts. Anything
                # named input_*, stem_input*, midi_*, or under drum_substems/
                # is a pre-generation input or a sub-stem cache — the user-
                # facing output is the top-level WAV/MIDI written by the
                # route itself.
                if fp.name.startswith("input_") or fp.name.startswith("stem_input"):
                    continue
                if fp.name.startswith("midi_") and fp.suffix.lower() not in (".mid", ".midi"):
                    continue
                # Only audio / MIDI outputs are user-facing — skip .pt, .json
                # intermediates but keep .wav, .mid, .mp3 etc.
                if fp.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg",
                                             ".mid", ".midi"):
                    continue
                rel = fp.relative_to(task_dir)
                key = f"generations/{user_id}/{task_id}/{rel}"
                ct = _guess_content_type(fp.name)
                try:
                    uploaded_key = _r2_upload_file(str(fp), key, ct)
                except Exception as e:
                    _r2_log.exception(
                        "R2 upload crashed for task=%s file=%s: %s", task_id, fp, e
                    )
                    uploaded_key = None

                if uploaded_key is None:
                    # R2 upload failed — skip registration but continue. The
                    # existing send_file route still works.
                    continue

                reg = _register_generation(
                    user_id=user_id,
                    task_id=task_id,
                    kind=kind,
                    r2_key=uploaded_key,
                    content_type=ct,
                    filename=fp.name,
                )
                if reg is None:
                    # Registration failed — we already uploaded, but the row
                    # didn't land. Include the key anyway so frontend can
                    # presign later if it wants; skip the generation_id.
                    persisted.append({
                        "key": uploaded_key,
                        "url": _r2_generate_download_url(uploaded_key) or "",
                        "generation_id": None,
                        "filename": fp.name,
                    })
                    continue

                persisted.append({
                    "key": reg.get("r2_key", uploaded_key),
                    "url": reg.get("download_url") or _r2_generate_download_url(uploaded_key) or "",
                    "generation_id": reg.get("id"),
                    "filename": fp.name,
                })

            with _task_lock:
                _task_r2_cache[task_id] = persisted
            return persisted

        @stemphonic_server.app.after_request
        def _r2_persist_and_register(response):
            from flask import request, g

            # Opportunistic GC — cheap, no-op if there's nothing to drop.
            try:
                _gc_task_owners()
            except Exception:
                pass

            # Only touch JSON responses. send_file binary downloads pass
            # through untouched.
            ct = (response.content_type or "").lower()
            if not ct.startswith("application/json"):
                return response

            path = request.path

            # -----------------------------------------------------------------
            # (A) POST create-task endpoints — remember user_id for each task_id
            #     that appears in the response body.
            # -----------------------------------------------------------------
            if request.method == "POST" and path in _KIND_BY_CREATE_PATH:
                user_id = getattr(g, "gate_user_id", None)
                kind = _KIND_BY_CREATE_PATH[path]
                if user_id is not None:
                    try:
                        body = response.get_json(silent=True) or {}
                    except Exception:
                        body = {}
                    task_ids = []
                    if isinstance(body, dict):
                        if body.get("task_id"):
                            task_ids.append(body["task_id"])
                        if body.get("audio_task_id"):
                            # /api/generate-score-from-video spawns a stemphonic
                            # audio render — treat its downstream outputs as
                            # stemphonic kind (it goes through run_generation).
                            task_ids.append(body["audio_task_id"])
                        # repaint-meter returns {"results": [{task_id, ...}, ...]}
                        for entry in (body.get("results") or []):
                            if isinstance(entry, dict) and entry.get("task_id"):
                                task_ids.append(entry["task_id"])
                    for tid in task_ids:
                        # video-score spawns into the `tasks` dict as kind
                        # "stemphonic" (shared run_generation), so the poll
                        # family resolver maps it correctly.
                        resolved_kind = "stemphonic" if path == "/api/generate-score-from-video" else kind
                        with _task_lock:
                            _task_owners[tid] = {
                                "user_id": user_id,
                                "kind": resolved_kind,
                                "created_at": _gt.time(),
                            }

                    # /api/generate-score-from-video also produces a MIDI file
                    # synchronously at SCORE_OUTPUT_DIR/<score_id>/score.mid.
                    # Persist it inline (not via the polling-task machinery
                    # since score_id is not a task_id in the `tasks` dict).
                    if path == "/api/generate-score-from-video" and isinstance(body, dict):
                        score_id = body.get("score_id")
                        if score_id:
                            try:
                                score_dir_root = getattr(
                                    self._server_module, "SCORE_OUTPUT_DIR",
                                    "/scratch/stemphonic_outputs/scores",
                                )
                                # Reuse _persist_task_outputs by passing the
                                # score_id as task_id — R2 key becomes
                                # generations/<user_id>/<score_id>/score.mid
                                # which is still unique per user.
                                persisted = _persist_task_outputs(
                                    user_id, score_id, "video-score", score_dir_root,
                                )
                                if persisted:
                                    body["r2"] = persisted if len(persisted) > 1 else persisted[0]
                                    response.set_data(_gj.dumps(body))
                            except Exception as e:
                                _r2_log.exception(
                                    "inline persist (video-score) crashed score_id=%s: %s",
                                    score_id, e,
                                )

                # Synchronous /api/separate-stemphonic returns the audio_url
                # and task_id inline — persist immediately.
                if path == "/api/separate-stemphonic" and user_id is not None:
                    try:
                        body = response.get_json(silent=True) or {}
                    except Exception:
                        body = {}
                    task_id = body.get("task_id")
                    if task_id and body.get("status") == "completed":
                        try:
                            output_dir = getattr(
                                self._server_module, "OUTPUT_DIR",
                                "/scratch/stemphonic_outputs",
                            )
                            persisted = _persist_task_outputs(
                                user_id, task_id, "separate-stems", output_dir,
                            )
                            if persisted:
                                body["r2"] = persisted if len(persisted) > 1 else persisted[0]
                                response.set_data(_gj.dumps(body))
                        except Exception as e:
                            _r2_log.exception(
                                "inline persist (separate-stemphonic) crashed task=%s: %s",
                                task_id, e,
                            )

            # -----------------------------------------------------------------
            # (B) Polling endpoints — on first sighting of status==completed,
            #     upload outputs + register + inject "r2" block into response.
            # -----------------------------------------------------------------
            if request.method == "GET":
                for prefix, dict_attr, output_dir_attr, default_kind in _POLL_FAMILIES:
                    if not path.startswith(prefix):
                        continue
                    task_id = path[len(prefix):].split("/", 1)[0]
                    if not task_id:
                        return response

                    try:
                        body = response.get_json(silent=True) or {}
                    except Exception:
                        body = {}

                    status = (body.get("status") or "").lower()
                    if status != "completed":
                        return response

                    with _task_lock:
                        owner = _task_owners.get(task_id)
                    if not owner:
                        # Pre-existing task from before the gate started
                        # stashing, or gate was configured fail-open. Skip
                        # persistence but keep original response intact.
                        return response

                    try:
                        output_dir = getattr(
                            self._server_module, output_dir_attr,
                            "/scratch/stemphonic_outputs",
                        )
                        persisted = _persist_task_outputs(
                            owner["user_id"], task_id,
                            owner.get("kind", default_kind),
                            output_dir,
                        )
                    except Exception as e:
                        _r2_log.exception(
                            "persist on poll crashed task=%s: %s", task_id, e
                        )
                        persisted = []

                    if persisted:
                        # Use an array when multiple files, object when one.
                        body["r2"] = persisted if len(persisted) > 1 else persisted[0]
                        response.set_data(_gj.dumps(body))
                    return response

            return response

        self.flask_app = stemphonic_server.app

    @modal.wsgi_app()
    def wsgi(self):
        """Expose the Flask app over HTTPS as a Modal web endpoint.

        All ~30 routes in stemphonic_server.py (/api/generate-stemphonic,
        /api/repaint-meter, /api/encode-audio-latent, /api/upload-latent,
        /api/extract-midi, /api/transcribe-vocals, etc.) work unchanged.
        """
        return self.flask_app


# ---------------------------------------------------------------------------
# Scheduled cleanup: delete old /user_data/stemphonic_outputs dirs
# ---------------------------------------------------------------------------
# Runs nightly. Deletes stemphonic_outputs/{task_id} dirs older than the
# cutoff. Leaves /user_data/latents untouched — those are user-owned and
# referenced by session saves, must persist indefinitely.
@app.function(
    image=image,
    volumes={"/user_data": user_data_vol},
    schedule=modal.Period(days=1),
    timeout=600,
    cpu=1,
    memory=1024,
)
def cleanup_old_outputs(max_age_days: float = 7):
    import time, shutil, pathlib

    user_data_vol.reload()  # see anything written since the last run

    root = pathlib.Path("/user_data/stemphonic_outputs")
    if not root.exists():
        print("no /user_data/stemphonic_outputs dir — nothing to clean", flush=True)
        return

    cutoff = time.time() - max_age_days * 86400
    # Don't walk into these — they're shared caches, not per-task outputs
    protect = {"separations", "scores", "timbre_cache"}

    n_removed = 0
    bytes_removed = 0
    for entry in root.iterdir():
        if entry.name in protect:
            continue
        try:
            mtime = entry.stat().st_mtime
            if mtime >= cutoff:
                continue
            if entry.is_dir():
                sz = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                shutil.rmtree(entry)
            else:
                sz = entry.stat().st_size
                entry.unlink()
            n_removed += 1
            bytes_removed += sz
        except Exception as e:
            print(f"skip {entry.name}: {e}", flush=True)

    user_data_vol.commit()
    print(
        f"cleanup: removed {n_removed} entries, "
        f"{bytes_removed/1e9:.2f} GB (older than {max_age_days} days)",
        flush=True,
    )


@app.local_entrypoint()
def main():
    """
    Deployment is via `modal deploy modal_stemphonic.py`.

    Useful one-liners:
      modal volume put stemphonic-models <local> <remote>
      modal volume ls  stemphonic-models
      modal run        modal_stemphonic.py::cleanup_old_outputs
      modal deploy     modal_stemphonic.py
      modal app logs   doseedo-stemphonic
      modal app stop   doseedo-stemphonic
    """
    print(
        "See the docstring at the top of this file for the full deploy flow.\n"
        "TL;DR: modal volume put the weights, then modal deploy."
    )
