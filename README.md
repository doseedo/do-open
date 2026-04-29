# do-open

Open-source release of Doseedo — an AI-assisted music production web app
(stem separation, polyphonic pitch editing, vocal transcription, chat,
video scoring). This is a snapshot of the production codebase with
inference backends, the Next.js frontend, and the tooling used to train
and serve the latent-space models.

## What's in here

| Directory | What it is |
| --- | --- |
| `frontend-next/` | Next.js 14 + React 18 web app. Clerk auth, Vercel-deployed. Hosts the Studio (DAW), piano-roll, stem editor, chat, and dashboard. |
| `inference/` | Modal apps that serve the GPU backends — `modal_stemphonic.py` (A100, stem separation + polypitch + soundfont + drum-sep), `modal_chatbot.py` (vLLM Qwen3-14B-AWQ), `modal_video_scoring.py`, `modal_watermark.py`. Plus the Flask app (`stemphonic_server.py`) that runs inside the Modal container. |
| `tools/` | Local Python packages for training and inference of the `latent_*` model family — `latent_demucs` (stem sep), `latent_pitch` (polyphonic pitch), `latent_whisper_student` (vocal transcription), `latent_panns_student` (audio tagging), `latent_drumsep`, `latent_editor`, `latent_soundfont`, `latent_visual`, plus `video_scoring`. |
| `docs/` | Runbooks (launch, Sentry, telemetry), troubleshooting, support canned responses. |
| `Makefile` | Dev and deploy shortcuts (`make modal-dev`, `make fly-dev`, etc.). Run `make help` for the full list. |
| `ATTRIBUTION.md` | Third-party model licenses. **Read before shipping anything commercial** — Demucs weights are CC BY-NC 4.0. |

## Architecture at a glance

```
Browser
  │
  ▼
frontend-next (Vercel)            ── Clerk auth, Stripe, Sentry
  │
  ├─► /api/*  (Next.js route handlers)
  │     │
  │     ├─► Modal: doseedo-stemphonic   (A100, GPU inference)
  │     ├─► Modal: doseedo-chatbot      (vLLM Qwen3-14B-AWQ)
  │     ├─► Modal: doseedo-chatbot-vision (Moondream)
  │     └─► Modal: doseedo-video-scoring
  │
  └─► Storage: Cloudflare R2 (audio, stems)
      Database: Neon Postgres
      Cache:    Upstash Redis (db=0 only)
```

## Models

All third-party model licenses are documented in `ATTRIBUTION.md`. The
models that matter:

- **Demucs v4** — stem separation. CC BY-NC 4.0. Non-commercial only.
- **Whisper** (OpenAI) — vocal transcription. MIT.
- **PANNs** — audio tagging. MIT.
- **Qwen3-14B-AWQ** — chat. Apache 2.0.
- **Moondream** — vision. Apache 2.0.

The `latent_*` student models in `tools/` are distillations trained
in-house and released under this repo's license.

## Running locally

Frontend:

```bash
cd frontend-next
cp .env.local.example .env.local   # fill in Clerk, R2, Modal, Neon URLs
npm install
npm run dev                         # localhost:3000
```

Inference (Modal):

```bash
cd inference
pip install -r requirements.txt
modal serve modal_stemphonic.py     # hot-reload dev URL
modal deploy modal_stemphonic.py    # production
```

See `Makefile` for the full set of deploy targets and `docs/` for
operational runbooks.

## License

Code in this repo is released as-is for reference and research.
Third-party model weights remain under their original licenses — see
`ATTRIBUTION.md` before redistributing or building a commercial product
on top of this.
