# Third-party model attribution

This service incorporates several pretrained models. Each is governed by
its own license — most importantly, **Demucs is non-commercial only**.
Read this file before changing how user-facing features expose model
outputs to paying users.

## Demucs 

- **Used in**: stem separation pipelines (`latent_demucs/`,
  `latent_demucs_student/`, model serving in `modal_stemphonic.py`).
- **License**: Demucs v4 weights are released under the
  [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
  license. The Demucs source code is MIT.
- **Implication for production**: The CC BY-NC 4.0 license **prohibits
  commercial use** of the model weights, including offering stem
  separation as part of a paid product. Before charging users for any
  feature that calls Demucs:
    1. Negotiate a commercial license with Meta, OR
    2. Replace Demucs weights with a permissively-licensed model
       (e.g. our distilled `distill_demucs_fp16.onnx` if it was trained
       from scratch on permissive data — verify provenance).

## Whisper (OpenAI)

- **Used in**: vocal transcription (`latent_whisper_student/`,
  `/api/transcribe-vocals`).
- **License**: [MIT](https://github.com/openai/whisper/blob/main/LICENSE).
  Commercial use permitted with attribution.
- **Required attribution**: "Speech-to-text powered by OpenAI Whisper
  (MIT)."

## PANNs (Pretrained Audio Neural Networks)

- **Used in**: audio classification / instrument tagging
  (`latent_panns_student/`).
- **License**: [MIT](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/LICENSE).
  Commercial use permitted with attribution.
- **Required attribution**: "Audio tagging based on PANNs (Kong et al.,
  MIT)."

## Qwen3-14B-AWQ (Alibaba Cloud)

- **Used in**: chatbot via Modal vLLM (`/api/chat` proxy → Modal app
  `doseedo-chatbot`).
- **License**: [Apache 2.0](https://github.com/QwenLM/Qwen). Commercial
  use permitted.
- **Required attribution**: "Chat powered by Qwen3-14B (Apache 2.0,
  Alibaba Cloud)."

## Moondream (vision)

- **Used in**: vision endpoint (`/api/vision` proxy → Modal app
  `doseedo-chatbot-vision`).
- **License**: [Apache 2.0](https://github.com/vikhyat/moondream).
  Commercial use permitted.
- **Required attribution**: "Vision powered by Moondream (Apache 2.0)."

## ACE-Step (deploy_resources/)

- **Used in**: `/api/generate-ace-step` (currently unrouted in production).
- **License**: verify before re-enabling — check the upstream repo and
  weights distribution terms. This row stays as a TODO until that's
  documented.

---
