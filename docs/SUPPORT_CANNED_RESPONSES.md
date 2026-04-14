# Support Canned Responses

Short, accurate replies for recurring questions. Keep each under
four sentences, warm tone, link to the longer source of truth.

## "Do you train on my audio?"

> Short answer: no. Stem separation and waveform encoding happen in
> your browser via WebGPU, so your audio doesn't leave your device for
> those steps — we only upload a compact latent representation. Nothing
> you upload or generate is used to train, fine-tune, or evaluate our
> models. Full details in our [privacy
> policy](https://doseedo.com/privacy) under "How We Handle Your Audio".

### Variations

**Twitter/public (≤ 280):**
> Stem separation + encoding runs in your browser (WebGPU). We only
> receive a compact latent, not your audio. We don't train on uploads
> or generations. Full architecture details at doseedo.com/privacy 🧬

**Discord/email (medium length):**
> Nothing you upload or generate is used for training. Our pipeline is
> built so that the heavy audio processing — stem separation, VAE
> encoding — happens locally in your browser using ONNX + WebGPU. What
> we receive is a latent representation (~64 floats per 40ms), which
> is what our playback and editing tools read from. The only narrow
> exception is instrument classification and MIDI extraction, which
> still run server-side on ephemeral GPU containers that don't persist
> anything past the request. Full breakdown at doseedo.com/privacy.

**B2B / legal review (long form):**
> Doseedo's Studio product performs stem separation and VAE encoding
> client-side using ONNX Runtime Web on the WebGPU backend. Uploaded
> audio is decoded in-browser, resampled to 48 kHz, and passed through
> the <code>distill_demucs</code> and <code>oobleck_encoder</code> models
> directly in the tab. What is transmitted to our backend is a dense
> latent tensor (64-dimensional, ~25 Hz), not the original waveform.
> Downstream generation tools (stemphonic, meter repaint, stem regen)
> operate on that latent on our Modal GPU backend. Two analysis
> endpoints — instrument classification and MIDI extraction — do
> require raw audio today; their containers are ephemeral and clear
> any temp files when they scale down (typically within 15 min of
> request completion). Generated output is stored in a per-user
> folder in GCS and is deletable from the Studio UI. No uploaded
> audio, generated audio, or latent is used for training, fine-tuning,
> or evaluating our models; our training sets are licensed/public-domain
> and are frozen per release. Happy to walk through the architecture
> or produce a DPA if that's useful.

## "Where is my generation stored?"

> Generated audio is saved in a per-account folder inside our Google
> Cloud Storage bucket. You can list, download, or delete any
> generation from the Studio UI. Deletion propagates to backing
> storage within 7 business days.

## "Can I export my latent data / take my stems elsewhere?"

> Yes. Every latent we store has a stable ID and an export endpoint:
> <code>/api/latent/&lt;id&gt;</code> returns the raw DOAE binary. Your
> generated audio is downloadable directly from the Studio file
> browser. We're also happy to provide a full export tarball if you
> email support@doseedo.com from your registered address.

## "What happens to my account if I cancel?"

> Account data and generated audio are retained for 30 days after
> cancellation in case you reactivate, then permanently deleted on day
> 31. You can request immediate deletion at any time via
> <a href="mailto:support@doseedo.com">support@doseedo.com</a> and we
> honor the request within 7 business days.

---

**Updating these:** if our architecture changes (e.g., we move
analysis endpoints client-side or start retaining generations longer),
update this file in the same PR as the code change so the support
answer can't drift from what actually ships.
