"""
Stemphonic inference that matches training_step exactly.

Training step flow:
  1. encode_conditioning(): encoder(text+lyrics+timbre) → encoder_hs
  2. build_context_latents(): src + masks based on mode (cover uses FSQ)
  3. Decoder forward with MIDI hooks
  4. Flow matching: xt = noise → target via Euler

For inference (cover mode with FSQ preset):
  - is_cover=True, cover_fsq_raw=FSQ preset [T_5Hz, 2048]
  - is_conditional=False (no sub_mix)
  - build_context_latents → [T, 128] with src=detokenized FSQ
  - Direct decoder call in Euler denoising loop with CFG
"""

import torch
import torch.nn.functional as F


def generate_stemphonic(
    handler, module,
    prompt: str,
    lyrics: str = "[Instrumental]",
    midi_tensor=None,
    timbre_ref=None,          # [1, 30, 64]
    fsq_raw=None,             # [T_5Hz, 2048] — cover FSQ
    sub_mix_latents=None,     # [1, T, 64] — sub-mix conditioning
    duration=16.0,
    steps=50,
    cfg=7.0,
    seed=42,
    cover_src_latents=None,    # [T, 64] raw VAE latents of the cover audio
                               # (renoise base — must NOT be FSQ-detokenized)
    cover_noise_strength=0.0,  # 0 = pure noise init, 1 = start directly from src
    audio_cover_strength=1.0,  # how many steps to keep cover-mode encoder context
):
    """Replicates training_step's forward pass for inference (Euler denoising)."""
    device = "cuda"
    dtype = torch.bfloat16
    T = int(duration * 25)
    D = 64
    B = 1

    # ------------------------------------------------------------------
    # 1. Build text + lyric encodings (as training does via cached embeddings)
    # ------------------------------------------------------------------
    from stemphonic_trainer.preprocess_v4 import format_sft_prompt, format_lyrics
    # Pass real BPM/time_sig (not the empty defaults that turn into "N/A").
    # Training cache was keyed on real values; "N/A" is OOD for the model.
    text_prompt = format_sft_prompt(prompt, bpm="120", time_sig="4/4",
                                    duration_sec=int(duration))
    lyric_prompt = format_lyrics(lyrics or "[Instrumental]")

    with torch.inference_mode():
        text_tokens = handler.text_tokenizer(
            text_prompt, return_tensors="pt", padding="max_length",
            max_length=256, truncation=True,
        ).to(device)
        text_hs = handler.text_encoder(
            input_ids=text_tokens["input_ids"], lyric_attention_mask=None,
        ).last_hidden_state.to(dtype)
        text_mask = text_tokens["attention_mask"].to(dtype)

        lyric_tokens = handler.text_tokenizer(
            lyric_prompt, return_tensors="pt", padding="max_length",
            max_length=512, truncation=True,
        ).to(device)
        lyric_hs = handler.text_encoder.embed_tokens(lyric_tokens["input_ids"]).to(dtype)
        lyric_mask = lyric_tokens["attention_mask"].to(dtype)

        # ------------------------------------------------------------------
        # 2. Timbre ref (matches training_module's refer_packed format)
        # ------------------------------------------------------------------
        if timbre_ref is not None:
            refer = timbre_ref.to(device, dtype=dtype)
            if refer.dim() == 2:
                refer = refer.unsqueeze(0)
            # Ensure [B, 30, 64]
            if refer.shape[1] > 30:
                refer = refer[:, :30]
            elif refer.shape[1] < 30:
                refer = F.pad(refer, (0, 0, 0, 30 - refer.shape[1]))
        else:
            refer = torch.zeros(B, 30, D, device=device, dtype=dtype)
        refer_order = torch.zeros(B, device=device, dtype=torch.long)

        # ------------------------------------------------------------------
        # 3. Run encoder (same as training_module.encode_conditioning, minus resonance)
        # ------------------------------------------------------------------
        encoder_hs, encoder_mask = handler.model.encoder(
            text_hidden_states=text_hs,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hs,
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=refer,
            refer_audio_order_mask=refer_order,
        )

        # Null conditioning for CFG (from training's apply_independent_cfg_dropout path,
        # which uses null_condition_emb when text is dropped)
        null_enc = handler.model.null_condition_emb
        if null_enc is not None:
            null_enc_hs = null_enc.expand(B, -1, -1).to(device, dtype=dtype)
            null_enc_mask = torch.ones(B, null_enc_hs.shape[1], device=device, dtype=dtype)
        else:
            null_enc_hs = torch.zeros_like(encoder_hs)
            null_enc_mask = torch.ones_like(encoder_mask)

        # ------------------------------------------------------------------
        # 4. Build context_latents [B, T, 128] — matches build_context_latents()
        # ------------------------------------------------------------------
        # Start with from-scratch mode: src = zeros, chunk_masks = ones
        src = torch.zeros(B, T, D, device=device, dtype=dtype)
        chunk_masks = torch.ones(B, T, D, device=device, dtype=dtype)

        # Sub-mix conditional (if provided)
        if sub_mix_latents is not None:
            sm = sub_mix_latents.to(device, dtype=dtype)
            if sm.dim() == 2:
                sm = sm.unsqueeze(0)
            if sm.shape[1] >= T:
                src[:, :T] = sm[:, :T]
            else:
                src[:, :sm.shape[1]] = sm

        # Cover mode: detokenize FSQ (or use raw latent) → src
        if fsq_raw is not None:
            fsq_in = fsq_raw.to(device, dtype=dtype)
            if fsq_in.dim() == 2:
                fsq_in = fsq_in.unsqueeze(0)
            # Handle both formats:
            # - [B, T_5Hz, 2048] → detokenize to [B, T_25Hz, 64]
            # - [B, T, 64] → already raw latent, use directly
            if fsq_in.shape[-1] == 2048:
                lm_hints = handler.model.detokenize(fsq_in)  # [B, T_25Hz, 64]
            elif fsq_in.shape[-1] == 64:
                lm_hints = fsq_in  # already [B, T, 64]
            else:
                raise ValueError(f"Unknown FSQ shape: {fsq_in.shape}")
            T_hint = lm_hints.shape[1]
            if T_hint >= T:
                src[:, :T] = lm_hints[:, :T].to(dtype=dtype)
            else:
                # Tile to fill T (instead of zero-pad which weakens conditioning)
                reps = (T // T_hint) + 1
                tiled = lm_hints.repeat(1, reps, 1)[:, :T]
                src[:, :T] = tiled.to(dtype=dtype)

        context_latents = torch.cat([src, chunk_masks], dim=-1)  # [B, T, 128]

        # ------------------------------------------------------------------
        # 5. MIDI hooks (same as training_step)
        # ------------------------------------------------------------------
        if midi_tensor is not None:
            midi_feat = module.build_midi_features(
                midi_tensor.to(device, dtype=dtype), torch.tensor([True]),
                B=B, T=T, device=device, dtype=dtype,
            )
            patch_size = getattr(handler.model.decoder, 'patch_size', 2)
            # Decoder rounds T UP to patch_size multiples (T=287, patch=2
            # → 144 tokens). Two alignment steps:
            #   1. Pad/trim mf along the frame axis to T_dec*patch_size so
            #      the reshape divides evenly. Without the pad-up branch,
            #      odd T (e.g. 225 for 9s) failed with
            #        shape '[1, 113, 2, -1]' is invalid for input of size …
            #      because mf[:,:226] only had 225 frames.
            #   2. Reshape + mean over patch dim → [B, T_dec, D_midi].
            T_dec = -(-T // patch_size)
            needed = T_dec * patch_size
            mf = midi_feat
            if mf.shape[1] > needed:
                mf = mf[:, :needed]
            elif mf.shape[1] < needed:
                mf = F.pad(mf, (0, 0, 0, needed - mf.shape[1]))
            mf = mf.reshape(B, T_dec, patch_size, -1).mean(dim=2)
            module._midi_features_for_hook = mf
        else:
            module._midi_features_for_hook = None

        # ------------------------------------------------------------------
        # 6. Euler denoising with CFG (flow matching, same as training_step)
        # ------------------------------------------------------------------
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dt = 1.0 / steps
        attn_mask = torch.ones(B, T, device=device, dtype=dtype)

        # Cover-noise: when > 0 with a real renoise base, dispatch to the
        # model's built-in generate_audio(). It needs RAW VAE latents for
        # src_latents (the renoise base), and SEPARATELY accepts the FSQ
        # lm hints via precomputed_lm_hints_25Hz. Conflating the two
        # produces silent diffusion output (the FSQ-detokenized space is
        # NOT the diffusion latent space).
        cns = float(max(0.0, min(1.0, cover_noise_strength)))
        have_renoise_base = cover_src_latents is not None
        if have_renoise_base and cns > 0.0:
            # Build the lm_hints tensor [B, T_lm, 64] from fsq_raw if given
            lm_hints_for_model = None
            if fsq_raw is not None:
                lh = fsq_raw.to(device=device, dtype=dtype)
                if lh.dim() == 2:
                    lh = lh.unsqueeze(0)
                # _generate_via_model_api expects raw lm_hints [B, T, 64]
                # not the [B, T, 2048] FSQ tokens — detokenize if needed
                if lh.shape[-1] == 2048:
                    lh = handler.model.detokenize(lh)
                lm_hints_for_model = lh
            return _generate_via_model_api(
                handler=handler, module=module,
                text_hs=text_hs, text_mask=text_mask,
                lyric_hs=lyric_hs, lyric_mask=lyric_mask,
                refer=refer, refer_order=refer_order,
                cover_src_latents=cover_src_latents.to(device=device, dtype=dtype),
                lm_hints=lm_hints_for_model,
                T=T, D=D,
                duration=duration, steps=steps, cfg=cfg, seed=seed,
                cover_noise_strength=cns,
                audio_cover_strength=float(audio_cover_strength),
            )

        xt = torch.randn(B, T, D, device=device, dtype=dtype)
        for i in range(0, steps):
            t_val = 1.0 - i * dt
            t = torch.tensor([t_val] * B, device=device, dtype=dtype)

            # Conditional prediction
            pred_cond = handler.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attn_mask,
                encoder_hidden_states=encoder_hs,
                encoder_attention_mask=encoder_mask,
                context_latents=context_latents,
            )[0]

            # Unconditional prediction
            pred_uncond = handler.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attn_mask,
                encoder_hidden_states=null_enc_hs,
                encoder_attention_mask=null_enc_mask,
                context_latents=context_latents,
            )[0]

            # CFG
            pred = pred_uncond + cfg * (pred_cond - pred_uncond)
            xt = xt - dt * pred

        module._midi_features_for_hook = None

    # ------------------------------------------------------------------
    # 7. Decode latent → audio
    # ------------------------------------------------------------------
    latent = xt.squeeze(0).permute(1, 0)  # [D, T]
    with torch.no_grad():
        audio = handler.vae.decode(latent.unsqueeze(0)).sample  # [1, 2, S]
    return audio.squeeze(0).cpu().float().numpy()  # [2, S]


def _generate_via_model_api(
    handler, module,
    text_hs, text_mask, lyric_hs, lyric_mask,
    refer, refer_order,
    cover_src_latents,   # [T, 64] raw VAE latents (renoise base)
    lm_hints,            # [B, T_lm, 64] FSQ-roundtripped (semantic context)
    T, D,
    duration, steps, cfg, seed,
    cover_noise_strength=0.0,
    audio_cover_strength=1.0,
):
    # Defensive clamp: model.generate_audio's audio_cover_strength<1.0
    # branch tries to build a non-cover encoder context using
    # non_cover_text_hidden_states, which we don't pass. Clamping to
    # 1.0 here keeps the cover encoder running for all steps and avoids
    # the prepare_condition None-input crash.
    if audio_cover_strength < 1.0:
        audio_cover_strength = 1.0
    """Dispatch to handler.model.generate_audio() for cover-mode inference.

    src_latents (the renoise base) MUST be raw VAE latents in the diffusion
    space. lm_hints (semantic context) goes through precomputed_lm_hints_25Hz
    separately. The MIDI hook fires inside the decoder layers regardless."""
    device = "cuda"
    dtype = torch.bfloat16
    B = 1

    # Build [B, T, 64] renoise base, padding/cropping to T frames
    src = cover_src_latents
    if src.dim() == 2:
        src = src.unsqueeze(0)
    if src.shape[1] >= T:
        src = src[:, :T]
    else:
        pad = T - src.shape[1]
        src = F.pad(src, (0, 0, 0, pad))
    src_latents = src.to(device, dtype=dtype).contiguous()

    chunk_masks = torch.ones(B, T, D, device=device, dtype=dtype)
    is_covers = torch.ones(B, device=device, dtype=torch.long)

    # Pad/crop lm_hints to T frames as well
    precomputed_lm_hints_25Hz = None
    if lm_hints is not None:
        lh = lm_hints
        if lh.dim() == 2:
            lh = lh.unsqueeze(0)
        if lh.shape[1] >= T:
            lh = lh[:, :T]
        else:
            lh = F.pad(lh, (0, 0, 0, T - lh.shape[1]))
        precomputed_lm_hints_25Hz = lh.to(device, dtype=dtype).contiguous()

    with torch.inference_mode():
        outputs = handler.model.generate_audio(
            text_hidden_states=text_hs,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hs,
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=refer,
            refer_audio_order_mask=refer_order,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=handler.silence_latent,
            seed=seed,
            infer_method="ode",
            infer_steps=steps,
            diffusion_guidance_scale=cfg,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            cover_noise_strength=float(cover_noise_strength),
            audio_cover_strength=float(audio_cover_strength),
            use_progress_bar=False,
        )
    pred_latents = outputs["target_latents"]  # [B, T, D]
    latent = pred_latents.squeeze(0).permute(1, 0)  # [D, T]
    with torch.no_grad():
        audio = handler.vae.decode(latent.unsqueeze(0)).sample
    if module is not None:
        module._midi_features_for_hook = None
    return audio.squeeze(0).cpu().float().numpy()
