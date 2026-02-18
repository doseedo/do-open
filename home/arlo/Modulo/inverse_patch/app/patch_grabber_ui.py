#!/usr/bin/env python3
"""Patch Grabber — Gradio UI for inverse synthesis.

Upload audio, get synth patches back. Handles real-world recordings.

Usage:
    python app/patch_grabber_ui.py [--port 7861] [--share]
"""

import sys
import os
import time
import tempfile
import numpy as np
import gradio as gr

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')

from audio_input import load_audio, prepare_single, load_and_segment, normalize, SAMPLE_RATE
from patch_schema import optimizer_result_to_patch, save_patch, patch_summary, patch_to_render_params
from fast_dsp import (
    full_render, fm_render, ALL_WF_FNS, spectral_similarity,
    SAMPLE_RATE as SR, N_SAMPLES as NS,
)


def render_patch_audio(patch):
    """Render audio from patch dict."""
    params, wf_name, filter_type, pitch = patch_to_render_params(patch)
    if wf_name == 'fm':
        return fm_render(params, pitch)
    wf_fn = ALL_WF_FNS.get(wf_name, ALL_WF_FNS['saw'])
    waveform = wf_fn() if wf_name == 'noise' else wf_fn(pitch)
    return full_render(params, waveform, filter_type=filter_type)


def format_patch_display(patch):
    """Format patch as readable text for display."""
    lines = []
    lines.append(f"Synth Type: {patch['synth_type'].upper()}")
    lines.append(f"Pitch: {patch['pitch']:.1f} Hz")

    if patch['synth_type'] == 'fm':
        fm = patch['fm']
        lines.append(f"\nFM Parameters:")
        lines.append(f"  Mod Ratio: {fm['mod_ratio']:.3f}")
        lines.append(f"  Index Peak: {fm['index_peak']:.2f}")
        env = fm['envelope']
        lines.append(f"  FM Envelope: A={env['attack']:.3f} D={env['decay']:.3f} "
                      f"S={env['sustain']:.3f} R={env['release']:.3f}")
    else:
        lines.append(f"Waveform: {patch.get('waveform', '?')}")
        f = patch.get('filter', {})
        lines.append(f"\nFilter ({f.get('type', 'lowpass')}):")
        lines.append(f"  Cutoff: {f.get('base_hz', 0):.0f} - {f.get('peak_hz', 0):.0f} Hz")
        lines.append(f"  Resonance: {f.get('resonance', 0):.3f}")
        fe = f.get('envelope', {})
        lines.append(f"  Envelope: A={fe.get('attack', 0):.3f} D={fe.get('decay', 0):.3f} "
                      f"S={fe.get('sustain', 0):.3f} R={fe.get('release', 0):.3f} "
                      f"NoteOff={fe.get('noteoff', 0):.3f}")

    ae = patch.get('amp', {}).get('envelope', {})
    lines.append(f"\nAmp Envelope:")
    lines.append(f"  A={ae.get('attack', 0):.3f} D={ae.get('decay', 0):.3f} "
                  f"S={ae.get('sustain', 0):.3f} R={ae.get('release', 0):.3f} "
                  f"NoteOff={ae.get('noteoff', 0):.3f}")

    lfo = patch.get('lfo', {})
    if lfo.get('rate', 0) > 0:
        lines.append(f"\nLFO: {lfo['rate']:.2f} Hz, depth={lfo['depth']:.0f} Hz")

    effects = patch.get('effects', [])
    if effects:
        lines.append(f"\nEffects:")
        for fx in effects:
            fx_str = ', '.join(f"{k}={v}" for k, v in fx.items() if k != 'type')
            lines.append(f"  {fx['type']}: {fx_str}")

    q = patch.get('quality', {})
    lines.append(f"\nQuality:")
    lines.append(f"  Spectral Similarity: {q.get('spectral_similarity', 0):.4f}")
    lines.append(f"  Time Correlation: {q.get('time_correlation', 0):.4f}")
    lines.append(f"  Pipeline: {q.get('pipeline', '?')}")
    if 'optimization_time_s' in q:
        lines.append(f"  Optimization Time: {q['optimization_time_s']:.1f}s")

    return '\n'.join(lines)


def process_audio(audio_input, pitch_override):
    """Main processing function called by Gradio.

    Args:
        audio_input: tuple (sr, np.array) from Gradio audio component
        pitch_override: float or 0 for auto-detect

    Returns:
        (synth_audio, patch_text, patch_json_path, status)
    """
    if audio_input is None:
        return None, "No audio uploaded", None, "Upload an audio file to start"

    sr_in, audio_data = audio_input

    # Convert to float32 mono
    if audio_data.dtype != np.float32:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_data = audio_data.astype(np.float32)

    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)

    # Resample if needed
    if sr_in != SAMPLE_RATE:
        import torchaudio
        import torch
        resampler = torchaudio.transforms.Resample(sr_in, SAMPLE_RATE)
        audio_data = resampler(torch.from_numpy(audio_data).unsqueeze(0)).squeeze(0).numpy()

    audio_data = normalize(audio_data)

    # Trim and pad to 2s
    from audio_input import trim_silence
    audio_data, _, _ = trim_silence(audio_data, SAMPLE_RATE)
    target_len = int(2.0 * SAMPLE_RATE)
    if len(audio_data) < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(audio_data)] = audio_data
        audio_data = padded
    elif len(audio_data) > target_len:
        audio_data = audio_data[:target_len].copy()
        fade = min(int(0.05 * SAMPLE_RATE), target_len // 4)
        if fade > 0:
            audio_data[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    audio_data = normalize(audio_data)

    # Pitch
    pitch = None
    if pitch_override and pitch_override > 0:
        pitch = float(pitch_override)
    else:
        from audio_input import detect_pitch_yin
        pitch, conf = detect_pitch_yin(audio_data, SAMPLE_RATE)

    # Run optimizer
    t_start = time.time()
    try:
        from test_audio_domain import optimize_patch_full
        result = optimize_patch_full(audio_data, pitch=pitch, verbose=True)
    except Exception as e:
        return None, f"Error during optimization: {e}", None, f"Error: {e}"

    elapsed = time.time() - t_start
    patch = optimizer_result_to_patch(result, pitch=pitch)

    # Render synth audio
    synth_audio = render_patch_audio(patch)

    # Save patch JSON to temp file
    tmp_path = tempfile.mktemp(suffix='.json')
    save_patch(patch, tmp_path)

    # Format display
    patch_text = format_patch_display(patch)
    status = (f"Extracted {patch['synth_type']} patch in {elapsed:.1f}s | "
              f"spec={patch['quality']['spectral_similarity']:.3f}")

    return (SAMPLE_RATE, synth_audio), patch_text, tmp_path, status


def create_app():
    """Create the Gradio app."""
    with gr.Blocks(
        title="Patch Grabber — Inverse Synthesis",
        theme=gr.themes.Base(),
    ) as app:
        gr.Markdown("""
# Patch Grabber
Upload audio of a synth sound, get back the synthesis parameters.

Supports: subtractive synthesis (saw/square/triangle/sine/pulse/supersaw),
FM synthesis, filter LFO, and effects (distortion, delay, reverb, chorus).
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                pitch_input = gr.Number(
                    label="Pitch Override (Hz, 0 = auto-detect)",
                    value=0,
                    precision=1,
                )
                grab_btn = gr.Button("Grab Patch", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                synth_output = gr.Audio(
                    label="Re-synthesized Audio",
                    type="numpy",
                )
                patch_text = gr.Textbox(
                    label="Recovered Patch Parameters",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                )
                patch_download = gr.File(
                    label="Download Patch JSON",
                )

        grab_btn.click(
            fn=process_audio,
            inputs=[audio_input, pitch_input],
            outputs=[synth_output, patch_text, patch_download, status_text],
        )

        gr.Markdown("""
---
**How it works:**
1. Audio is loaded, normalized, and trimmed to 2 seconds
2. Pitch is auto-detected (or use manual override)
3. Neural model provides fast initial parameter estimate
4. L-BFGS-B optimizer refines filter + amp envelope parameters
5. LFO and effects are detected and optimized if needed
6. FM synthesis is tried as fallback for non-subtractive timbres
        """)

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Patch Grabber UI")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
