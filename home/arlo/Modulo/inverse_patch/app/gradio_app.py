"""
Gradio web interface for Inverse Audio Effects.
"""

import gradio as gr
import torch
import torchaudio
import json
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional

# Import system components
from ..training.train_system import InverseAFxSystem
from ..export.daw_export import (
    chain_to_daw_preset,
    generate_processing_report,
)


# Global model instance
_model = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path: str) -> InverseAFxSystem:
    """Load model from checkpoint."""
    global _model
    if _model is None or checkpoint_path:
        _model = InverseAFxSystem.load_from_checkpoint(
            checkpoint_path,
            map_location=_device,
        )
        _model.eval()
        _model.to(_device)
    return _model


def process_audio(
    wet_audio_path: str,
    checkpoint_path: str = "checkpoints/best.ckpt",
    max_effects: int = 6,
    refine_params: bool = True,
) -> Tuple[str, str, str]:
    """
    Process wet audio to recover dry signal and effect chain.

    Args:
        wet_audio_path: Path to wet audio file
        checkpoint_path: Path to model checkpoint
        max_effects: Maximum number of effects to detect
        refine_params: Whether to refine parameters

    Returns:
        Tuple of (dry_audio_path, chain_json, report)
    """
    # Load model
    try:
        model = load_model(checkpoint_path)
    except Exception as e:
        return None, json.dumps({"error": str(e)}), f"Error loading model: {e}"

    # Load audio
    try:
        wet, sr = torchaudio.load(wet_audio_path)
    except Exception as e:
        return None, json.dumps({"error": str(e)}), f"Error loading audio: {e}"

    # Convert to mono
    if wet.size(0) > 1:
        wet = wet.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != 44100:
        resampler = torchaudio.transforms.Resample(sr, 44100)
        wet = resampler(wet)
        sr = 44100

    # Add batch dimension
    wet = wet.unsqueeze(0).to(_device)

    # Process
    with torch.no_grad():
        dry_estimate, chain = model(wet, max_iterations=max_effects)

        # Get confidence scores
        confidences = []
        for _, _, prob in chain:
            confidences.append(prob.mean().item())

    # Convert chain for export
    chain_spec = [(fx_type, params) for fx_type, params, _ in chain]

    # Refine parameters if requested
    if refine_params and len(chain_spec) > 0:
        chain_spec = model.fx_chain.refine_params(
            wet, dry_estimate, chain_spec,
            n_steps=50, lr=0.01
        )

    # Export
    chain_json = chain_to_daw_preset(chain_spec, format='json', confidence_scores=confidences)
    report = generate_processing_report(chain_spec, confidences)

    # Save dry audio
    dry_audio = dry_estimate.squeeze(0).cpu()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        dry_path = f.name
    torchaudio.save(dry_path, dry_audio, sr)

    return dry_path, json.dumps(chain_json, indent=2), report


def create_demo(
    checkpoint_path: str = "checkpoints/best.ckpt",
    share: bool = False,
) -> gr.Blocks:
    """
    Create Gradio demo interface.

    Args:
        checkpoint_path: Default model checkpoint path
        share: Whether to create public share link

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(
        title="Inverse Audio Effects",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # Inverse Audio Effects

        Upload processed (wet) audio to recover the original dry signal and
        identify the effect chain that was applied.

        ## How it works
        1. Upload your processed audio file
        2. The system analyzes the audio and identifies effects
        3. Download the recovered dry audio and effect chain specification

        ## Supported Effects
        - EQ (Parametric EQ with shelves and bands)
        - Compressor (Threshold, ratio, attack, release)
        - Reverb (Decay, pre-delay, wet mix)
        - Distortion (Drive, tone, mix)
        - Chorus (Rate, depth, mix)
        - Delay (Time, feedback, mix)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                audio_input = gr.Audio(
                    label="Upload Wet Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                with gr.Accordion("Advanced Options", open=False):
                    max_effects = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=6,
                        step=1,
                        label="Max Effects to Detect",
                    )
                    refine_params = gr.Checkbox(
                        value=True,
                        label="Refine Parameters (slower but more accurate)",
                    )
                    checkpoint = gr.Textbox(
                        value=checkpoint_path,
                        label="Model Checkpoint Path",
                    )

                process_btn = gr.Button("Process Audio", variant="primary")

            with gr.Column(scale=1):
                # Output
                audio_output = gr.Audio(
                    label="Recovered Dry Audio",
                    type="filepath",
                )

                chain_json = gr.JSON(
                    label="Estimated Effect Chain",
                )

                report = gr.Textbox(
                    label="Processing Report",
                    lines=15,
                    max_lines=30,
                )

        # Examples
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                ["examples/guitar_with_reverb.wav"],
                ["examples/vocals_compressed.wav"],
                ["examples/drums_distorted.wav"],
            ],
            inputs=[audio_input],
            outputs=[audio_output, chain_json, report],
            fn=lambda x: process_audio(x, checkpoint_path),
            cache_examples=True,
        )

        # Process button click
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, checkpoint, max_effects, refine_params],
            outputs=[audio_output, chain_json, report],
        )

        # Footer
        gr.Markdown("""
        ---
        Built with [NablAFx](https://github.com/mcomunita/nablafx) and PyTorch.

        **Limitations:**
        - Heavy distortion may not be fully invertible (information loss)
        - Very long reverb tails are challenging to remove
        - Effect order may be ambiguous for some chains
        """)

    return demo


def launch_demo(
    checkpoint_path: str = "checkpoints/best.ckpt",
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
):
    """
    Launch the Gradio demo.

    Args:
        checkpoint_path: Path to model checkpoint
        share: Create public share link
        server_name: Server hostname
        server_port: Server port
    """
    demo = create_demo(checkpoint_path, share)
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Inverse AFx demo")
    parser.add_argument(
        "--checkpoint", "-c",
        default="checkpoints/best.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Server port",
    )

    args = parser.parse_args()

    launch_demo(
        checkpoint_path=args.checkpoint,
        share=args.share,
        server_port=args.port,
    )
