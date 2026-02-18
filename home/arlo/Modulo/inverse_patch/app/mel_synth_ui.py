#!/usr/bin/env python3
"""
Latent Audio Explorer

Features:
- Load and play clips
- Interpolate between two clips
- PCA analysis to find meaningful directions
- Navigate along principal components
"""

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import orjson
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
from mel_to_sines_mapper import MelMapperV2


class LatentExplorer:
    def __init__(self, device='cuda'):
        self.device = device
        self.dcae = None
        self.mel_mapper = None
        self.clips = []
        self.current_z = None
        self.clip_a_z = None
        self.clip_b_z = None
        # PCA
        self.pca = None
        self.pca_mean = None
        self.pca_components = None
        self.z_collection = []

    def load_models(self):
        print("Loading DCAE...")
        self.dcae = MusicDCAE(
            dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
            vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
        )
        self.dcae.dcae.to(self.device).eval()
        self.dcae.vocoder.to(self.device).eval()

        print("Loading mel mapper...")
        self.mel_mapper = MelMapperV2().to(self.device)
        ckpt = torch.load(
            '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_mapper/best_model.pt',
            weights_only=True
        )
        self.mel_mapper.load_state_dict(ckpt['model_state_dict'])
        self.mel_mapper.eval()

        print("Loading clip manifest...")
        self._load_clips()

        print("Running PCA on latent collection...")
        self._build_pca()

        print(f"Models loaded! {len(self.clips)} clips, PCA ready.")

    def _load_clips(self):
        """Load available clips from manifest."""
        manifest_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json'
        with open(manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        for entry in manifest['entries']:
            lat_path = entry.get('latent_path', '')
            if lat_path and os.path.exists(lat_path):
                name = os.path.basename(lat_path).replace('.pt', '')
                parent = os.path.basename(os.path.dirname(os.path.dirname(lat_path)))
                display_name = f"{parent}/{name}"[:60]
                self.clips.append({
                    'name': display_name,
                    'latent_path': lat_path,
                    'sms_path': entry['path']
                })
                if len(self.clips) >= 200:
                    break

    def _build_pca(self, n_samples=100):
        """Build PCA from sample of latents."""
        print(f"Collecting {n_samples} latents for PCA...")
        z_list = []

        for clip in self.clips[:n_samples]:
            try:
                lat_data = torch.load(clip['latent_path'], weights_only=True, map_location='cpu')
                z = lat_data.get('latents', lat_data)
                if z.dim() == 3:
                    z = z.unsqueeze(0)
                # Take mean across time for a single vector per clip
                z_mean = z[:, :, :, :64].mean(dim=-1).reshape(-1).numpy()  # [128]
                z_list.append(z_mean)
            except Exception as e:
                continue

        if len(z_list) < 10:
            print("Not enough latents for PCA")
            return

        Z = np.stack(z_list)  # [N, 128]
        self.pca = PCA(n_components=min(20, len(z_list)))
        self.pca.fit(Z)
        self.pca_mean = Z.mean(axis=0)
        self.pca_components = self.pca.components_  # [n_components, 128]

        # Print variance explained
        var_exp = self.pca.explained_variance_ratio_
        print(f"PCA variance explained:")
        for i, v in enumerate(var_exp[:10]):
            print(f"  PC{i+1}: {v*100:.1f}%")
        print(f"  Top 10 total: {sum(var_exp[:10])*100:.1f}%")

    def load_latent(self, clip_name, target='current'):
        """Load latent from clip."""
        clip = next((c for c in self.clips if c['name'] == clip_name), None)
        if not clip:
            return None

        lat_data = torch.load(clip['latent_path'], weights_only=True, map_location='cpu')
        z = lat_data.get('latents', lat_data)
        if z.dim() == 3:
            z = z.unsqueeze(0)
        z = z[:, :, :, :64].to(self.device)

        if target == 'current':
            self.current_z = z
        elif target == 'a':
            self.clip_a_z = z
        elif target == 'b':
            self.clip_b_z = z
        return z

    def render_z(self, z, use_dcae=True):
        """Render z to audio."""
        with torch.no_grad():
            if use_dcae:
                z_denorm = z / self.dcae.scale_factor + self.dcae.shift_factor
                mel = self.dcae.dcae.decoder(z_denorm).mean(dim=1)
            else:
                mel = self.mel_mapper(z).permute(0, 2, 1)

            mel_scaled = mel * 0.5 + 0.5
            mel_scaled = mel_scaled * (self.dcae.max_mel_value - self.dcae.min_mel_value) + self.dcae.min_mel_value
            audio = self.dcae.vocoder.decode(mel_scaled).squeeze()
            audio = audio / (audio.abs().max() + 1e-8) * 0.9
        return audio.cpu().numpy()

    def interpolate(self, alpha):
        """Interpolate between clip A and B."""
        if self.clip_a_z is None or self.clip_b_z is None:
            return None

        # Match temporal dimensions
        T = min(self.clip_a_z.shape[-1], self.clip_b_z.shape[-1])
        z_a = self.clip_a_z[..., :T]
        z_b = self.clip_b_z[..., :T]

        # Spherical interpolation (slerp) for better results
        z_interp = self._slerp(z_a, z_b, alpha)
        return z_interp

    def _slerp(self, z1, z2, alpha):
        """Spherical linear interpolation."""
        z1_flat = z1.reshape(-1)
        z2_flat = z2.reshape(-1)

        # Normalize
        z1_norm = z1_flat / (z1_flat.norm() + 1e-8)
        z2_norm = z2_flat / (z2_flat.norm() + 1e-8)

        # Compute angle
        dot = (z1_norm * z2_norm).sum().clamp(-1, 1)
        theta = torch.acos(dot)

        if theta.abs() < 1e-4:
            # Nearly parallel, use linear
            z_interp = (1 - alpha) * z1_flat + alpha * z2_flat
        else:
            # Slerp
            sin_theta = torch.sin(theta)
            z_interp = (torch.sin((1 - alpha) * theta) / sin_theta) * z1_flat + \
                       (torch.sin(alpha * theta) / sin_theta) * z2_flat

        # Scale to match original magnitudes
        mag1, mag2 = z1_flat.norm(), z2_flat.norm()
        target_mag = (1 - alpha) * mag1 + alpha * mag2
        z_interp = z_interp / (z_interp.norm() + 1e-8) * target_mag

        return z_interp.reshape(z1.shape)

    def apply_pca_controls(self, z, pc_values):
        """Apply PCA component adjustments to z."""
        if self.pca_components is None:
            return z

        z_mod = z.clone()
        B, C, H, T = z_mod.shape
        z_flat = z_mod.reshape(B, 128, T)

        # Apply each PC as an offset
        for i, val in enumerate(pc_values):
            if i >= len(self.pca_components):
                break
            # val is centered at 0, range [-3, 3] std devs
            pc_vec = torch.from_numpy(self.pca_components[i]).float().to(self.device)
            std = np.sqrt(self.pca.explained_variance_[i])
            offset = pc_vec.unsqueeze(0).unsqueeze(-1) * val * std
            z_flat = z_flat + offset

        return z_flat.reshape(B, C, H, T)


# Global instance
explorer = None


def init_explorer():
    global explorer
    if explorer is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        explorer = LatentExplorer(device=device)
        explorer.load_models()
    return explorer


def get_clip_list():
    e = init_explorer()
    return [c['name'] for c in e.clips]


def get_pca_info():
    e = init_explorer()
    if e.pca is None:
        return "PCA not computed"
    var = e.pca.explained_variance_ratio_
    lines = ["**PCA Variance Explained:**"]
    for i in range(min(10, len(var))):
        lines.append(f"PC{i+1}: {var[i]*100:.1f}%")
    lines.append(f"**Top 10 total: {sum(var[:10])*100:.1f}%**")
    return "\n".join(lines)


# === Tab 1: Single Clip ===
def load_single_clip(clip_name):
    if not clip_name:
        return None, None, "Select a clip"
    e = init_explorer()
    z = e.load_latent(clip_name, target='current')
    if z is None:
        return None, None, "Failed to load"

    audio_dcae = e.render_z(z, use_dcae=True)
    audio_pred = e.render_z(z, use_dcae=False)
    info = f"Loaded: {clip_name}\nShape: {list(z.shape)}"
    return (44100, audio_dcae), (44100, audio_pred), info


def render_with_pca(pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8):
    e = init_explorer()
    if e.current_z is None:
        return None

    pc_values = [pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8]
    z_mod = e.apply_pca_controls(e.current_z, pc_values)
    audio = e.render_z(z_mod, use_dcae=True)
    return (44100, audio)


def reset_pca_sliders():
    return [0.0] * 8


# === Tab 2: Interpolation ===
def load_clip_a(clip_name):
    if not clip_name:
        return None, "Select clip A"
    e = init_explorer()
    z = e.load_latent(clip_name, target='a')
    if z is None:
        return None, "Failed"
    audio = e.render_z(z, use_dcae=True)
    return (44100, audio), f"Clip A: {clip_name}"


def load_clip_b(clip_name):
    if not clip_name:
        return None, "Select clip B"
    e = init_explorer()
    z = e.load_latent(clip_name, target='b')
    if z is None:
        return None, "Failed"
    audio = e.render_z(z, use_dcae=True)
    return (44100, audio), f"Clip B: {clip_name}"


def render_interpolation(alpha):
    e = init_explorer()
    z_interp = e.interpolate(alpha)
    if z_interp is None:
        return None
    audio = e.render_z(z_interp, use_dcae=True)
    return (44100, audio)


# === Tab 3: Dimension Knockout ===
def knockout_dims(dim_start, dim_end):
    """Zero out a range of dimensions to hear what they control."""
    e = init_explorer()
    if e.current_z is None:
        return None, None, "Load a clip first"

    z_mod = e.current_z.clone()
    B, C, H, T = z_mod.shape
    z_flat = z_mod.reshape(B, 128, T)

    # Zero out the specified range
    dim_start = int(dim_start)
    dim_end = int(dim_end)
    z_flat[:, dim_start:dim_end, :] = 0

    z_mod = z_flat.reshape(B, C, H, T)

    audio_orig = e.render_z(e.current_z, use_dcae=True)
    audio_knockout = e.render_z(z_mod, use_dcae=True)

    info = f"Knocked out dims {dim_start}-{dim_end}\nOriginal energy in range: {e.current_z.reshape(1,128,-1)[:,dim_start:dim_end,:].abs().mean():.4f}"

    return (44100, audio_orig), (44100, audio_knockout), info


# Build UI
with gr.Blocks(title="Latent Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Latent Audio Explorer")

    with gr.Tabs():
        # === Tab 1: PCA Control ===
        with gr.Tab("PCA Control"):
            gr.Markdown("Navigate the latent space along principal component directions.")

            with gr.Row():
                with gr.Column(scale=1):
                    clip_select = gr.Dropdown(choices=get_clip_list(), label="Select Clip", allow_custom_value=True)
                    load_btn = gr.Button("Load", variant="primary")
                    clip_info = gr.Textbox(label="Info", lines=2, interactive=False)
                    pca_info = gr.Markdown(get_pca_info())

                with gr.Column(scale=2):
                    with gr.Row():
                        audio_orig = gr.Audio(label="Original", type="numpy")
                        audio_pred = gr.Audio(label="Predicted (mel_mapper)", type="numpy")

            gr.Markdown("### Principal Component Controls")
            gr.Markdown("Each slider moves along a discovered direction. PC1 explains most variance.")

            with gr.Row():
                pc1 = gr.Slider(-3, 3, 0, step=0.1, label="PC1")
                pc2 = gr.Slider(-3, 3, 0, step=0.1, label="PC2")
                pc3 = gr.Slider(-3, 3, 0, step=0.1, label="PC3")
                pc4 = gr.Slider(-3, 3, 0, step=0.1, label="PC4")
            with gr.Row():
                pc5 = gr.Slider(-3, 3, 0, step=0.1, label="PC5")
                pc6 = gr.Slider(-3, 3, 0, step=0.1, label="PC6")
                pc7 = gr.Slider(-3, 3, 0, step=0.1, label="PC7")
                pc8 = gr.Slider(-3, 3, 0, step=0.1, label="PC8")

            with gr.Row():
                render_pca_btn = gr.Button("Render with PCA", variant="primary", size="lg")
                reset_pca_btn = gr.Button("Reset", variant="secondary")

            audio_pca = gr.Audio(label="PCA Modified", type="numpy", autoplay=True)

            pc_controls = [pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8]
            load_btn.click(load_single_clip, inputs=[clip_select], outputs=[audio_orig, audio_pred, clip_info])
            render_pca_btn.click(render_with_pca, inputs=pc_controls, outputs=[audio_pca])
            reset_pca_btn.click(reset_pca_sliders, outputs=pc_controls)

        # === Tab 2: Interpolation ===
        with gr.Tab("Interpolation"):
            gr.Markdown("Morph between two clips using spherical interpolation (slerp).")

            with gr.Row():
                with gr.Column():
                    clip_a_select = gr.Dropdown(choices=get_clip_list(), label="Clip A", allow_custom_value=True)
                    load_a_btn = gr.Button("Load A")
                    audio_a = gr.Audio(label="Clip A", type="numpy")
                    info_a = gr.Textbox(label="", lines=1, interactive=False)

                with gr.Column():
                    clip_b_select = gr.Dropdown(choices=get_clip_list(), label="Clip B", allow_custom_value=True)
                    load_b_btn = gr.Button("Load B")
                    audio_b = gr.Audio(label="Clip B", type="numpy")
                    info_b = gr.Textbox(label="", lines=1, interactive=False)

            gr.Markdown("### Interpolate")
            interp_slider = gr.Slider(0, 1, 0.5, step=0.01, label="A ←→ B")
            render_interp_btn = gr.Button("Render Interpolation", variant="primary", size="lg")
            audio_interp = gr.Audio(label="Interpolated", type="numpy", autoplay=True)

            load_a_btn.click(load_clip_a, inputs=[clip_a_select], outputs=[audio_a, info_a])
            load_b_btn.click(load_clip_b, inputs=[clip_b_select], outputs=[audio_b, info_b])
            render_interp_btn.click(render_interpolation, inputs=[interp_slider], outputs=[audio_interp])

        # === Tab 3: Dimension Knockout ===
        with gr.Tab("Dimension Knockout"):
            gr.Markdown("Zero out dimension ranges to discover what they control.")
            gr.Markdown("""
            **Known ranges:**
            - 0-47: Energy/amplitude
            - 48-63: Frequency content
            - 64-127: Temporal/other
            """)

            with gr.Row():
                dim_start = gr.Slider(0, 127, 0, step=1, label="Start Dim")
                dim_end = gr.Slider(1, 128, 48, step=1, label="End Dim")

            knockout_btn = gr.Button("Knockout & Compare", variant="primary")
            knockout_info = gr.Textbox(label="Info", lines=2, interactive=False)

            with gr.Row():
                audio_ko_orig = gr.Audio(label="Original", type="numpy")
                audio_ko_result = gr.Audio(label="After Knockout", type="numpy", autoplay=True)

            knockout_btn.click(
                knockout_dims,
                inputs=[dim_start, dim_end],
                outputs=[audio_ko_orig, audio_ko_result, knockout_info]
            )

    gr.Markdown("""
    ---
    ### What to try
    - **PCA**: Move PC1-PC8 sliders to find directions that change timbre, pitch, energy
    - **Interpolation**: Pick two different instruments and morph between them
    - **Knockout**: Zero dims 0-47 vs 48-63 vs 64-127 to hear what disappears
    """)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8098)
    parser.add_argument('--root-path', type=str, default="")
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        root_path=args.root_path if args.root_path else None,
        share=False
    )
