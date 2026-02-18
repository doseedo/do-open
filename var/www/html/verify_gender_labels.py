#!/usr/bin/env python3
"""
Verify Gender Labels - Simple verification interface for gendered manifest

Shuffle through vocal_training_manifest_gendered.json and verify if
the predicted gender labels are correct.
"""

import json
import random
from pathlib import Path
import gradio as gr


class GenderVerificationInterface:
    """Simple interface to verify gender labels."""

    def __init__(self, manifest_path, n_samples=None, filter_manifest_path=None):
        self.manifest_path = manifest_path
        self.current_idx = 0
        self.corrections = {}  # Track corrections: {audio_path: corrected_gender}
        self.removed_entries = []

        # Load manifest
        print(f"Loading manifest: {manifest_path}")
        with open(manifest_path, 'r') as f:
            self.full_manifest = json.load(f)

        # NEW: Load filter manifest if provided
        filter_paths = None
        if filter_manifest_path:
            print(f"Loading filter manifest: {filter_manifest_path}")
            with open(filter_manifest_path, 'r') as f:
                filter_manifest = json.load(f)
            filter_paths = {e.get('audio_path') for e in filter_manifest}
            print(f"   Filter manifest contains {len(filter_paths)} entries")

        # Filter to gendered vocal entries
        vocal_entries = [e for e in self.full_manifest
                        if e.get('group') == 'vocal'
                        and e.get('gender') in ['male', 'female']
                        and Path(e['audio_path']).exists()]

        # NEW: Apply filter manifest if provided
        if filter_paths:
            before_count = len(vocal_entries)
            vocal_entries = [e for e in vocal_entries if e.get('audio_path') in filter_paths]
            filtered_out = before_count - len(vocal_entries)
            print(f"   Filtered out {filtered_out} entries not in filter manifest")

        # Shuffle for random verification
        random.seed(42)
        random.shuffle(vocal_entries)

        if n_samples is None or n_samples == 0:
            self.samples = vocal_entries
        else:
            self.samples = vocal_entries[:n_samples]

        print(f"✅ Loaded {len(self.samples)} gendered vocal samples for verification")

    def get_current_sample(self):
        """Get current audio sample info."""
        if self.current_idx >= len(self.samples):
            return None, "All samples verified!", f"Verified: {self.current_idx}/{len(self.samples)}\nCorrections: {len(self.corrections)}\nRemoved: {len(self.removed_entries)}"

        entry = self.samples[self.current_idx]
        audio_path = entry['audio_path']
        filename = Path(audio_path).name

        predicted_gender = entry.get('gender', 'unknown')
        gender_confidence = entry.get('gender_confidence', 0.0)
        gender_source = entry.get('gender_source', 'unknown')
        gender_conf_cat = entry.get('gender_confidence_category', 'unknown')

        info = f"Sample {self.current_idx + 1}/{len(self.samples)}\n"
        info += f"File: {filename}\n\n"
        info += f"🤖 PREDICTED: {predicted_gender.upper()}\n"
        info += f"   Confidence: {gender_confidence:.2%}\n"
        info += f"   Category: {gender_conf_cat}\n"
        info += f"   Source: {gender_source}\n\n"
        info += f"Corrections made: {len(self.corrections)}\n"
        info += f"Removed: {len(self.removed_entries)}"

        progress = f"Progress: {self.current_idx}/{len(self.samples)} | Corrections: {len(self.corrections)} | Removed: {len(self.removed_entries)}"

        return audio_path, info, progress

    def mark_correct(self):
        """Mark current prediction as correct and move to next."""
        if self.current_idx < len(self.samples):
            self.current_idx += 1
            return self.get_current_sample()
        return None, "All done!", f"Verified: {self.current_idx}/{len(self.samples)}"

    def correct_to_male(self):
        """Correct current sample to male."""
        if self.current_idx < len(self.samples):
            entry = self.samples[self.current_idx]
            audio_path = entry['audio_path']
            original = entry.get('gender')

            if original != 'male':
                self.corrections[audio_path] = 'male'
                print(f"✏️  Corrected: {Path(audio_path).name} | {original} → male")

            self.current_idx += 1
            return self.get_current_sample()
        return None, "All done!", f"Verified: {self.current_idx}/{len(self.samples)}"

    def correct_to_female(self):
        """Correct current sample to female."""
        if self.current_idx < len(self.samples):
            entry = self.samples[self.current_idx]
            audio_path = entry['audio_path']
            original = entry.get('gender')

            if original != 'female':
                self.corrections[audio_path] = 'female'
                print(f"✏️  Corrected: {Path(audio_path).name} | {original} → female")

            self.current_idx += 1
            return self.get_current_sample()
        return None, "All done!", f"Verified: {self.current_idx}/{len(self.samples)}"

    def remove_sample(self):
        """Mark current sample for removal."""
        if self.current_idx < len(self.samples):
            entry = self.samples[self.current_idx]
            audio_path = entry['audio_path']
            self.removed_entries.append(audio_path)
            print(f"🗑️  Removed: {Path(audio_path).name}")

            self.current_idx += 1
            return self.get_current_sample()
        return None, "All done!", f"Verified: {self.current_idx}/{len(self.samples)}"

    def save_corrections(self, output_path=None):
        """Apply corrections to manifest and save."""
        if output_path is None:
            output_path = self.manifest_path

        # Apply corrections
        corrected_count = 0
        for entry in self.full_manifest:
            audio_path = entry.get('audio_path')

            if audio_path in self.corrections:
                entry['gender'] = self.corrections[audio_path]
                corrected_count += 1

        # Remove marked entries
        if self.removed_entries:
            original_count = len(self.full_manifest)
            self.full_manifest = [e for e in self.full_manifest
                                 if e.get('audio_path') not in self.removed_entries]
            removed_count = original_count - len(self.full_manifest)
        else:
            removed_count = 0

        # Save
        with open(output_path, 'w') as f:
            json.dump(self.full_manifest, f, indent=2)

        print(f"\n✅ Saved corrected manifest: {output_path}")
        print(f"   Applied {corrected_count} gender corrections")
        print(f"   Removed {removed_count} entries")

        return output_path


def verify_gender_labels_gradio(manifest_path, n_samples=None, server_port=7860, filter_manifest_path=None):
    """
    Launch Gradio interface for verifying gender labels.

    Args:
        manifest_path: Path to gendered manifest
        n_samples: Number of samples to verify (None = all)
        server_port: Port for Gradio server
        filter_manifest_path: Optional path to filter manifest (only verify entries that exist in this manifest)
    """
    interface = GenderVerificationInterface(manifest_path, n_samples, filter_manifest_path=filter_manifest_path)

    with gr.Blocks(title="Gender Label Verification") as demo:
        gr.Markdown("# 🔍 Gender Label Verification")
        gr.Markdown("**Listen and verify**: Click ✅ if correct, or click the correct gender button.")

        with gr.Row():
            with gr.Column(scale=2):
                audio_player = gr.Audio(label="Audio Sample", type="filepath")
                info_text = gr.Textbox(label="Sample Info", lines=10, interactive=False)

            with gr.Column(scale=1):
                progress_text = gr.Textbox(label="Progress", interactive=False)

                gr.Markdown("### Is the prediction correct?")
                correct_btn = gr.Button("✅ Correct - Next", variant="primary", size="lg")

                gr.Markdown("### Or correct to:")
                with gr.Row():
                    male_btn = gr.Button("👨 Male", size="lg")
                    female_btn = gr.Button("👩 Female", size="lg")

                gr.Markdown("---")
                remove_btn = gr.Button("🗑️ Remove from Manifest", variant="stop", size="lg")

                gr.Markdown("---")
                save_btn = gr.Button("💾 Save Corrections & Exit", variant="secondary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)

        # Initial load
        def load_first():
            return interface.get_current_sample()

        demo.load(load_first, outputs=[audio_player, info_text, progress_text])

        # Button actions
        correct_btn.click(interface.mark_correct, outputs=[audio_player, info_text, progress_text])
        male_btn.click(interface.correct_to_male, outputs=[audio_player, info_text, progress_text])
        female_btn.click(interface.correct_to_female, outputs=[audio_player, info_text, progress_text])
        remove_btn.click(interface.remove_sample, outputs=[audio_player, info_text, progress_text])

        def save_and_finish():
            output = interface.save_corrections()
            return f"✅ Corrections saved to: {output}\n\nCorrections: {len(interface.corrections)}\nRemoved: {len(interface.removed_entries)}\n\nClose this browser tab."

        save_btn.click(save_and_finish, outputs=[status_text])

    print(f"\n{'='*60}")
    print(f"🔍 Starting Gender Verification Interface...")
    print(f"📊 {len(interface.samples)} samples to verify")
    print(f"🌐 Opening browser at http://localhost:{server_port}")
    print(f"{'='*60}\n")
    print("Instructions:")
    print("  1. Listen to each audio sample")
    print("  2. Click ✅ if prediction is correct")
    print("  3. Click 👨 Male or 👩 Female to correct wrong predictions")
    print("  4. Click 🗑️ to mark bad samples for removal")
    print("  5. Click 💾 when done to save corrections")
    print(f"\n{'='*60}\n")

    demo.launch(server_port=server_port, inbrowser=True)

    return interface


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify gender labels in gendered manifest")
    parser.add_argument("--manifest",
                       default="/home/arlo/Data/vocal_training_manifest_gendered.json",
                       help="Path to gendered manifest")
    parser.add_argument("--filter-manifest",
                       default="/home/arlo/Data/vocal_training_manifest_filtered.json",
                       help="Path to filter manifest (only verify entries in this manifest)")
    parser.add_argument("--samples", type=int, default=None,
                       help="Number of samples to verify (default: all)")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port for Gradio interface")
    args = parser.parse_args()

    verify_gender_labels_gradio(
        manifest_path=args.manifest,
        n_samples=args.samples,
        server_port=args.port,
        filter_manifest_path=args.filter_manifest
    )


if __name__ == "__main__":
    main()
