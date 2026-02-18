#!/usr/bin/env python3
"""
Gender Classify Vocal Manifest using SpeechBrain

Adds gender classification to vocal manifest entries using SpeechBrain's
speaker recognition model + trained gender classifier.
"""

import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import gradio as gr
import random

class SpeechBrainGenderClassifier:
    def __init__(self):
        """Initialize SpeechBrain speaker encoder."""
        print("Loading SpeechBrain encoder (this may take a minute)...")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        self.classifier = None
        
    def extract_embedding(self, audio_path, max_duration=15.0):
        """Extract speaker embedding from audio file."""
        try:
            # Load audio
            signal, sr = torchaudio.load(audio_path)
            
            # Limit duration
            max_samples = int(max_duration * sr)
            if signal.shape[1] > max_samples:
                signal = signal[:, :max_samples]
            
            # Resample if needed (SpeechBrain expects 16kHz)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                signal = resampler(signal)
            
            # Convert to mono if stereo
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(signal)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding from {audio_path}: {e}")
            return None
    
    def train_classifier(self, labeled_data):
        """
        Train gender classifier on labeled subset.
        
        Args:
            labeled_data: List of (audio_path, gender) tuples
                         where gender is 'male' or 'female'
        """
        print(f"Training gender classifier on {len(labeled_data)} labeled samples...")
        
        embeddings = []
        labels = []
        
        for audio_path, gender in tqdm(labeled_data, desc="Extracting embeddings"):
            emb = self.extract_embedding(audio_path)
            if emb is not None:
                embeddings.append(emb)
                labels.append(1 if gender == 'female' else 0)
        
        X = np.array(embeddings)
        y = np.array(labels)
        
        print(f"Training logistic regression classifier...")
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(X, y)
        
        # Report training accuracy
        train_acc = self.classifier.score(X, y)
        print(f"Training accuracy: {train_acc*100:.1f}%")
        
    def classify(self, audio_path):
        """
        Classify gender of audio file.
        
        Returns:
            tuple: (gender, confidence)
                gender: 'male' or 'female'
                confidence: probability score (0-1)
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier() first.")
        
        emb = self.extract_embedding(audio_path)
        if emb is None:
            return 'unknown', 0.0
        
        # Get prediction and probability
        pred = self.classifier.predict([emb])[0]
        proba = self.classifier.predict_proba([emb])[0]
        
        gender = 'female' if pred == 1 else 'male'
        confidence = float(proba[pred])
        
        return gender, confidence
    
    def save_classifier(self, path):
        """Save trained classifier to disk."""
        if self.classifier is None:
            raise ValueError("No classifier to save.")
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Classifier saved to: {path}")
    
    def load_classifier(self, path):
        """Load trained classifier from disk."""
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
        print(f"Classifier loaded from: {path}")

class GradioLabelingInterface:
    """Gradio web interface for labeling audio files."""

    def __init__(self, manifest_path, n_samples=None, labels_json="labels.json", verification_mode=False):
        self.manifest_path = manifest_path
        self.n_samples = n_samples
        self.labels_json = labels_json
        self.verification_mode = verification_mode  # NEW: verification mode
        self.samples = []
        self.current_idx = 0
        self.labeled_data = {}  # Changed to dict: {audio_path: {gender, range, ethnicity, soul, weight}}
        self.removed_entries = []  # Track entries to remove from manifest

        # Current sample's temporary labels
        self.current_labels = {
            'range': None,
            'ethnicity': 'none',
            'soul': 5,
            'weight': 5
        }

        # Load existing labels if available
        if Path(labels_json).exists():
            try:
                with open(labels_json, 'r') as f:
                    self.labeled_data = json.load(f)
                print(f"✅ Loaded {len(self.labeled_data)} existing labels from {labels_json}")
            except Exception as e:
                print(f"⚠️ Could not load labels: {e}")
                self.labeled_data = {}

        # Load and sample data
        with open(manifest_path, 'r') as f:
            self.full_manifest = json.load(f)

        # NEW: Different behavior for verification mode
        if verification_mode:
            # In verification mode, load gendered entries
            vocal_entries = [e for e in self.full_manifest
                           if e.get('group') == 'vocal'
                           and e.get('gender') in ['male', 'female']
                           and Path(e['audio_path']).exists()]

            # Shuffle for random verification
            random.seed(42)
            random.shuffle(vocal_entries)

            if n_samples is None or n_samples == 0:
                self.samples = vocal_entries
                print(f"✅ Verification mode: Loaded {len(self.samples)} gendered vocal samples")
            else:
                self.samples = vocal_entries[:n_samples]
                print(f"✅ Verification mode: Loaded {len(self.samples)} gendered vocal samples")
        else:
            # Original labeling mode
            vocal_entries = [e for e in self.full_manifest if e.get('group') == 'vocal'
                            and Path(e['audio_path']).exists()]

            # Filter out already-labeled samples
            unlabeled_entries = [e for e in vocal_entries
                                if e['audio_path'] not in self.labeled_data]

            # If n_samples is None or 0, use all unlabeled samples
            if n_samples is None or n_samples == 0:
                self.samples = unlabeled_entries
                print(f"Loaded ALL {len(self.samples)} unlabeled vocal samples (skipping {len(self.labeled_data)} already labeled)")
            else:
                random.seed(42)
                self.samples = random.sample(unlabeled_entries, min(n_samples, len(unlabeled_entries)))
                print(f"Loaded {len(self.samples)} unlabeled samples (skipping {len(self.labeled_data)} already labeled)")

    def get_current_sample(self):
        """Get current audio sample info."""
        if self.current_idx >= len(self.samples):
            return None, "All samples labeled!", f"Labeled: {len(self.labeled_data)}/{len(self.samples)} | Removed: {len(self.removed_entries)}", None, "none", 5, 5

        entry = self.samples[self.current_idx]
        audio_path = entry['audio_path']
        filename = Path(audio_path).name

        info = f"Sample {self.current_idx + 1}/{len(self.samples)}\n"
        info += f"File: {filename}\n"
        info += f"Path: {audio_path}\n"

        # NEW: Show predicted gender in verification mode
        if self.verification_mode:
            predicted_gender = entry.get('gender', 'unknown')
            gender_confidence = entry.get('gender_confidence', 0.0)
            gender_source = entry.get('gender_source', 'unknown')
            info += f"\n🤖 PREDICTED: {predicted_gender.upper()}\n"
            info += f"   Confidence: {gender_confidence:.2%}\n"
            info += f"   Source: {gender_source}\n"

        info += f"\nLabeled so far: {len(self.labeled_data)}\n"
        info += f"Removed: {len(self.removed_entries)}"

        # Reset current labels for new sample
        self.current_labels = {
            'range': None,
            'ethnicity': 'none',
            'soul': 5,
            'weight': 5
        }

        return audio_path, info, f"Progress: {self.current_idx}/{len(self.samples)}", None, "none", 5, 5

    def _save_labels(self):
        """Save labels to JSON file."""
        try:
            with open(self.labels_json, 'w') as f:
                json.dump(self.labeled_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save labels: {e}")

    def update_range(self, range_val):
        """Update current sample's range."""
        self.current_labels['range'] = range_val
        return range_val

    def update_ethnicity(self, ethnicity_val):
        """Update current sample's ethnicity."""
        self.current_labels['ethnicity'] = ethnicity_val
        return ethnicity_val

    def update_soul(self, soul_val):
        """Update current sample's soul slider."""
        self.current_labels['soul'] = int(soul_val)
        return soul_val

    def update_weight(self, weight_val):
        """Update current sample's weight slider."""
        self.current_labels['weight'] = int(weight_val)
        return weight_val

    def label_male(self):
        """Label current sample as male."""
        if self.current_idx < len(self.samples):
            audio_path = self.samples[self.current_idx]['audio_path']
            self.labeled_data[audio_path] = {
                'gender': 'male',
                'range': self.current_labels['range'],
                'ethnicity': self.current_labels['ethnicity'],
                'soul': self.current_labels['soul'],
                'weight': self.current_labels['weight']
            }
            self._save_labels()
            self.current_idx += 1
            return self.get_current_sample()
        return None, "Done!", f"Labeled: {len(self.labeled_data)}/{len(self.samples)}", None, "none", 5, 5

    def label_female(self):
        """Label current sample as female."""
        if self.current_idx < len(self.samples):
            audio_path = self.samples[self.current_idx]['audio_path']
            self.labeled_data[audio_path] = {
                'gender': 'female',
                'range': self.current_labels['range'],
                'ethnicity': self.current_labels['ethnicity'],
                'soul': self.current_labels['soul'],
                'weight': self.current_labels['weight']
            }
            self._save_labels()
            self.current_idx += 1
            return self.get_current_sample()
        return None, "Done!", f"Labeled: {len(self.labeled_data)}/{len(self.samples)} | Removed: {len(self.removed_entries)}", None, "none", 5, 5

    def skip_sample(self):
        """Skip current sample."""
        if self.current_idx < len(self.samples):
            self.current_idx += 1
            return self.get_current_sample()
        return None, "Done!", f"Labeled: {len(self.labeled_data)}/{len(self.samples)} | Removed: {len(self.removed_entries)}", None, "none", 5, 5

    def remove_from_manifest(self):
        """Mark current sample for removal from manifest."""
        if self.current_idx < len(self.samples):
            entry = self.samples[self.current_idx]
            audio_path = entry['audio_path']
            self.removed_entries.append(audio_path)
            self.current_idx += 1
            return self.get_current_sample()
        return None, "Done!", f"Labeled: {len(self.labeled_data)}/{len(self.samples)} | Removed: {len(self.removed_entries)}", None, "none", 5, 5

    def get_labeled_data(self):
        """Return all labeled data as list of tuples for training."""
        # Convert to old format for backward compatibility with classifier training
        result = []
        for path, labels in self.labeled_data.items():
            if isinstance(labels, dict):
                result.append((path, labels['gender']))
            else:
                # Old format compatibility
                result.append((path, labels))
        return result

    def save_updated_manifest(self, output_path=None):
        """Save manifest with removed entries excluded."""
        if not self.removed_entries:
            print("No entries to remove.")
            return

        if output_path is None:
            output_path = self.manifest_path

        # Filter out removed entries
        updated_manifest = [e for e in self.full_manifest
                          if e.get('audio_path') not in self.removed_entries]

        with open(output_path, 'w') as f:
            json.dump(updated_manifest, f, indent=2)

        removed_count = len(self.full_manifest) - len(updated_manifest)
        print(f"✅ Saved updated manifest: {output_path}")
        print(f"   Removed {removed_count} entries ({len(self.removed_entries)} marked)")
        return output_path

def create_training_set_gradio(manifest_path, n_samples=None, share=False, server_port=7860, labels_json="labels.json", verification_mode=False):
    """
    Web interface for labeling audio samples.

    Args:
        manifest_path: Path to vocal manifest
        n_samples: Number of samples to label (None = all samples)
        share: Whether to create a public link
        server_port: Port to run the server on
        labels_json: Path to persistent labels JSON file
        verification_mode: If True, shuffle through gendered entries to verify labels

    Returns:
        List of (audio_path, gender) tuples
    """
    interface = GradioLabelingInterface(manifest_path, n_samples, labels_json, verification_mode=verification_mode)

    with gr.Blocks(title="Audio Gender Labeling") as demo:
        if verification_mode:
            gr.Markdown("# 🔍 Gender Verification Interface")
            gr.Markdown("**Verification Mode**: Review predicted gender labels. Click the correct gender button to confirm or correct.")
        else:
            gr.Markdown("# Audio Gender Labeling Interface")
            gr.Markdown("Listen to each audio sample and label the gender of the voice.")

        with gr.Row():
            with gr.Column(scale=2):
                audio_player = gr.Audio(label="Audio Sample", type="filepath")
                info_text = gr.Textbox(label="Sample Info", lines=5, interactive=False)

            with gr.Column(scale=1):
                progress_text = gr.Textbox(label="Progress", interactive=False)

                gr.Markdown("### Vocal Range:")
                range_radio = gr.Radio(
                    choices=["bass", "tenor", "alto", "soprano"],
                    label="Range",
                    value=None
                )

                gr.Markdown("### Ethnicity:")
                ethnicity_radio = gr.Radio(
                    choices=["none", "latin", "black", "asian"],
                    label="Ethnicity",
                    value="none"
                )

                gr.Markdown("### Sliders:")
                soul_slider = gr.Slider(
                    minimum=1, maximum=10, step=1, value=5,
                    label="Soul (1-10)"
                )
                weight_slider = gr.Slider(
                    minimum=1, maximum=10, step=1, value=5,
                    label="Weight (1-10)"
                )

                gr.Markdown("### Gender:")
                male_btn = gr.Button("👨 Male", variant="primary", size="lg")
                female_btn = gr.Button("👩 Female", variant="primary", size="lg")
                skip_btn = gr.Button("⏭️ Skip", size="lg")

                gr.Markdown("### Actions:")
                remove_btn = gr.Button("🗑️ Remove from Manifest", variant="stop", size="lg")

                gr.Markdown("---")
                finish_btn = gr.Button("✅ Finish & Save Labels", variant="secondary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)

        # Initial load
        def load_first():
            return interface.get_current_sample()

        demo.load(load_first, outputs=[audio_player, info_text, progress_text, range_radio, ethnicity_radio, soul_slider, weight_slider])

        # Update callbacks for controls
        range_radio.change(interface.update_range, inputs=[range_radio], outputs=[range_radio])
        ethnicity_radio.change(interface.update_ethnicity, inputs=[ethnicity_radio], outputs=[ethnicity_radio])
        soul_slider.change(interface.update_soul, inputs=[soul_slider], outputs=[soul_slider])
        weight_slider.change(interface.update_weight, inputs=[weight_slider], outputs=[weight_slider])

        # Button actions
        male_btn.click(interface.label_male, outputs=[audio_player, info_text, progress_text, range_radio, ethnicity_radio, soul_slider, weight_slider])
        female_btn.click(interface.label_female, outputs=[audio_player, info_text, progress_text, range_radio, ethnicity_radio, soul_slider, weight_slider])
        skip_btn.click(interface.skip_sample, outputs=[audio_player, info_text, progress_text, range_radio, ethnicity_radio, soul_slider, weight_slider])
        remove_btn.click(interface.remove_from_manifest, outputs=[audio_player, info_text, progress_text, range_radio, ethnicity_radio, soul_slider, weight_slider])

        def finish_labeling():
            labeled = interface.get_labeled_data()
            removed = len(interface.removed_entries)

            # Save updated manifest if entries were removed
            if removed > 0:
                interface.save_updated_manifest()
                return f"✅ Labeling complete!\n{len(labeled)} samples labeled.\n{removed} entries removed from manifest.\nClose this browser tab and return to terminal."
            else:
                return f"✅ Labeling complete! {len(labeled)} samples labeled.\nClose this browser tab and return to terminal."

        finish_btn.click(finish_labeling, outputs=[status_text])

    print(f"\n{'='*60}")
    print(f"🎵 Starting Gradio labeling interface...")
    print(f"📊 {len(interface.samples)} samples to label")
    print(f"🌐 Opening browser at http://localhost:{server_port}")
    print(f"{'='*60}\n")
    print("Instructions:")
    print("  1. Listen to each audio sample")
    print("  2. Click 'Male' or 'Female' to label")
    print("  3. Click 'Skip' to skip a sample")
    print("  4. Click 'Remove from Manifest' to exclude bad samples")
    print("  5. Click 'Finish & Save Labels' when done")
    print(f"\n{'='*60}\n")

    demo.launch(share=share, server_port=server_port, inbrowser=True)

    # Return labeled data after interface closes
    labeled = interface.get_labeled_data()
    print(f"\n✅ Labeling complete! {len(labeled)} samples labeled.")
    return labeled

def filter_manifest_by_duration(input_manifest, output_manifest, min_duration=6.0):
    """
    Remove entries from manifest with audio shorter than min_duration seconds.

    Args:
        input_manifest: Input manifest JSON path
        output_manifest: Output manifest JSON path
        min_duration: Minimum duration in seconds (default: 6.0)
    """
    print(f"Loading manifest: {input_manifest}")
    with open(input_manifest, 'r') as f:
        data = json.load(f)

    print(f"Filtering entries shorter than {min_duration} seconds...")
    filtered = []
    removed_count = 0

    for entry in tqdm(data, desc="Checking durations"):
        audio_path = entry.get('audio_path')

        if not audio_path or not Path(audio_path).exists():
            # Keep entries without audio (might be non-vocal)
            filtered.append(entry)
            continue

        try:
            # Get audio duration
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate

            if duration >= min_duration:
                filtered.append(entry)
            else:
                removed_count += 1
                print(f"  Removed: {Path(audio_path).name} ({duration:.2f}s)")
        except Exception as e:
            # If we can't read the file, keep it in manifest
            print(f"  Warning: Could not read {audio_path}: {e}")
            filtered.append(entry)

    # Save filtered manifest
    print(f"\nSaving filtered manifest: {output_manifest}")
    with open(output_manifest, 'w') as f:
        json.dump(filtered, f, indent=2)

    print(f"\n=== Filter Results ===")
    print(f"Original entries: {len(data)}")
    print(f"Kept:            {len(filtered)}")
    print(f"Removed:         {removed_count}")
    print(f"Saved to: {output_manifest}")

def create_training_set_interactive(manifest_path, n_samples=50):
    """
    CLI version - kept for backwards compatibility.
    Use create_training_set_gradio() for web interface.
    """
    with open(manifest_path, 'r') as f:
        data = json.load(f)

    vocal_entries = [e for e in data if e.get('group') == 'vocal'
                    and Path(e['audio_path']).exists()]

    random.seed(42)
    samples = random.sample(vocal_entries, min(n_samples, len(vocal_entries)))

    print(f"\nInteractive labeling: Listen and label {len(samples)} samples")
    print("Enter 'm' for male, 'f' for female, 's' to skip")

    labeled = []
    for i, entry in enumerate(samples):
        audio_path = entry['audio_path']
        filename = Path(audio_path).name

        print(f"\n[{i+1}/{len(samples)}] {filename}")
        print(f"Path: {audio_path}")

        label = input("Gender (m/f/s): ").strip().lower()

        if label == 'm':
            labeled.append((audio_path, 'male'))
        elif label == 'f':
            labeled.append((audio_path, 'female'))
        else:
            print("Skipped")

    print(f"\nLabeled {len(labeled)} samples")
    return labeled

def gender_classify_manifest_speechbrain(input_manifest, output_manifest,
                                        classifier_path=None,
                                        training_samples=None):
    """
    Add gender classification using SpeechBrain.
    
    Args:
        input_manifest: Input manifest JSON path
        output_manifest: Output manifest JSON path
        classifier_path: Path to save/load trained classifier
        training_samples: List of (audio_path, gender) tuples for training
    """
    print("Initializing SpeechBrain gender classifier...")
    classifier = SpeechBrainGenderClassifier()
    
    # Load or train classifier
    if classifier_path and Path(classifier_path).exists():
        print(f"Loading pre-trained classifier: {classifier_path}")
        classifier.load_classifier(classifier_path)
    elif training_samples:
        classifier.train_classifier(training_samples)
        if classifier_path:
            classifier.save_classifier(classifier_path)
    else:
        raise ValueError("Must provide either classifier_path or training_samples")
    
    # Load manifest
    print(f"Loading manifest: {input_manifest}")
    with open(input_manifest, 'r') as f:
        data = json.load(f)
    
    vocal_entries = [e for e in data if e.get('group') == 'vocal']
    print(f"Classifying {len(vocal_entries)} vocal entries...")
    
    # Statistics
    stats = {'male': 0, 'female': 0, 'unknown': 0, 'high_conf': 0, 'low_conf': 0}
    
    # Classify each vocal
    for entry in tqdm(vocal_entries):
        audio_path = entry['audio_path']
        
        if not Path(audio_path).exists():
            entry['gender'] = 'unknown'
            entry['gender_confidence'] = 0.0
            stats['unknown'] += 1
            continue
        
        gender, confidence = classifier.classify(audio_path)
        
        entry['gender'] = gender
        entry['gender_confidence'] = float(confidence)
        
        stats[gender] += 1
        if confidence > 0.75:
            stats['high_conf'] += 1
        else:
            stats['low_conf'] += 1
    
    # Save
    print(f"Saving gender-classified manifest: {output_manifest}")
    with open(output_manifest, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Report
    total = len(vocal_entries)
    print(f"\n=== Gender Classification Results ===")
    print(f"Male:            {stats['male']:4d} ({stats['male']/total*100:.1f}%)")
    print(f"Female:          {stats['female']:4d} ({stats['female']/total*100:.1f}%)")
    print(f"Unknown:         {stats['unknown']:4d} ({stats['unknown']/total*100:.1f}%)")
    print(f"\nHigh conf (>75%): {stats['high_conf']:4d}")
    print(f"Low conf (<=75%): {stats['low_conf']:4d}")
    
    print(f"\nDone! Saved to: {output_manifest}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gender classification for vocal manifest")
    parser.add_argument("--manifest", default="/home/arlo/Data/vocal_training_manifest.json",
                       help="Input vocal manifest path")
    parser.add_argument("--output", default="/home/arlo/Data/vocal_training_manifest_gendered.json",
                       help="Output manifest path")
    parser.add_argument("--classifier", default="/home/arlo/Data/gender_classifier.pkl",
                       help="Classifier model path")
    parser.add_argument("--samples", type=int, default=None,
                       help="Number of samples to label (default: all samples)")
    parser.add_argument("--labels", default="/home/arlo/Data/labels.json",
                       help="Path to labels JSON file (persistent across sessions)")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port for Gradio interface")
    parser.add_argument("--share", action="store_true",
                       help="Create public Gradio link")
    parser.add_argument("--cli", action="store_true",
                       help="Use CLI interface instead of web interface")
    parser.add_argument("--filter-duration", action="store_true",
                       help="Filter manifest to remove audio shorter than --min-duration")
    parser.add_argument("--min-duration", type=float, default=6.0,
                       help="Minimum duration in seconds for --filter-duration (default: 6.0)")
    parser.add_argument("--verify", action="store_true",
                       help="Verification mode: shuffle through gendered manifest to verify labels")
    args = parser.parse_args()

    input_manifest = args.manifest
    output_manifest = args.output
    classifier_path = args.classifier

    # Option 0: Verification mode
    if args.verify:
        print("🔍 Starting verification mode...")
        print(f"Loading gendered manifest: {args.manifest}")
        create_training_set_gradio(
            manifest_path=args.manifest,
            n_samples=args.samples,
            share=args.share,
            server_port=args.port,
            labels_json=args.labels,
            verification_mode=True
        )
        return

    # Option 1: Filter by duration if requested
    if args.filter_duration:
        filter_manifest_by_duration(input_manifest, output_manifest, args.min_duration)
        return

    # Option 2: Interactive training (first time)
    if not Path(classifier_path).exists():
        print("No trained classifier found. Creating training set...")

        if args.cli:
            # Use old CLI interface
            training_data = create_training_set_interactive(input_manifest, n_samples=args.samples)
        else:
            # Use new Gradio web interface
            training_data = create_training_set_gradio(input_manifest, n_samples=args.samples,
                                                      share=args.share, server_port=args.port,
                                                      labels_json=args.labels)

        if len(training_data) < 10:
            print("Not enough labeled samples. Need at least 10.")
            return

        gender_classify_manifest_speechbrain(
            input_manifest=input_manifest,
            output_manifest=output_manifest,
            classifier_path=classifier_path,
            training_samples=training_data
        )

    # Option 2: Use existing classifier
    else:
        gender_classify_manifest_speechbrain(
            input_manifest=input_manifest,
            output_manifest=output_manifest,
            classifier_path=classifier_path
        )

if __name__ == "__main__":
    main()