#!/usr/bin/env python3
"""
Gender Classify Vocal Manifest using VoiceGenderRecognition

Adds gender classification to vocal manifest entries using the CNN model
from /home/arlo/Data/VoiceGenderRecognition.

First attempts to classify by filename (detecting gendered names),
then falls back to CNN model for unclassified entries.
"""

import json
import os
import sys
import numpy as np
import librosa
import librosa.display
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
import re
warnings.filterwarnings('ignore')

# Add VoiceGenderRecognition to path
VGR_PATH = '/home/arlo/Data/VoiceGenderRecognition'
sys.path.insert(0, VGR_PATH)

from tensorflow.keras.models import load_model

# Load the CNN model
MODEL_PATH = os.path.join(VGR_PATH, 'models', 'latestCNN')
print(f"Loading gender recognition model from {MODEL_PATH}...")
modelCNN = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Ollama configuration for LLM-based filename gender detection
OLLAMA_MODEL = "phi3"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def batch_detect_gender_from_filenames_llm(filenames, batch_size=50):
    """
    Batch detect gender from filenames using Ollama LLM (phi3).

    Args:
        filenames: List of audio filenames or paths
        batch_size: Number of filenames to process per LLM call

    Returns:
        dict: Mapping of filename -> (gender, source)
    """
    import requests

    results = {}

    # Process in batches with progress bar
    total_batches = (len(filenames) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(filenames), batch_size), start=1):
        batch = filenames[i:i+batch_size]

        # Extract just filenames without paths and extensions
        name_parts = []
        for filename in batch:
            basename = os.path.basename(filename)
            name_part = os.path.splitext(basename)[0]
            name_parts.append(name_part)

        # Build batch prompt
        filenames_list = "\n".join([f"{idx+1}. {name}" for idx, name in enumerate(name_parts)])

        prompt = f"""Analyze these filenames and determine if they contain gendered names. these are vocal files, they will often contain "VOX" or "Soprano" - These are not gendered names. If it contains "Sara" or "John" these are gendered names. 

Filenames:
{filenames_list}

For each filename, respond with ONLY the number followed by a colon and one word (male/female/unknown).
Example format:
1: male
2: female
3: unknown

Answer:"""

        # Log progress
        progress_pct = (batch_idx / total_batches) * 100
        print(f"[LLM Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%)] Processing {len(batch)} filenames...", end=' ', flush=True)

        try:
            import time
            start_time = time.time()

            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": batch_size * 5  # ~5 tokens per response line
                    }
                },
                timeout=30
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()

                # Parse responses line by line
                classified_count = 0
                for line in answer.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        try:
                            idx_str, gender = line.split(':', 1)
                            idx = int(idx_str.strip()) - 1
                            gender = gender.strip().lower()

                            if 0 <= idx < len(batch):
                                filename = batch[idx]
                                if 'male' in gender and 'female' not in gender:
                                    results[filename] = ('male', 'llm')
                                    classified_count += 1
                                elif 'female' in gender:
                                    results[filename] = ('female', 'llm')
                                    classified_count += 1
                                else:
                                    results[filename] = (None, None)
                        except (ValueError, IndexError):
                            continue

                print(f"Done! ({elapsed:.2f}s, {classified_count} gendered names found)")

            # Mark any unprocessed files in this batch as None
            for filename in batch:
                if filename not in results:
                    results[filename] = (None, None)

        except Exception as e:
            print(f"Failed! ({str(e)})")
            # If Ollama fails for this batch, mark all as None
            for filename in batch:
                results[filename] = (None, None)

    return results

def predict_gender_from_audio(audio_path, temp_dir='/tmp/gender_classify'):
    """
    Predict gender from audio file using VoiceGenderRecognition CNN model.

    Args:
        audio_path: Path to audio file
        temp_dir: Directory for temporary spectrogram images

    Returns:
        tuple: (gender, confidence) where gender is 'male' or 'female'
               and confidence is a float between 0 and 1
    """
    fig = None
    try:
        # Create temp directory if needed
        os.makedirs(temp_dir, exist_ok=True)

        # Load audio
        sound, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', sr=None)

        # Create spectrogram image (same as VoiceGenderRecognition)
        fig = plt.figure(figsize=[1, 1])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        # Generate mel spectrogram
        spectrogram = librosa.feature.melspectrogram(y=sound, sr=sample_rate)
        librosa.display.specshow(
            librosa.power_to_db(spectrogram, ref=np.max),
            fmin=50, fmax=280, x_axis='time', y_axis='mel'
        )

        # Save spectrogram image
        temp_image = os.path.join(temp_dir, 'temp_spec.jpg')
        plt.savefig(temp_image, dpi=500, pad_inches=0, bbox_inches='tight')

        # Close figure and clear memory immediately
        plt.close(fig)
        plt.close('all')  # Ensure all figures are closed
        fig = None

        # Load and preprocess image
        img = skimage.io.imread(temp_image)
        img = skimage.transform.resize(img, (64, 64, 3))
        img = img[np.newaxis, ...]

        # Predict
        prediction = modelCNN.predict(img, verbose=0)

        # Clean up temp image
        if os.path.exists(temp_image):
            os.remove(temp_image)

        # Interpret prediction
        # prediction[0][0] = Female confidence
        # prediction[0][1] = Male confidence
        female_conf = float(prediction[0][0])
        male_conf = float(prediction[0][1])

        # Explicitly delete large arrays
        del sound, spectrogram, img, prediction

        if female_conf > male_conf:
            return 'female', female_conf
        else:
            return 'male', male_conf

    except Exception as e:
        print(f"Error predicting gender for {audio_path}: {e}")
        return 'unknown', 0.0
    finally:
        # Ensure figure is closed even on exception
        if fig is not None:
            plt.close(fig)
        plt.close('all')

def gender_classify_manifest(input_manifest, output_manifest, save_interval=100, use_llm=False, llm_batch_size=50):
    """
    Add gender classification to vocal manifest using VoiceGenderRecognition.
    Optionally batch processes filenames with LLM first, then uses CNN for remaining files.
    Saves progress incrementally every save_interval entries.

    Args:
        input_manifest: Path to input manifest JSON
        output_manifest: Path to save output manifest with gender labels
        save_interval: Save progress every N entries (default 100)
        use_llm: Whether to use Ollama LLM for filename detection (default False)
        llm_batch_size: Batch size for LLM processing (default 50)
    """
    print(f"Loading manifest: {input_manifest}")

    with open(input_manifest, 'r') as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")

    # Filter to only vocal entries
    vocal_entries = [entry for entry in data if entry.get('group') == 'vocal']
    print(f"Vocal entries to classify: {len(vocal_entries)}")

    # Statistics tracking
    stats = {
        'male': 0,
        'female': 0,
        'unknown': 0,
        'high_confidence': 0,  # > 0.8
        'medium_confidence': 0,  # 0.6 - 0.8
        'low_confidence': 0,   # < 0.6
        'from_filename': 0,
        'from_model': 0
    }

    # Don't validate file existence upfront - handle during processing
    # This prevents blocking for 38k+ file existence checks
    entries_needing_cnn = []

    # PHASE 1: Optionally batch process ALL filenames with LLM first
    if use_llm:
        print(f"\n=== PHASE 1: Batch processing filenames with LLM (phi3) ===")
        print(f"Processing {len(vocal_entries)} filenames in batches of {llm_batch_size}...")

        # Batch process all filenames with LLM
        all_paths = [e['audio_path'] for e in vocal_entries]
        llm_results = batch_detect_gender_from_filenames_llm(all_paths, batch_size=llm_batch_size)

        # Apply LLM results and identify entries that need CNN
        for entry in vocal_entries:
            audio_path = entry['audio_path']
            gender, source = llm_results.get(audio_path, (None, None))

            if gender:
                # LLM detected gender
                entry['gender'] = gender
                entry['gender_confidence'] = 1.0
                entry['gender_confidence_category'] = 'llm'
                entry['gender_source'] = 'llm'
                stats[gender] += 1
                stats['from_filename'] += 1
            else:
                # Need CNN for this one
                entries_needing_cnn.append(entry)

        print(f"LLM classified: {stats['from_filename']} entries")
        print(f"Remaining for CNN: {len(entries_needing_cnn)} entries")
    else:
        # Skip LLM, process all with CNN
        entries_needing_cnn = vocal_entries
        print(f"\n=== LLM disabled, using CNN for all {len(entries_needing_cnn)} entries ===")

    # PHASE 2: Process remaining entries with CNN
    if entries_needing_cnn:
        print(f"\n=== PHASE 2: Processing remaining {len(entries_needing_cnn)} files with CNN ===")
        print(f"Saving progress every {save_interval} entries to {output_manifest}")

        for idx, entry in enumerate(tqdm(entries_needing_cnn), start=1):
            audio_path = entry['audio_path']

            # Check if file exists before processing
            if not Path(audio_path).exists():
                entry['gender'] = 'unknown'
                entry['gender_confidence'] = 0.0
                entry['gender_confidence_category'] = 'missing'
                entry['gender_source'] = 'file_not_found'
                stats['unknown'] += 1
                continue

            # Use CNN model
            gender, confidence = predict_gender_from_audio(audio_path)

            # Categorize confidence
            if confidence > 0.8:
                confidence_cat = 'high'
                stats['high_confidence'] += 1
            elif confidence > 0.6:
                confidence_cat = 'medium'
                stats['medium_confidence'] += 1
            else:
                confidence_cat = 'low'
                stats['low_confidence'] += 1

            # Add to entry
            entry['gender'] = gender
            entry['gender_confidence'] = confidence
            entry['gender_confidence_category'] = confidence_cat
            entry['gender_source'] = 'model'

            # Update stats
            stats[gender] += 1
            stats['from_model'] += 1

            # Periodic garbage collection to prevent memory buildup
            if idx % 50 == 0:
                import gc
                gc.collect()

            # Save progress every save_interval entries
            if idx % save_interval == 0:
                with open(output_manifest, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"\n[Progress saved at {idx}/{len(entries_needing_cnn)} CNN entries]")

    # Final save
    print(f"\nSaving final gender-classified manifest: {output_manifest}")
    with open(output_manifest, 'w') as f:
        json.dump(data, f, indent=2)

    # Print statistics
    print(f"\n=== Gender Classification Results ===")
    print(f"Male voices:     {stats['male']:4d} ({stats['male']/len(vocal_entries)*100:.1f}%)")
    print(f"Female voices:   {stats['female']:4d} ({stats['female']/len(vocal_entries)*100:.1f}%)")
    print(f"Unknown/Failed:  {stats['unknown']:4d} ({stats['unknown']/len(vocal_entries)*100:.1f}%)")
    print(f"\nSource breakdown:")
    print(f"From LLM (filename): {stats['from_filename']:4d} ({stats['from_filename']/len(vocal_entries)*100:.1f}%)")
    print(f"From CNN model:      {stats['from_model']:4d} ({stats['from_model']/len(vocal_entries)*100:.1f}%)")
    print(f"\nModel confidence breakdown (for CNN classified only):")
    print(f"High confidence (>0.8):   {stats['high_confidence']:4d}")
    print(f"Medium confidence (0.6-0.8): {stats['medium_confidence']:4d}")
    print(f"Low confidence (<0.6):    {stats['low_confidence']:4d}")

    # Show sample classifications
    print(f"\n=== Sample Classifications ===")
    for gender in ['male', 'female']:
        samples = [e for e in vocal_entries if e.get('gender') == gender][:3]
        if samples:
            print(f"\n{gender.upper()}:")
            for entry in samples:
                filename = Path(entry['audio_path']).name
                conf = entry.get('gender_confidence', 0.0)
                source = entry.get('gender_source', 'unknown')
                conf_cat = entry.get('gender_confidence_category', 'none')
                print(f"  {filename}: {source} source, {conf:.3f} confidence ({conf_cat})")

    print(f"\nDone! Gender-classified manifest saved to: {output_manifest}")

def main():
    import sys

    input_manifest = "/home/arlo/Data/vocal_training_manifest_filtered.json"
    output_manifest = "/home/arlo/Data/vocal_training_manifest_filtered_gendered.json"

    # Check for --use-llm flag
    use_llm = '--use-llm' in sys.argv

    if use_llm:
        print("LLM mode enabled (Ollama with phi3)")
    else:
        print("LLM mode disabled (CNN only)")

    gender_classify_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        use_llm=use_llm
    )

if __name__ == "__main__":
    main()
