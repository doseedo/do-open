#!/usr/bin/env python3
"""
Gender Classify Vocal Manifest using SpeechBrain

Adds gender classification to vocal manifest entries based on pitch analysis.
Uses librosa for pitch extraction (faster than loading full SpeechBrain models).
"""

import json
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_vocal_pitch(audio_path, duration=15.0):
    """
    Analyze vocal pitch to classify gender.
    
    Args:
        audio_path: Path to audio file
        duration: Seconds to analyze (default 15s)
    
    Returns:
        dict with pitch statistics or None if analysis fails
    """
    try:
        # Load audio (limit duration for speed)
        y, sr = librosa.load(audio_path, duration=duration, sr=22050)
        
        # Use YIN algorithm for pitch detection (good for vocals)
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                         fmax=librosa.note_to_hz('C6'),      # ~1047 Hz
                         sr=sr)
        
        # Filter out unvoiced frames (f0 > 0)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) < 20:  # Need reasonable amount of voiced frames
            return None
        
        return {
            'median': float(np.median(f0_voiced)),
            'mean': float(np.mean(f0_voiced)),
            'std': float(np.std(f0_voiced)),
            'min': float(np.percentile(f0_voiced, 5)),   # 5th percentile (ignore outliers)
            'max': float(np.percentile(f0_voiced, 95)),  # 95th percentile
            'voiced_frames': int(len(f0_voiced))
        }
    
    except Exception as e:
        print(f"Error analyzing {audio_path}: {e}")
        return None

def classify_gender(pitch_stats, median_threshold=165, confidence_margin=20):
    """
    Classify gender based on pitch statistics.
    
    Args:
        pitch_stats: Dict with pitch statistics
        median_threshold: Hz threshold between male/female (default 165)
        confidence_margin: Hz margin for "ambiguous" zone
    
    Returns:
        tuple: (gender, confidence)
            gender: 'male', 'female', or 'ambiguous'
            confidence: 'high' or 'low'
    """
    if pitch_stats is None:
        return 'unknown', 'none'
    
    median_f0 = pitch_stats['median']
    
    # Define thresholds
    male_high = median_threshold - confidence_margin      # 145 Hz
    female_low = median_threshold + confidence_margin     # 185 Hz
    
    if median_f0 < male_high:
        return 'male', 'high'
    elif median_f0 > female_low:
        return 'female', 'high'
    elif median_f0 < median_threshold:
        return 'male', 'low'
    elif median_f0 >= median_threshold:
        return 'female', 'low'
    else:
        return 'ambiguous', 'none'

def gender_classify_manifest(input_manifest, output_manifest, 
                             median_threshold=165,
                             sample_duration=15.0):
    """
    Add gender classification to vocal manifest.
    
    Args:
        input_manifest: Path to input manifest JSON
        output_manifest: Path to save output manifest with gender labels
        median_threshold: Hz threshold for male/female classification
        sample_duration: Seconds of audio to analyze per file
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
        'ambiguous': 0,
        'unknown': 0,
        'high_confidence': 0,
        'low_confidence': 0
    }
    
    # Process each vocal entry
    print(f"\nAnalyzing vocals (using {sample_duration}s samples)...")
    for entry in tqdm(vocal_entries):
        audio_path = entry['audio_path']
        
        if not Path(audio_path).exists():
            entry['gender'] = 'unknown'
            entry['gender_confidence'] = 'none'
            entry['pitch_stats'] = None
            stats['unknown'] += 1
            continue
        
        # Analyze pitch
        pitch_stats = analyze_vocal_pitch(audio_path, duration=sample_duration)
        
        # Classify gender
        gender, confidence = classify_gender(pitch_stats, 
                                            median_threshold=median_threshold)
        
        # Add to entry
        entry['gender'] = gender
        entry['gender_confidence'] = confidence
        entry['pitch_stats'] = pitch_stats
        
        # Update stats
        stats[gender] += 1
        if confidence == 'high':
            stats['high_confidence'] += 1
        elif confidence == 'low':
            stats['low_confidence'] += 1
    
    # Save updated manifest
    print(f"\nSaving gender-classified manifest: {output_manifest}")
    with open(output_manifest, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Print statistics
    print(f"\n=== Gender Classification Results ===")
    print(f"Male voices:     {stats['male']:4d} ({stats['male']/len(vocal_entries)*100:.1f}%)")
    print(f"Female voices:   {stats['female']:4d} ({stats['female']/len(vocal_entries)*100:.1f}%)")
    print(f"Ambiguous:       {stats['ambiguous']:4d} ({stats['ambiguous']/len(vocal_entries)*100:.1f}%)")
    print(f"Unknown/Failed:  {stats['unknown']:4d} ({stats['unknown']/len(vocal_entries)*100:.1f}%)")
    print(f"\nConfidence breakdown:")
    print(f"High confidence: {stats['high_confidence']:4d}")
    print(f"Low confidence:  {stats['low_confidence']:4d}")
    
    # Show sample classifications
    print(f"\n=== Sample Classifications ===")
    for gender in ['male', 'female', 'ambiguous']:
        samples = [e for e in vocal_entries if e.get('gender') == gender][:3]
        if samples:
            print(f"\n{gender.upper()}:")
            for entry in samples:
                filename = Path(entry['audio_path']).name
                pitch = entry.get('pitch_stats', {})
                if pitch:
                    median_f0 = pitch.get('median', 0)
                    confidence = entry.get('gender_confidence', 'none')
                    print(f"  {filename}: {median_f0:.1f} Hz ({confidence} confidence)")
    
    print(f"\nDone! Gender-classified manifest saved to: {output_manifest}")

def main():
    input_manifest = "/home/arlo/Data/vocal_training_manifest.json"
    output_manifest = "/home/arlo/Data/vocal_training_manifest_gendered.json"
    
    # Adjust these parameters if needed:
    # - median_threshold: Hz dividing male/female (default 165)
    # - sample_duration: seconds to analyze per file (default 15)
    
    gender_classify_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        median_threshold=165,
        sample_duration=15.0
    )

if __name__ == "__main__":
    main()