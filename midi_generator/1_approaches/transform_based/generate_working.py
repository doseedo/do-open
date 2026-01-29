#!/usr/bin/env python3
"""
WORKING MIDI GENERATION
=======================
Properly decodes patterns using the actual file format:
- Pattern file has contour IDs as keys (e.g., "129")
- pattern_id_to_name maps PATTERN_X -> "GM{program}_{contour_id}"
- We extract gm_program and contour_id, load contour data, apply proper decode
"""
import torch
import torch.nn as nn
import json
import sys
import os
import math
from pathlib import Path
from midiutil import MIDIFile

# Change to correct directory
os.chdir('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')

print("=" * 60)
print("WORKING MIDI GENERATION")
print("=" * 60)

# ============================================
# STEP 1: Load vocab and mappings
# ============================================
print("\n[1] Loading vocab and mappings...")

with open('piece_level_vocab.json') as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']

with open('pattern_id_to_name.json') as f:
    id_to_name = json.load(f)

# Create reverse mapping
name_to_id = {v: k for k, v in id_to_name.items()}

# Parse pattern names to get contour_id and gm_program
pattern_info = {}
needed_contours = set()
for pattern_id, name in id_to_name.items():
    # name = "GM67_129" -> gm_program=67, contour_id=129
    parts = name.replace('GM', '').split('_')
    gm_program = int(parts[0])
    contour_id = parts[1]
    pattern_info[pattern_id] = {
        'name': name,
        'gm_program': gm_program,
        'contour_id': contour_id
    }
    needed_contours.add(contour_id)

print(f"  Vocab size: {len(vocab)}")
print(f"  Pattern mappings: {len(id_to_name)}")
print(f"  Unique contours needed: {len(needed_contours)}")

# ============================================
# STEP 2: Load pattern data from extracted file
# ============================================
print("\n[2] Loading pattern data...")

# Use pre-extracted needed patterns (1.6MB instead of 2GB)
pattern_file = 'needed_patterns.json'
if not os.path.exists(pattern_file):
    print(f"  ERROR: {pattern_file} not found. Run extract_needed.py first.")
    sys.exit(1)

print(f"  Using: {pattern_file}")
with open(pattern_file) as f:
    patterns = json.load(f)

print(f"  Loaded {len(patterns)} patterns")

# ============================================
# STEP 3: Load model
# ============================================
print("\n[3] Loading model...")

# Find checkpoint
ckpt_path = None
for candidate in [
    'checkpoints/patternid/best_model.pt',
    'checkpoints/patternid_v2/best_model.pt',
    'checkpoints/patternid/checkpoint_epoch_10.pt'
]:
    if os.path.exists(candidate):
        ckpt_path = candidate
        break

if not ckpt_path:
    # Find any checkpoint
    for root, dirs, files in os.walk('checkpoints'):
        for f in files:
            if f.endswith('.pt'):
                ckpt_path = os.path.join(root, f)
                break
        if ckpt_path:
            break

print(f"  Checkpoint: {ckpt_path}")

# Define model architecture (must match training exactly)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatternIDTransformer(nn.Module):
    """Transformer for pattern ID sequence modeling."""

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, dropout=0.1, max_len=1024):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        x = self.transformer(x, mask=mask, is_causal=True)
        return self.fc_out(x)

# Load model
device = 'cpu'
vocab_size = len(vocab)
model = PatternIDTransformer(vocab_size, max_len=512)  # Must match training

checkpoint = torch.load(ckpt_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
print(f"  Model loaded successfully")

# ============================================
# STEP 4: Generate sequence
# ============================================
print("\n[4] Generating pattern sequence...")

def generate(model, vocab, start_tokens, max_length=200, temperature=0.9):
    """Generate sequence with nucleus sampling."""
    model.eval()

    # Convert vocab for lookup
    id_to_token = {v: k for k, v in vocab.items()}

    # Start with given tokens or BOS
    if start_tokens:
        current = torch.tensor([vocab.get(t, vocab['PAD']) for t in start_tokens], dtype=torch.long).unsqueeze(0)
    else:
        current = torch.tensor([[vocab.get('BOS', vocab['PAD'])]], dtype=torch.long)

    generated = list(current[0].tolist())

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(current)
            next_logits = logits[0, -1, :] / temperature

            # Top-p (nucleus) sampling
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum > 0.95
            mask[0] = False  # Keep at least top token
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()

            idx = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices[idx].item()

            generated.append(next_token)
            current = torch.cat([current, torch.tensor([[next_token]])], dim=1)

            # Stop conditions
            token_str = id_to_token.get(next_token, '')
            if token_str == 'EOS' or token_str == 'PAD':
                break

            # Limit context window
            if current.size(1) > 512:
                current = current[:, -512:]

    # Convert to tokens
    tokens = [id_to_token.get(t, f'UNK_{t}') for t in generated]
    return tokens

# Generate (use higher temp and length for variety)
tokens = generate(model, vocab, ['BOS'], max_length=500, temperature=1.0)
print(f"  Generated {len(tokens)} tokens")
print(f"  First 20: {tokens[:20]}")

# ============================================
# STEP 5: PROPER DECODE - Use all factored data
# ============================================
print("\n[5] Decoding with FULL pattern data...")

def decode_pattern(pattern_id, patterns, pattern_info, time_offset=0.0, transpose=0):
    """
    Properly decode a pattern using ALL available data:
    - canonical_pitches: actual note pitches
    - rhythm_ratios: timing between notes (actual timing!)
    - duration_ratios: note lengths (actual lengths!)
    - velocity_ratios: dynamics (actual dynamics!)
    """
    if pattern_id not in pattern_info:
        return []

    info = pattern_info[pattern_id]
    pattern_name = info['name']  # e.g., "GM67_129"
    gm_program = info['gm_program']

    if pattern_name not in patterns:
        return []

    pattern = patterns[pattern_name]

    # Get canonical pitches
    pitches = pattern.get('canonical_pitches', [])
    if not pitches:
        return []

    # Get timing data (ACTUAL rhythm ratios from corpus)
    rhythm = pattern.get('rhythm_ratios', [])
    if not rhythm:
        rhythm = [0.5] * (len(pitches) - 1)  # Default only if missing

    # Get durations (ACTUAL duration ratios from corpus)
    durations = pattern.get('duration_ratios', [])
    if not durations:
        durations = [0.4] * len(pitches)

    # Get velocities (ACTUAL velocity ratios from corpus)
    velocities = pattern.get('velocity_ratios', [])
    if not velocities:
        velocities = [0.8] * len(pitches)

    # Generate notes with REAL data
    notes = []
    current_time = time_offset

    for i, pitch in enumerate(pitches):
        # Apply transpose
        actual_pitch = pitch + transpose
        actual_pitch = max(0, min(127, actual_pitch))

        # Get ACTUAL duration for this note (scale to beats)
        dur = durations[i] if i < len(durations) else 0.4
        dur = max(0.1, dur * 0.5)  # Scale ratio to beat duration

        # Get ACTUAL velocity for this note
        vel = velocities[i] if i < len(velocities) else 0.8
        vel = int(max(40, min(127, vel * 80)))  # Scale to MIDI velocity (80 = mf)

        notes.append({
            'pitch': actual_pitch,
            'time': current_time,
            'duration': dur,
            'velocity': vel,
            'gm_program': gm_program
        })

        # Advance time using ACTUAL rhythm ratio
        if i < len(rhythm):
            current_time += max(0.1, rhythm[i] * 0.5)  # Scale to beats
        else:
            current_time += 0.25  # Default spacing

    return notes

# Parse generated tokens and decode
events = []
current_time = 0.0
patterns_used = set()
patterns_missing = set()

transpose = 0
for token in tokens:
    if token.startswith('PATTERN_'):
        decoded = decode_pattern(token, patterns, pattern_info, current_time, transpose)
        if decoded:
            events.extend(decoded)
            patterns_used.add(token)
            # Advance time by pattern duration
            if decoded:
                current_time = max(e['time'] + e['duration'] for e in decoded) + 0.1
        else:
            patterns_missing.add(token)
        transpose = 0  # Reset after use
    elif token.startswith('BEAT_'):
        # Use BEAT tokens for timing
        try:
            beat_val = int(token.replace('BEAT_', ''))
            current_time = beat_val * 0.5  # Each beat unit = 0.5 beats
        except:
            pass
    elif token.startswith('OFFSET_'):
        # OFFSET is pitch transposition
        try:
            transpose = int(token.replace('OFFSET_', '')) - 6  # Center around 0
        except:
            pass
    elif token.startswith('TIME_'):
        try:
            time_val = float(token.replace('TIME_', ''))
            current_time += time_val
        except:
            pass

print(f"  Decoded {len(events)} notes from {len(patterns_used)} patterns")
print(f"  Patterns missing contour data: {len(patterns_missing)}")

if not events:
    print("\nERROR: No events generated!")
    sys.exit(1)

# ============================================
# STEP 6: Create MIDI file
# ============================================
print("\n[6] Creating MIDI file...")

# Group by instrument
by_program = {}
for e in events:
    prog = e['gm_program']
    if prog not in by_program:
        by_program[prog] = []
    by_program[prog].append(e)

print(f"  Instruments: {sorted(by_program.keys())}")

# Create MIDI
midi = MIDIFile(len(by_program))
tempo = 120

for track_idx, (gm_prog, track_events) in enumerate(sorted(by_program.items())):
    channel = 9 if gm_prog >= 128 else track_idx % 9  # Avoid drum channel for melodic
    if channel >= 9:
        channel = (channel + 1) % 16

    midi.addTempo(track_idx, 0, tempo)
    midi.addProgramChange(track_idx, channel, 0, min(127, gm_prog))

    for e in track_events:
        midi.addNote(
            track=track_idx,
            channel=channel,
            pitch=e['pitch'],
            time=e['time'],
            duration=e['duration'],
            volume=e['velocity']
        )

    print(f"  Track {track_idx}: GM{gm_prog} - {len(track_events)} notes")

# Save
output_path = 'generated_proper.mid'
with open(output_path, 'wb') as f:
    midi.writeFile(f)

print(f"\n  Saved: {output_path}")

# ============================================
# STEP 7: Validate output
# ============================================
print("\n[7] Validating output...")

# Check file size
file_size = os.path.getsize(output_path)
print(f"  File size: {file_size} bytes")

# Analyze note distribution
all_pitches = [e['pitch'] for e in events]
all_velocities = [e['velocity'] for e in events]
all_durations = [e['duration'] for e in events]

print(f"  Total notes: {len(events)}")
print(f"  Pitch range: {min(all_pitches)} - {max(all_pitches)}")
print(f"  Velocity range: {min(all_velocities)} - {max(all_velocities)}")
print(f"  Duration range: {min(all_durations):.2f} - {max(all_durations):.2f}")
print(f"  Unique pitches: {len(set(all_pitches))}")
print(f"  Unique velocities: {len(set(all_velocities))}")

# Check for variety (not just repeating)
pitch_variety = len(set(all_pitches)) / len(all_pitches) if all_pitches else 0
vel_variety = len(set(all_velocities)) / len(all_velocities) if all_velocities else 0

print(f"\n  Quality metrics:")
print(f"    Pitch variety: {pitch_variety:.1%}")
print(f"    Velocity variety: {vel_variety:.1%}")

if pitch_variety < 0.05:
    print("    WARNING: Low pitch variety - may be repetitive")
if vel_variety < 0.03:
    print("    WARNING: Low velocity variety - may sound robotic")

# Show sample notes
print(f"\n  Sample notes (first 10):")
for i, e in enumerate(events[:10]):
    print(f"    {i}: pitch={e['pitch']}, vel={e['velocity']}, dur={e['duration']:.2f}, time={e['time']:.2f}, prog={e['gm_program']}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
