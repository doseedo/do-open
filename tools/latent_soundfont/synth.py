"""Latent-space MIDI synthesizer.

Reads a precached latent soundfont (built by latent_soundfont.build) and
turns a MIDI file into one or more [T, 64] latent tracks WITHOUT calling
fluidsynth or vae.encode at inference time. Polyphony is handled by
splitting concurrent notes into multiple parallel monophonic latent
tracks; the caller decodes each track and sums waveforms (latent space
is non-linear, so latent-sum doesn't equal audio-sum).

Public API:

  latent_sf = load_latent_soundfont("/scratch/latent_soundfonts/acoustic_piano.pt")
  tracks   = latent_synthesize_midi(midi_path, latent_sf, silence_frame)
  # tracks: List[torch.Tensor] each [T, 64] — decode + sum to get audio

Drum tracks: drum kit pitches map directly (kick=36, snare=38, …).
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import os
import numpy as np
import torch
import pretty_midi

SR = 48000
SAMPLES_PER_FRAME = 1920
FPS = SR / SAMPLES_PER_FRAME  # 25.0


def load_latent_soundfont(path: str) -> dict:
    """Load a latent soundfont .pt produced by latent_soundfont.build.
    Returns the dict {instrument, sr, samples_per_frame, note_seconds,
    velocity, is_drum, gm_program, notes: {pitch -> [T, 64]}}."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "notes" not in blob:
        raise ValueError(f"{path} is not a latent soundfont (missing 'notes')")
    return blob


def _silence(silence_frame: torch.Tensor, n: int) -> torch.Tensor:
    return silence_frame.expand(n, -1).clone()


def _assign_voices(notes: List[pretty_midi.Note]) -> List[List[pretty_midi.Note]]:
    """Greedy voice assignment for polyphony: walk notes in start order
    and place each note on the first voice whose last note ended at or
    before this note's start. Returns a list of voice tracks (each a
    list of non-overlapping notes)."""
    voices: List[List[pretty_midi.Note]] = []
    voice_last_end: List[float] = []
    for n in sorted(notes, key=lambda nn: (nn.start, nn.pitch)):
        placed = False
        for vi, last_end in enumerate(voice_last_end):
            if n.start >= last_end - 1e-6:
                voices[vi].append(n)
                voice_last_end[vi] = n.end
                placed = True
                break
        if not placed:
            voices.append([n])
            voice_last_end.append(n.end)
    return voices


def _place_note_in_track(track: torch.Tensor, note_lat: torch.Tensor,
                          start_frame: int, dur_frames: int,
                          silence_frame: torch.Tensor = None,
                          rf_guard: int = 4,
                          editor=None) -> torch.Tensor:
    """Splice note_lat into track at start_frame.

    Strategy (no editor required):
      • Overwrite the frames [start_frame - rf_guard .. start_frame)
        with silence_latent so the decoder's left receptive field reads
        true silence before the new attack. This kills the previous
        note's tail-smear without an editor.edit call.
      • Direct slice-assign the note_lat into [start_frame .. end_frame).

    Editor variant kept as a fallback (e.g. for very dense polyphony
    where pre-clearing the guard region would chop a still-ringing
    parallel voice — currently unused since voices are split into
    separate monophonic tracks)."""
    if note_lat.shape[0] == 0 or dur_frames <= 0:
        return track
    use = note_lat[:dur_frames]
    end_frame = min(track.shape[0], start_frame + use.shape[0])
    paste_n = end_frame - start_frame
    if paste_n <= 0:
        return track
    if editor is None:
        # Pre-clear left-context guard with silence latent
        if silence_frame is not None and rf_guard > 0:
            guard_lo = max(0, start_frame - rf_guard)
            if guard_lo < start_frame:
                track[guard_lo:start_frame] = silence_frame.expand(start_frame - guard_lo, -1)
        track[start_frame:start_frame + paste_n] = use[:paste_n]
        return track
    L_b = track.clone()
    L_b[start_frame:start_frame + paste_n] = use[:paste_n]
    cut_sample = start_frame * SAMPLES_PER_FRAME
    return editor.edit(track, L_b, cut_sample)


def latent_synthesize_midi(
    midi_path: str,
    latent_sf: dict,
    silence_frame: torch.Tensor,
    editor=None,
) -> List[torch.Tensor]:
    """Synthesize a MIDI file into latent-space tracks using the precached
    latent soundfont. Returns a list of [T, 64] latents (one per voice).

    Caller is responsible for decoding each voice latent with the VAE
    and summing the resulting waveforms to produce a polyphonic mix.
    For monophonic instruments the list has length 1.

    Notes whose pitch isn't in the soundfont are silently dropped (with
    a warning printed)."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes_by_pitch = latent_sf["notes"]
    is_drum = bool(latent_sf.get("is_drum", False))

    # Collect notes from instruments matching the latent soundfont's drum
    # status. (We honor the precached SF's is_drum flag — caller picks
    # the right SF for the role.)
    all_notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        if bool(inst.is_drum) != is_drum:
            continue
        all_notes.extend(inst.notes)
    if not all_notes:
        # Fall back: take everything if nothing matched
        for inst in pm.instruments:
            all_notes.extend(inst.notes)
    if not all_notes:
        return []

    end_time = max(n.end for n in all_notes)
    total_frames = int(np.ceil(end_time * FPS)) + 1

    voices = _assign_voices(all_notes)
    tracks: List[torch.Tensor] = []
    missing = set()
    RF_GUARD_FRAMES = 4
    for v_notes in voices:
        track = _silence(silence_frame, total_frames)
        # Collect all the pastes for this voice in time order.
        pastes = []
        for n in sorted(v_notes, key=lambda nn: nn.start):
            note_lat = notes_by_pitch.get(n.pitch)
            if note_lat is None:
                missing.add(n.pitch)
                continue
            start_frame = int(round(n.start * FPS))
            dur_frames  = max(1, int(round((n.end - n.start) * FPS)))
            pastes.append((note_lat, start_frame, dur_frames))

        if editor is not None and pastes:
            # Single batched forward pass for the entire voice's
            # boundary repairs (one model call instead of N).
            track = editor.edit_many_into_track(track, pastes)
        else:
            # No-editor fast path: pre-clear left-context guard with
            # silence latent so the decoder can't smear the previous
            # tail into the new attack. ~1ms per voice, no model calls.
            for note_lat, start_frame, dur_frames in pastes:
                track = _place_note_in_track(
                    track, note_lat, start_frame, dur_frames,
                    silence_frame=silence_frame, rf_guard=RF_GUARD_FRAMES,
                    editor=None,
                )
        tracks.append(track)

    if missing:
        print(f"[latent_synth] dropped {len(missing)} unmapped pitches: {sorted(missing)}")
    return tracks
