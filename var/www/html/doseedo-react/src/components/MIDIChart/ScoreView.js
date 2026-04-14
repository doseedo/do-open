import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Renderer, Stave, StaveNote, Formatter, Voice, Accidental } from 'vexflow';

/**
 * ScoreView — renders the MIDI notes as sheet music using VexFlow.
 *
 * Inputs: array of `{note, time, duration, velocity}` (time + duration in
 * beats), same shape MIDIChart uses. Notes are quantized to standard
 * durations (whole / half / quarter / eighth / sixteenth), packed into
 * 4/4 bars, and rendered clef by clef.
 *
 * Click interaction: clicking on the clef region at the start of the
 * first line cycles treble → bass → alto → tenor → treble. Clef choice
 * is persisted in local state and re-rendered immediately.
 */
const CLEF_CYCLE = ['treble', 'bass', 'alto', 'tenor'];

// Standard VexFlow durations, longest first. Used to greedy-split a note
// whose beat-length doesn't match a single duration symbol cleanly
// (e.g. 1.5 beats → quarter + eighth tied).
const DUR_TABLE = [
  { beats: 4,    code: 'w'  },
  { beats: 2,    code: 'h'  },
  { beats: 1,    code: 'q'  },
  { beats: 0.5,  code: '8'  },
  { beats: 0.25, code: '16' },
  { beats: 0.125,code: '32' },
];

const NOTE_NAMES = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];

/** MIDI number → VexFlow key ("c/4", "f#/5", ...). */
function midiToVexKey(midi) {
  const octave = Math.floor(midi / 12) - 1;
  const name = NOTE_NAMES[midi % 12];
  return `${name}/${octave}`;
}

/** Pull the accidental (if any) out of a VexFlow key so we can attach
 *  a proper Accidental modifier — VexFlow won't render sharps/flats
 *  from the key string alone. */
function accidentalFor(key) {
  if (key.includes('#')) return '#';
  if (key.includes('b/')) return 'b';
  return null;
}

/** Snap a continuous beat value to the nearest representable duration,
 *  returning an array of VexFlow duration codes that sum to `beats`.
 *  Approximate notes shorter than a 32nd are dropped. */
function quantizeDuration(beats) {
  const out = [];
  let remaining = beats;
  // Snap to 16ths to avoid drift from float comparisons.
  remaining = Math.round(remaining * 16) / 16;
  for (const { beats: b, code } of DUR_TABLE) {
    while (remaining >= b - 1e-6) {
      out.push(code);
      remaining -= b;
    }
  }
  if (out.length === 0) out.push('16');
  return out;
}

/** Group notes into bars of `barBeats` beats, inserting rests for gaps.
 *  Overlapping notes at the same start time are merged into a single
 *  chord (VexFlow `StaveNote` with multiple keys). */
function buildBars(notes, barBeats = 4) {
  if (!notes || notes.length === 0) return [];
  const sorted = [...notes].sort((a, b) => a.time - b.time);
  const totalBeats = sorted.reduce((m, n) => Math.max(m, n.time + n.duration), 0);
  const nBars = Math.max(1, Math.ceil(totalBeats / barBeats));

  // Chord-group by identical onset time (rounded to 16th-beats).
  const chords = [];
  for (const n of sorted) {
    const t = Math.round(n.time * 16) / 16;
    const last = chords[chords.length - 1];
    if (last && Math.abs(last.time - t) < 1e-6 && Math.abs(last.duration - n.duration) < 1e-6) {
      last.keys.push(midiToVexKey(n.note));
    } else {
      chords.push({ time: t, duration: n.duration, keys: [midiToVexKey(n.note)] });
    }
  }

  const bars = [];
  for (let b = 0; b < nBars; b++) bars.push([]);

  let cursor = 0; // current beat position inside the flattened timeline
  for (const c of chords) {
    // Insert rests to fill the gap from `cursor` up to `c.time`.
    if (c.time > cursor + 1e-6) {
      const gap = c.time - cursor;
      for (const code of quantizeDuration(gap)) {
        const barIdx = Math.min(nBars - 1, Math.floor(cursor / barBeats));
        bars[barIdx].push({ kind: 'rest', code });
        cursor += DUR_TABLE.find(d => d.code === code).beats;
      }
    }
    // Emit the chord itself — may span multiple bars if the duration is
    // long enough; splitting honours bar boundaries with tied rests.
    let remain = c.duration;
    while (remain > 1e-6) {
      const barIdx = Math.min(nBars - 1, Math.floor(cursor / barBeats));
      const barStart = barIdx * barBeats;
      const barEnd = barStart + barBeats;
      const thisSlice = Math.min(remain, barEnd - cursor);
      for (const code of quantizeDuration(thisSlice)) {
        bars[barIdx].push({ kind: 'note', code, keys: c.keys });
        cursor += DUR_TABLE.find(d => d.code === code).beats;
      }
      remain -= thisSlice;
    }
  }
  // Pad trailing rests so every bar totals exactly `barBeats`.
  for (let b = 0; b < nBars; b++) {
    const filled = bars[b].reduce((s, e) => s + (DUR_TABLE.find(d => d.code === e.code)?.beats || 0), 0);
    let gap = barBeats - filled;
    while (gap > 1e-6) {
      for (const code of quantizeDuration(gap)) {
        bars[b].push({ kind: 'rest', code });
        gap -= DUR_TABLE.find(d => d.code === code).beats;
      }
      break;
    }
  }
  return bars;
}

/** Turn a bar (array of {kind, code, keys}) into VexFlow StaveNotes. */
function barToStaveNotes(bar, clef) {
  return bar.map(evt => {
    if (evt.kind === 'rest') {
      // Rest key midline varies by clef but b/4 works for all — VexFlow
      // auto-positions rests ignoring the key.
      return new StaveNote({ clef, keys: ['b/4'], duration: evt.code + 'r' });
    }
    const note = new StaveNote({ clef, keys: evt.keys, duration: evt.code });
    evt.keys.forEach((k, i) => {
      const acc = accidentalFor(k);
      if (acc) note.addModifier(new Accidental(acc), i);
    });
    return note;
  });
}

const ScoreView = ({
  notes,
  tempo = 120,
  width = 900,
  height = 480,
  timeSignature = [4, 4],
}) => {
  const containerRef = useRef(null);
  const clefZonesRef = useRef([]); // [{x, y, w, h}] in container coords
  const [clef, setClef] = useState('treble');

  const handleClick = useCallback((e) => {
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    for (const z of clefZonesRef.current) {
      if (x >= z.x && x <= z.x + z.w && y >= z.y && y <= z.y + z.h) {
        setClef(prev => CLEF_CYCLE[(CLEF_CYCLE.indexOf(prev) + 1) % CLEF_CYCLE.length]);
        return;
      }
    }
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = '';
    clefZonesRef.current = [];

    const renderer = new Renderer(container, Renderer.Backends.SVG);
    renderer.resize(width, height);
    const context = renderer.getContext();
    context.setFont('Arial', 10);

    const barBeats = timeSignature[0] * (4 / timeSignature[1]);
    const bars = buildBars(notes || [], barBeats);

    // Layout: 4 bars per line, each bar sized to fit container width,
    // 100px vertical per stave.
    const barsPerLine = 4;
    const margin = 10;
    const staveWidth = Math.floor((width - margin * 2) / barsPerLine);
    const staveHeight = 110;

    bars.forEach((bar, i) => {
      const lineIdx = Math.floor(i / barsPerLine);
      const col = i % barsPerLine;
      const x = margin + col * staveWidth;
      const y = margin + lineIdx * staveHeight;
      const w = staveWidth;
      const stave = new Stave(x, y, w);
      // Put the clef + time-signature on the first bar of each line.
      if (col === 0) {
        stave.addClef(clef);
        if (i === 0) stave.addTimeSignature(`${timeSignature[0]}/${timeSignature[1]}`);
      }
      stave.setContext(context).draw();

      // Record clef hit-zone (left 50px of first bar in every line).
      if (col === 0) {
        clefZonesRef.current.push({ x, y, w: 50, h: staveHeight - 10 });
      }

      const vexNotes = barToStaveNotes(bar, clef);
      if (vexNotes.length === 0) return;
      try {
        const voice = new Voice({ numBeats: barBeats, beatValue: timeSignature[1] });
        voice.setStrict(false);
        voice.addTickables(vexNotes);
        new Formatter().joinVoices([voice]).format([voice], w - 40);
        voice.draw(context, stave);
      } catch (err) {
        // If formatting fails (e.g., notes don't sum to a full bar due to
        // quantization drift), skip this bar gracefully rather than
        // crashing the whole component.
        // eslint-disable-next-line no-console
        console.warn('ScoreView bar render failed:', err?.message || err);
      }
    });
  }, [notes, clef, width, height, timeSignature]);

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      title={`Clef: ${clef} (click the clef to cycle)`}
      style={{
        width: `${width}px`,
        height: `${height}px`,
        background: '#fdfdfa',
        color: '#111',
        borderRadius: 4,
        overflow: 'auto',
        cursor: 'default',
        userSelect: 'none',
      }}
    />
  );
};

export default ScoreView;
