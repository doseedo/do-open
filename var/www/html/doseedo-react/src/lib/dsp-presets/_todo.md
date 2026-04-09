# Plugins still pending preset conversion

These plugins can't be wrapped as a single-line preset yet because the
underlying DSP isn't packaged in a form `custom_worklet` can load. Each
needs one of:

- **(W) Worklet extraction** — the plugin defines its
  AudioWorkletProcessor inline via `URL.createObjectURL(new Blob([...]))`.
  We need to move that processor code into a standalone JS file under
  `src/lib/web-audio-plugins/worklets/`, then write a normal
  `workletPreset()` manifest pointing at it. ~30 min per plugin.

- **(G) Graph decomposition** — the plugin builds a multi-node graph
  inside its JS class (e.g., a 1176 = envelope follower → VCA → output
  filter). We trace the existing graph topology and re-express it as a
  `seriesPreset()` calling existing DSP-lang nodes. ~1–2 h per plugin.

| Plugin | Source | Path | Effort |
|---|---|---|---|
| PitchShifter | creative/PitchShifter.js | inline worklet (or use `pitch_shift` node — currently mapped) | (W) optional |
| Granular | creative/Granular.js | inline worklet | (W) ~30 min |
| FrequencyShifter (creative) | creative/FrequencyShifter.js | inline worklet | (W) ~30 min |
| FrequencyShifter (spectral) | spectral/FrequencyShifter.js | inline worklet | (W) ~30 min |
| SpectralTime | spectral/SpectralTime.js | inline worklet | (W) ~30 min |
| 1176 | vintage/VintageCompressor1176.js | JS class graph | (G) ~2 h |
| LA-2A | vintage/VintageCompressorLA2A.js | JS class graph | (G) ~2 h |
| SSL | vintage/VintageCompressorSSL.js | JS class graph | (G) ~1 h |
| Neve EQ | vintage/VintageEQNeve.js | JS class graph | (G) ~2 h |
| Analog Console | vintage/AnalogConsole.js | JS class graph | (G) ~3 h |
| Analog Modeling | vintage/AnalogModeling.js | JS class graph | (G) ~3 h |
| Spectrum Analyzer | analysis/SpectrumAnalyzer.js | analysis-only, not insert | (W) ~30 min |
| Oscilloscope | analysis/Oscilloscope.js | analysis-only | (W) ~30 min |

## Done since last revision

- Spectral Filter, Spectral Freeze, Pitch Shift → mapped to native DSP-lang nodes
- Peak Meter, RMS Meter → mapped to native DSP-lang nodes
- Polarity → wraps `gain` node (gain ∈ [-1, 1])
- Stereo Width → wraps existing `stereo-width-processor.js` worklet via
  `custom_worklet`. Required generalising buildCustomWorklet to post both
  `{type:'param',...}` and legacy `{type:'setParams',...}` formats so the
  existing worklet works without source mods.

**Total remaining work: ~14 h** (12 h graph decomposition + 2 h inline
worklet extraction).
