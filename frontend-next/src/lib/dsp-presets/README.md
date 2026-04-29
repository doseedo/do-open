# DSP-lang preset plugins

Each file in this directory is a `dspConfig` manifest that wraps an existing
studio plugin in the universal DSP-lang format used by `/plugins/create`.

Two flavours:

1. **Native DSP-lang nodes** — when the studio plugin's algorithm is
   identical to (or a parameterisation of) an existing DSP-lang node, the
   preset is just `{type: '<node>', params: {...}}` with parameter
   bindings. No worklet involved. Examples: `compressor.js`, `gain.js`,
   `simple_delay.js`, `tremolo.js`.

2. **Custom-worklet wrappers** — when the studio plugin uses an
   `AudioWorkletProcessor` whose math isn't expressible with vanilla Web
   Audio nodes (FFT, phase vocoder, granular, exotic algorithms), the
   preset uses the universal `custom_worklet` node, pointing
   `processor_url` at the existing `lib/web-audio-plugins/worklets/*.js`
   file. **No DSP code is rewritten.** The existing processor becomes the
   preset's body. Examples: `reverb.js`, `pitch_shifter.js`,
   `granular.js`, `spectral_time.js`.

Loading a preset = loading the JSON into `WebAudioDSPEngine`. Same
runtime, same param plumbing, same UI editor — no special-casing per
plugin.
