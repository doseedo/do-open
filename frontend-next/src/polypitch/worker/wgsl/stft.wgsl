// Short-time Fourier transform with sqrt-Hann analysis window.
//
// Patent reference — US 8,022,286 B2 (Neubaecker, Celemony Melodyne DNA),
// col.6 L.5-25: "the root is taken of the values of the actual window
// function and then the window is used before FFT and following inverse
// FFT". Applying sqrt(Hann) on both analysis and synthesis sides gives
// Hann as the combined window, which satisfies COLA at 4x overlap
// (hop = n_fft/4) and preserves amplitude across bin manipulation
// without fade-out artefacts at frame boundaries.
//
// This shader is stage 1/2: it writes the *windowed, zero-padded real
// frames* into the complex FFT input buffer. A subsequent FFT kernel
// dispatch consumes that buffer and produces the full complex spectrum.
// We emit only the sqrt-Hann multiplication + frame extraction here so
// the FFT kernel stays a pure transform (easier to test and reuse).
//
// Input: `pcm[frames]` mono planar float32. For stereo we run twice.
// Output: `complex[n_frames, n_fft, 2]` float32 interleaved (im=0 on
// input side; the FFT kernel fills in `im` only after forward transform).
//
// Shape rules:
//   n_frames = 1 + floor((nSamples - nFft) / hop)       if nSamples >= n_fft
//   else n_frames = 1 (first frame zero-padded)
//   The TS host precomputes n_frames and handles tail padding.
//
// Output layout for frame f, bin b:
//   out[ (f * n_fft + b) * 2 + 0 ] = pcm[f*hop + b] * sqrt_hann[b]
//   out[ (f * n_fft + b) * 2 + 1 ] = 0.0

struct Params {
  n_fft: u32,
  hop: u32,
  n_frames: u32,
  n_samples: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pcm: array<f32>;
@group(0) @binding(2) var<storage, read> window: array<f32>;  // sqrt(Hann), len n_fft
@group(0) @binding(3) var<storage, read_write> out_complex: array<f32>;

// 2-D workgroup: x = bin within frame, y = frame.
// 64 threads along x amortizes one frame across one warp/wave; y is the
// outer grid dim so we get per-frame parallelism too.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let bin = gid.x;
  let frame = gid.y;
  if (bin >= params.n_fft || frame >= params.n_frames) {
    return;
  }

  let sample_idx = frame * params.hop + bin;
  var v: f32 = 0.0;
  if (sample_idx < params.n_samples) {
    v = pcm[sample_idx] * window[bin];
  }
  let out_base = (frame * params.n_fft + bin) * 2u;
  out_complex[out_base]      = v;
  out_complex[out_base + 1u] = 0.0;
}
