// Magnitude spectrum from complex STFT.
//
// Given complex `[n_frames, n_bins, 2]` (interleaved re/im), produces
// `[n_frames, n_bins]` with either |X| (power=1) or |X|^2 (power=2).
//
// Used by:
//   - harmonic-prior postprocess on the HCQT (power=2 before softmax)
//   - mask U-Net input (power=1, i.e. magnitude)
//
// Threading: one thread per output element. We use a 1-D dispatch over
// the flattened [n_frames * n_bins] grid since the work per thread is a
// couple of FMAs + sqrt — kernel-launch overhead dominates over clever
// workgroup layouts at this scale.

struct Params {
  n_frames: u32,
  n_bins: u32,
  power: u32,  // 1 or 2
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> complex_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> mag_out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.n_frames * params.n_bins;
  let idx = gid.x;
  if (idx >= total) { return; }

  let base = idx * 2u;
  let re = complex_in[base];
  let im = complex_in[base + 1u];
  let p2 = re * re + im * im;
  var v: f32;
  if (params.power == 2u) {
    v = p2;
  } else {
    v = sqrt(p2);
  }
  mag_out[idx] = v;
}
