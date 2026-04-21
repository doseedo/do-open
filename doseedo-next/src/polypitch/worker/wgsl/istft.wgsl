// Inverse STFT with sqrt-Hann synthesis window + overlap-add.
//
// Patent US 8,022,286 B2 col.6 L.5-25: sqrt of the window function is
// applied on both sides so the combined effective window is the full
// Hann. That makes the overlap-add sum-of-windows equal to sum_k(Hann^2)
// which for hop = n_fft/4 is a constant (≈ 1.5). We still normalize by
// the sum-of-window-squared per-sample to handle tail frames and any
// hop != n_fft/4 case gracefully.
//
// Three modes share this shader to keep the file count low:
//   mode=2  extract real parts from interleaved complex inverse-FFT output
//           into the `frames_time` scratch buffer (binding 1 is reused as
//           the complex input, binding 3 is the real destination).
//   mode=0  gather-style overlap-add from real `frames_time` into `pcm_out`.
//           Also accumulates per-sample window^2 sum into `w_sum`.
//   mode=1  normalize `pcm_out[i] /= w_sum[i]`.
//
// We split into these three passes because overlap-add requires
// per-sample accumulation from multiple frames; doing it as a gather
// (one thread per output sample, reading from the overlapping frames)
// avoids atomics entirely. For 4x overlap the inner loop is at most 4
// iterations.

struct Params {
  n_fft: u32,
  hop: u32,
  n_frames: u32,
  out_len: u32,
  mode: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// In mode 2 this is `complex_in` (interleaved re/im), else `frames_time`.
@group(0) @binding(1) var<storage, read> frames_time: array<f32>;
@group(0) @binding(2) var<storage, read> window: array<f32>;   // sqrt(Hann)
@group(0) @binding(3) var<storage, read_write> pcm_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> w_sum: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;

  if (params.mode == 2u) {
    // Extract real parts. idx = frame * n_fft + bin (flattened).
    let total = params.n_frames * params.n_fft;
    if (idx >= total) { return; }
    // frames_time is ALIASED to the complex buffer here: read stride-2.
    let re = frames_time[idx * 2u];
    pcm_out[idx] = re;  // pcm_out is aliased to real frames buffer
    return;
  }

  if (params.mode == 0u) {
    if (idx >= params.out_len) { return; }
    // k ranges over frames that touch output sample `idx`:
    //   k_min = max(0, ceil((idx - n_fft + 1) / hop))
    //   k_max = min(n_frames - 1, floor(idx / hop))
    var k_min: u32 = 0u;
    if (idx + 1u > params.n_fft) {
      k_min = (idx - params.n_fft + 1u + params.hop - 1u) / params.hop;
    }
    var k_max: u32 = idx / params.hop;
    if (k_max >= params.n_frames) {
      k_max = params.n_frames - 1u;
    }

    var acc: f32 = 0.0;
    var wacc: f32 = 0.0;
    var k: u32 = k_min;
    loop {
      if (k > k_max) { break; }
      let in_frame_bin = idx - k * params.hop;
      let win = window[in_frame_bin];
      let sample = frames_time[k * params.n_fft + in_frame_bin] * win;
      acc = acc + sample;
      wacc = wacc + win * win;
      k = k + 1u;
    }
    pcm_out[idx] = acc;
    w_sum[idx] = wacc;
    return;
  }

  // mode == 1u
  if (idx >= params.out_len) { return; }
  let w = w_sum[idx];
  if (w > 1e-12) {
    pcm_out[idx] = pcm_out[idx] / w;
  }
}
