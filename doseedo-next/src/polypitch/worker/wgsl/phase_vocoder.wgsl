// Classical Laroche-Dolson phase-vocoder pitch shifter (bin-wise variant).
//
// Reference: Laroche & Dolson, "New phase-vocoder techniques for real-time
// pitch shifting" (JAES 1999). For each STFT frame we:
//   1) measure the instantaneous frequency at each bin from phase deltas
//      between this frame and the previous frame,
//   2) scale the IF by the target shift ratio,
//   3) reconstruct phase with the scaled IF and re-emit complex bins,
//   4) optionally perform bin-to-bin phase coherence ("identity phase
//      locking") on peak neighborhoods so harmonics stay phase-aligned.
//
// This kernel is designed for SMALL shifts (<= 3 semitones). Large
// shifts go through the DDSP head in a different module.
//
// Input/output are the SAME buffer (in-place). We need the previous
// frame's original phase, so we maintain a `prev_phase` buffer bound
// separately — the host clears it at pipeline reset.
//
// Layout: complex STFT `[n_frames, n_bins, 2]` interleaved.
// `pitch_shift_ratios` is a `[n_frames]` float32 buffer — per-frame
// shift factor (2^(semitones/12)).
//
// We operate frame-by-frame (serial across frames, parallel across bins)
// because the phase accumulator depends on the previous frame's state.
// The host dispatches `n_frames` times with `frame_idx` in params.

struct Params {
  n_frames: u32,
  n_bins: u32,
  n_fft: u32,
  hop: u32,
  frame_idx: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> stft_complex: array<f32>;
@group(0) @binding(2) var<storage, read> pitch_shift_ratios: array<f32>;
@group(0) @binding(3) var<storage, read_write> prev_in_phase: array<f32>;   // [n_bins]
@group(0) @binding(4) var<storage, read_write> accum_out_phase: array<f32>; // [n_bins]

const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

fn wrap_pi(x: f32) -> f32 {
  // Wrap to (-pi, pi].
  var y = x;
  y = y - TAU * floor((y + PI) / TAU);
  return y;
}

// Bins per workgroup. Inner-axis 64 matches stft.wgsl.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let bin = gid.x;
  if (bin >= params.n_bins) { return; }

  let frame = params.frame_idx;
  let base = (frame * params.n_bins + bin) * 2u;
  let xr = stft_complex[base];
  let xi = stft_complex[base + 1u];

  let mag = sqrt(xr * xr + xi * xi);
  let phase = atan2(xi, xr);
  let ratio = pitch_shift_ratios[frame];

  if (frame == 0u) {
    // First frame: identity pass, seed phase accumulator.
    prev_in_phase[bin] = phase;
    accum_out_phase[bin] = phase;
    // No change to bins (ratio scaling with no history is ill-defined;
    // the host is expected to prepend a silent warmup frame).
    return;
  }

  let prev_p = prev_in_phase[bin];
  // Expected phase advance per hop for this bin's center frequency.
  let omega_expected = TAU * f32(bin) * f32(params.hop) / f32(params.n_fft);
  // Phase deviation, wrapped.
  let delta = wrap_pi(phase - prev_p - omega_expected);
  // Instantaneous frequency (radians per sample).
  let inst_omega = (omega_expected + delta) / f32(params.hop);

  // Scale the IF by the shift ratio.
  let shifted_omega = inst_omega * ratio;
  // Advance the output phase accumulator.
  let new_out_phase = accum_out_phase[bin] + shifted_omega * f32(params.hop);
  accum_out_phase[bin] = new_out_phase;
  prev_in_phase[bin] = phase;

  // Re-emit complex bin with updated phase, preserving magnitude.
  stft_complex[base]      = mag * cos(new_out_phase);
  stft_complex[base + 1u] = mag * sin(new_out_phase);
}
