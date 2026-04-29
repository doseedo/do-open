// Auto-generated from hcqt.wgsl — keep in sync if the .wgsl is edited.
const source = `// Harmonic Constant-Q Transform (HCQT).
//
// Implements Bittner et al. 2017/2022 HCQT: for each harmonic scale
// h in {0.5, 1, 2, 3, 4, 5}, compute the magnitude CQT at frequencies
// f_k * h where f_k spans n_octaves * bins_per_octave cent-bins starting
// at fmin (32.7 Hz = C1). The six channels stack into
// \`[n_harmonics, n_frames, n_cent_bins]\`.
//
// We precompute the sparse CQT kernel matrix K on the CPU (Brown/Puckette
// 1992: constant-Q bank of complex-exponential analysis windows) and
// upload it as CSR: values[nnz] + indices[nnz] + rowStart[nRows+1].
// Rows correspond to cent-bins, columns to FFT bin indices.
//
// For the harmonic scaling we build ONE kernel matrix covering the full
// n_harmonics * n_octaves * bins_per_octave range (contiguous), ordered
// so that cent-bin rows for harmonic h start at \`harmonic_offsets[h]\` in
// the CSR. At runtime we project:
//    Y[h, frame, k] = | sum_j K[h_off+k, j] * X[frame, j] |
// where X is the complex STFT (NOT the magnitude — CQT uses complex
// kernels for correct phase alignment, then we take magnitude on output).
//
// Input: STFT complex \`[n_frames, n_bins_stft, 2]\` interleaved re/im.
// Output: magnitude HCQT \`[n_harmonics, n_frames, n_cent_bins]\`.
//
// Workgroup layout: one thread per (harmonic, frame, cent_bin) output
// element. We use a 3-D dispatch so the driver picks good tiling.

struct Params {
  n_frames: u32,
  n_bins_stft: u32,
  n_cent_bins_per_harmonic: u32,
  n_harmonics: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stft_complex: array<f32>;  // [n_frames, n_bins, 2]
@group(0) @binding(2) var<storage, read> k_values_re: array<f32>;
@group(0) @binding(3) var<storage, read> k_values_im: array<f32>;
@group(0) @binding(4) var<storage, read> k_indices: array<u32>;
@group(0) @binding(5) var<storage, read> k_row_start: array<u32>;   // len = n_harmonics * n_cent_bins + 1
@group(0) @binding(6) var<storage, read_write> out_hcqt: array<f32>;

// 1-D workgroup along cent_bin (innermost, contiguous output writes);
// y = frame, z = harmonic. 64-wide so each workgroup covers ~1 octave.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cent = gid.x;
  let frame = gid.y;
  let harm = gid.z;

  if (cent >= params.n_cent_bins_per_harmonic ||
      frame >= params.n_frames ||
      harm >= params.n_harmonics) {
    return;
  }

  // Row in the big CSR: harmonic block comes first.
  let row = harm * params.n_cent_bins_per_harmonic + cent;
  let row_start = k_row_start[row];
  let row_end = k_row_start[row + 1u];

  // STFT row base (complex interleaved).
  let x_base = frame * params.n_bins_stft * 2u;

  var acc_re: f32 = 0.0;
  var acc_im: f32 = 0.0;
  var j: u32 = row_start;
  loop {
    if (j >= row_end) { break; }
    let col = k_indices[j];
    let kr = k_values_re[j];
    let ki = k_values_im[j];
    let xr = stft_complex[x_base + col * 2u];
    let xi = stft_complex[x_base + col * 2u + 1u];
    // (kr + j ki) * (xr + j xi)
    acc_re = acc_re + kr * xr - ki * xi;
    acc_im = acc_im + kr * xi + ki * xr;
    j = j + 1u;
  }

  let mag = sqrt(acc_re * acc_re + acc_im * acc_im);
  // Output layout: [harmonic, frame, cent]
  let out_idx =
    harm * params.n_frames * params.n_cent_bins_per_harmonic
    + frame * params.n_cent_bins_per_harmonic
    + cent;
  out_hcqt[out_idx] = mag;
}
`;
export default source;
