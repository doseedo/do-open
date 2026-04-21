// Auto-generated from fft.wgsl — keep in sync if the .wgsl is edited.
const source = `// Stockham radix-2 FFT / inverse FFT compute kernel.
//
// Processes a batch of \`batch_size\` independent complex transforms of size
// \`n_fft\` in parallel. Complex samples are stored as interleaved (re, im)
// float32 pairs; the full input buffer has length batch_size * n_fft * 2
// float32s (== batch_size * n_fft * 8 bytes).
//
// Algorithm: log2(n_fft) Stockham "auto-sort" stages ping-ponging between
// two in-shader storage buffers. Each stage reads from \`src\` and writes to
// \`dst\` with an index permutation that keeps the final output in natural
// order (no bit-reversal pass needed). We dispatch one workgroup per (stage,
// batch_row) pair conceptually; in practice the outer \`for stage in\` loop is
// inside the shader, with a workgroupBarrier() after every stage.
//
// Inverse transform: set params.inverse = 1u. We use the +j convention on
// forward (exp(-j*2*pi*k*n/N)); inverse flips the sign and scales by 1/N
// on the final stage.
//
// Math note:
//   Given a pair (a, b) at stage s with twiddle w = exp(-j*2*pi*k/M):
//     dst[2*j]   = a + w * b
//     dst[2*j+1] = a - w * b
//   Stockham layout means we walk the input in a particular strided order
//   so the output never needs bit-reversal. See:
//     Govindaraju, "High performance discrete Fourier transforms on graphics
//     processors" (SC'08), §3.1.
//
// This kernel does NOT do any windowing or real->complex packing; callers
// pad real input to complex (im=0) before dispatch.

struct Params {
  n_fft: u32,
  inverse: u32,  // 0 = forward, 1 = inverse
  batch_size: u32,
  log2_n_fft: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> buf_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> buf_b: array<f32>;

const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

// One thread per complex sample pair (output side).
// Workgroup size 64: good occupancy on both Intel and Apple GPUs.
@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let n = params.n_fft;
  let half_n = n >> 1u;
  let log2n = params.log2_n_fft;
  let batch = params.batch_size;

  // gid.x = output-pair index within [0, half_n), gid.y = batch row.
  let pair_idx = gid.x;
  let row = gid.y;
  if (pair_idx >= half_n || row >= batch) {
    return;
  }

  // Stockham writes are structured so stage s reads stride M = 1<<s and writes
  // with output stride 2*M. We ping-pong: stage 0 reads buf_a, writes buf_b;
  // stage 1 reads buf_b, writes buf_a; etc.
  //
  // All threads participate in every stage; the barrier between stages keeps
  // ordering consistent. We use a tiny helper to index complex elements.
  let row_offset = row * n * 2u;

  // Prepare: pre-scale inverse input by nothing (division happens at the end).
  // Stockham variable definitions:
  //   at stage s (s=0..log2n-1), subproblem size M = 1<<s.
  //   stride (between butterfly twins) = M.
  //   there are N/(2M) concurrent subproblems.
  for (var s: u32 = 0u; s < log2n; s = s + 1u) {
    let m = 1u << s;          // current subproblem half-size
    let m2 = m << 1u;         // current subproblem full-size
    // Which subproblem are we in, and which element within it?
    let group_idx = pair_idx / m;
    let k = pair_idx % m;     // butterfly index within subproblem

    // Twiddle angle. Forward: -2*pi*k/m2. Inverse: +2*pi*k/m2.
    var angle: f32 = -TAU * f32(k) / f32(m2);
    if (params.inverse == 1u) {
      angle = -angle;
    }
    let wr = cos(angle);
    let wi = sin(angle);

    // Stockham source indices: read from buf_a on even stages, buf_b on odd.
    // Even stage s: src=buf_a, dst=buf_b.
    // Input offsets for the butterfly pair:
    //   in_a = row_offset + 2*(group_idx*m + k)
    //   in_b = row_offset + 2*(group_idx*m + k + half_n)
    let in_a_base = row_offset + 2u * (group_idx * m + k);
    let in_b_base = row_offset + 2u * (group_idx * m + k + half_n);

    // Output position: we write interleaved within the subproblem so the
    // *next* stage can read with stride 2M.
    //   out0 = row_offset + 2*(group_idx*m2 + k)
    //   out1 = row_offset + 2*(group_idx*m2 + k + m)
    let out0_base = row_offset + 2u * (group_idx * m2 + k);
    let out1_base = row_offset + 2u * (group_idx * m2 + k + m);

    var ar: f32; var ai: f32; var br: f32; var bi: f32;
    if ((s & 1u) == 0u) {
      ar = buf_a[in_a_base]; ai = buf_a[in_a_base + 1u];
      br = buf_a[in_b_base]; bi = buf_a[in_b_base + 1u];
    } else {
      ar = buf_b[in_a_base]; ai = buf_b[in_a_base + 1u];
      br = buf_b[in_b_base]; bi = buf_b[in_b_base + 1u];
    }

    // t = w * b
    let tr = wr * br - wi * bi;
    let ti = wr * bi + wi * br;

    let y0r = ar + tr;
    let y0i = ai + ti;
    let y1r = ar - tr;
    let y1i = ai - ti;

    if ((s & 1u) == 0u) {
      buf_b[out0_base]      = y0r;
      buf_b[out0_base + 1u] = y0i;
      buf_b[out1_base]      = y1r;
      buf_b[out1_base + 1u] = y1i;
    } else {
      buf_a[out0_base]      = y0r;
      buf_a[out0_base + 1u] = y0i;
      buf_a[out1_base]      = y1r;
      buf_a[out1_base + 1u] = y1i;
    }

    // Ensure every thread has written its pair before the next stage reads.
    workgroupBarrier();
    storageBarrier();
  }

  // If log2n is odd, the final result landed in buf_b; else in buf_a.
  // Final normalization for inverse. We also may need to copy buf_b -> buf_a
  // so the caller always sees results in buf_a.
  let inv_scale = select(1.0, 1.0 / f32(n), params.inverse == 1u);
  let final_in_b = (log2n & 1u) == 1u;

  // Each thread now owns two output indices: pair_idx and pair_idx + half_n
  let off0 = row_offset + 2u * pair_idx;
  let off1 = row_offset + 2u * (pair_idx + half_n);

  var r0: f32; var i0: f32; var r1: f32; var i1: f32;
  if (final_in_b) {
    r0 = buf_b[off0]; i0 = buf_b[off0 + 1u];
    r1 = buf_b[off1]; i1 = buf_b[off1 + 1u];
    buf_a[off0]      = r0 * inv_scale;
    buf_a[off0 + 1u] = i0 * inv_scale;
    buf_a[off1]      = r1 * inv_scale;
    buf_a[off1 + 1u] = i1 * inv_scale;
  } else if (params.inverse == 1u) {
    // Already in buf_a, just scale in-place.
    buf_a[off0]      = buf_a[off0] * inv_scale;
    buf_a[off0 + 1u] = buf_a[off0 + 1u] * inv_scale;
    buf_a[off1]      = buf_a[off1] * inv_scale;
    buf_a[off1 + 1u] = buf_a[off1 + 1u] * inv_scale;
  }
}
`;
export default source;
