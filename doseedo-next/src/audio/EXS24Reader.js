/**
 * EXS24Reader — parser for Logic's EXS24 / Sampler instrument files (.exs).
 *
 * The .exs format is a binary container of fixed-size "chunks" written in
 * a stream. Each chunk has a 0x54-byte header followed by a payload whose
 * size depends on the chunk type. The format is undocumented by Apple but
 * has been reverse-engineered consistently across multiple open-source
 * projects (Sebastian Kraft / Logic-EXS24, libexs24, the SoundFont-converter
 * community). The fields below match those projects' findings; we accept
 * the conservative subset that's stable across EXS24 versions 0..3.
 *
 * Endianness:
 *   - Older files (often saved on PowerPC Logic) are big-endian.
 *   - Modern Intel/Apple-Silicon Logic writes little-endian.
 *   - Bit 0x01 of the chunk-header magic distinguishes them: the 4-byte
 *     "magic" field reads `SOBT` (little) or `TBOS` (big). We sniff the
 *     first chunk and pick a DataView accessor accordingly for the file.
 *
 * Chunk types (in the type byte at offset 4 of the header):
 *   0x00  Header chunk         (instrument name, version, sizes)
 *   0x01  Zone (key region)    — 184 bytes typical (varies by version)
 *   0x02  Group                — 80 bytes typical
 *   0x03  Sample reference     — 192 bytes (path + filename + length)
 *   0x04  Parameters / globals — variable
 *
 * Versions differ in zone-payload size (104..184 bytes). We use the chunk
 * header's `size` field to advance past unknown variants safely rather
 * than relying on a hard-coded layout. Required fields (key range, root
 * note, tuning, sample-ref index) live at fixed offsets in every observed
 * version, so we read those and treat the rest as opaque.
 *
 * Returns: { zones, groups, sampleRefs, name, version } where:
 *   zones[]      = { keyRange:[lo,hi], velRange:[lo,hi], rootNote, tuning,
 *                     pitch, pan, volume, sampleIndex, groupIndex,
 *                     sampleStart, sampleEnd, loopStart, loopEnd, loopOn }
 *   groups[]     = { name, polyphony, exclusiveGroup, volume, pan, releaseTrig }
 *   sampleRefs[] = { name, path, length, sampleRate }
 *
 * The caller is responsible for resolving sampleRefs[].path → an AudioBuffer
 * (we don't do file I/O here — the reader is pure parser). See INTEGRATION_A3.md
 * for the recommended pipeline.
 */

const CHUNK_HEADER_SIZE = 0x54;          // 84 bytes
const MAGIC_LE = 0x54424F53;             // 'SOBT' little-endian read
const MAGIC_BE = 0x534F4254;             // 'TBOS' big-endian read

const CHUNK_TYPE_HEADER     = 0x00;
const CHUNK_TYPE_ZONE       = 0x01;
const CHUNK_TYPE_GROUP      = 0x02;
const CHUNK_TYPE_SAMPLE_REF = 0x03;
const CHUNK_TYPE_PARAMS     = 0x04;

/**
 * Decode a NUL-terminated ASCII / UTF-8 string from a DataView slice.
 */
function readString(view, offset, maxLen) {
  let end = offset;
  const limit = Math.min(view.byteLength, offset + maxLen);
  while (end < limit && view.getUint8(end) !== 0) end++;
  const bytes = new Uint8Array(view.buffer, view.byteOffset + offset, end - offset);
  if (typeof TextDecoder !== 'undefined') {
    try {
      return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
    } catch (_) { /* fall through */ }
  }
  // Fallback for legacy environments
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

/**
 * Read an EXS24 chunk header.
 *
 * Layout (84 bytes):
 *   0x00 u32  size of payload (NOT including this header)
 *   0x04 u8   chunk type
 *   0x05 u8   flags
 *   0x06 u16  chunk index / id
 *   0x08 u32  magic ('SOBT' / 'TBOS')
 *   0x0C u32  reserved
 *   0x10 char[64] name (NUL-padded)
 */
function readChunkHeader(view, offset, littleEndian) {
  if (offset + CHUNK_HEADER_SIZE > view.byteLength) return null;
  const size  = view.getUint32(offset + 0x00, littleEndian);
  const type  = view.getUint8(offset + 0x04);
  const flags = view.getUint8(offset + 0x05);
  const id    = view.getUint16(offset + 0x06, littleEndian);
  const magic = view.getUint32(offset + 0x08, false);   // read raw to compare
  const name  = readString(view, offset + 0x10, 64);
  return { size, type, flags, id, magic, name, headerOffset: offset };
}

/**
 * Sniff the file's endianness by inspecting the first chunk's magic field.
 * Returns true for little-endian, false for big-endian. Defaults to LE if
 * the magic doesn't match either expected pattern (best-effort recovery).
 */
function sniffEndianness(view) {
  if (view.byteLength < 0x10 + 4) return true;
  const rawLE = view.getUint32(0x08, true);
  const rawBE = view.getUint32(0x08, false);
  if (rawLE === MAGIC_LE) return true;
  if (rawBE === MAGIC_BE) return false;
  if (rawBE === MAGIC_LE) return true;   // odd: BE-read sees LE bytes in order
  if (rawLE === MAGIC_BE) return false;
  return true; // default: modern files are little-endian
}

/**
 * Parse a single zone chunk payload. Layout (relative to payload start; all
 * little-endian on modern files, big-endian on old). Fields we extract:
 *
 *   0x00 u8    flags         (bit 0 = enabled, bit 1 = pitch tracking,
 *                              bit 2 = oneshot, bit 3 = reverse, bit 4 = loop)
 *   0x01 u8    rootNote      (MIDI 0..127)
 *   0x02 i8    fineTune      (-50..+50 cents)
 *   0x03 u8    pan           (signed -64..+63 in older files)
 *   0x04 u8    volume        (0..127)
 *   0x05 u8    coarseTune    (signed semitones, -36..+36)
 *   0x06 u8    keyLow
 *   0x07 u8    keyHigh
 *   0x08 u8    velLow
 *   0x09 u8    velHigh
 *   0x0C u32   sampleStart   (sample frames into the audio file)
 *   0x10 u32   sampleEnd
 *   0x14 u32   loopStart
 *   0x18 u32   loopEnd
 *   0x1C u32   loopCrossfade (samples)
 *   0x40 i32   sampleIndex   (-1 = none)
 *   0x44 i32   groupIndex    (-1 = none)
 *   0x48 i32   pitchBendUp   / pitchBendDown / etc.
 *
 * Older files only include fields up to ~0x68; newer up to ~0xB8. We tolerate
 * both by clamping reads to chunk-payload size.
 */
function parseZonePayload(view, offset, size, littleEndian) {
  const safeU32 = (off) => (off + 4 <= size) ? view.getUint32(offset + off, littleEndian) : 0;
  const safeI32 = (off) => (off + 4 <= size) ? view.getInt32(offset + off, littleEndian) : 0;
  const safeU8  = (off) => (off + 1 <= size) ? view.getUint8(offset + off) : 0;
  const safeI8  = (off) => (off + 1 <= size) ? view.getInt8(offset + off) : 0;

  const flags     = safeU8(0x00);
  const rootNote  = safeU8(0x01);
  const fineTune  = safeI8(0x02);
  const panRaw    = safeI8(0x03);
  const volume    = safeU8(0x04);
  const coarseTune= safeI8(0x05);
  const keyLow    = safeU8(0x06);
  const keyHigh   = safeU8(0x07);
  const velLow    = safeU8(0x08);
  const velHigh   = safeU8(0x09);

  const sampleStart    = safeU32(0x0C);
  const sampleEnd      = safeU32(0x10);
  const loopStart      = safeU32(0x14);
  const loopEnd        = safeU32(0x18);
  const loopCrossfade  = safeU32(0x1C);
  const sampleIndex    = safeI32(0x40);
  const groupIndex     = safeI32(0x44);

  return {
    flags,
    enabled:    !!(flags & 0x01),
    pitchTrack: !!(flags & 0x02),
    oneShot:    !!(flags & 0x04),
    reverse:    !!(flags & 0x08),
    loopOn:     !!(flags & 0x10),

    rootNote,
    tuning: coarseTune * 100 + fineTune,   // total cents
    pan: panRaw / 64,                       // -1..+1 (approx)
    volume: volume / 127,

    keyRange: [Math.min(keyLow, keyHigh), Math.max(keyLow, keyHigh)],
    velRange: [Math.max(1, Math.min(velLow, velHigh)), Math.min(127, Math.max(velLow, velHigh))],

    sampleStart, sampleEnd,
    loopStart, loopEnd, loopCrossfade,

    sampleIndex, groupIndex,
  };
}

/**
 * Parse a group chunk. Groups carry per-group polyphony, exclusive-group ID
 * (round-robin / choke groups), volume and pan. Fields:
 *   0x00 u32 polyphony
 *   0x04 u32 exclusiveGroup
 *   0x08 i8  volume
 *   0x09 i8  pan
 *   0x0A u8  releaseTriggered
 */
function parseGroupPayload(view, offset, size, littleEndian, name) {
  const safeU32 = (off) => (off + 4 <= size) ? view.getUint32(offset + off, littleEndian) : 0;
  const safeI8  = (off) => (off + 1 <= size) ? view.getInt8(offset + off) : 0;
  const safeU8  = (off) => (off + 1 <= size) ? view.getUint8(offset + off) : 0;

  return {
    name,
    polyphony:        safeU32(0x00),
    exclusiveGroup:   safeU32(0x04),
    volume:           safeI8(0x08),
    pan:              safeI8(0x09),
    releaseTrig:      !!safeU8(0x0A),
  };
}

/**
 * Parse a sample-reference chunk. Layout:
 *   0x00 u32 length        (sample frames)
 *   0x04 u32 sampleRate
 *   0x08 u32 bitDepth
 *   0x0C char[256] filename
 *   ...    char[1024] path  (HFS or POSIX path; on modern files often POSIX)
 *
 * The path/filename are NUL-terminated. We expose both — the caller decides
 * whether to honour the absolute path (rare, fragile) or look up by filename
 * inside an instrument-relative directory (the EXS24 convention).
 */
function parseSampleRefPayload(view, offset, size, littleEndian, name) {
  const safeU32 = (off) => (off + 4 <= size) ? view.getUint32(offset + off, littleEndian) : 0;

  const length     = safeU32(0x00);
  const sampleRate = safeU32(0x04);
  const bitDepth   = safeU32(0x08);

  // Filename block (256 bytes) is the field we care about; if the chunk
  // payload is shorter, we just return what's there.
  const fnLen = Math.min(256, Math.max(0, size - 0x0C));
  const filename = (fnLen > 0) ? readString(view, offset + 0x0C, fnLen) : '';

  // Path block — many writers omit it; treat absence as empty.
  const pathOff = 0x0C + 256;
  const pathLen = Math.max(0, Math.min(size - pathOff, 1024));
  const path = pathLen > 0 ? readString(view, offset + pathOff, pathLen) : '';

  return {
    name: name || filename,
    filename,
    path,
    length,
    sampleRate,
    bitDepth,
  };
}

/**
 * Parse the header chunk. We only need the instrument name + version.
 */
function parseHeaderPayload(view, offset, size, littleEndian, name) {
  const safeU32 = (off) => (off + 4 <= size) ? view.getUint32(offset + off, littleEndian) : 0;
  return {
    name,
    version: safeU32(0x00),
    zoneCount: safeU32(0x04),
    groupCount: safeU32(0x08),
    sampleCount: safeU32(0x0C),
  };
}

/**
 * Parse a complete .exs file (ArrayBuffer or DataView).
 *
 * Returns:
 *   {
 *     name, version,
 *     zones: [...],        // see parseZonePayload
 *     groups: [...],       // see parseGroupPayload
 *     sampleRefs: [...],   // see parseSampleRefPayload
 *     littleEndian: bool,  // sniffed
 *   }
 *
 * Caller is responsible for loading the actual sample audio. To do so:
 *   1) for each ref in result.sampleRefs, fetch the file referenced by
 *      ref.filename (look it up in a sibling "Samples" / "Audio Files"
 *      directory next to the .exs file)
 *   2) decode it (e.g. via OfflineAudioContext.decodeAudioData)
 *   3) assemble a Keymap by mapping zones[i].sampleIndex →
 *      decodedBuffers[zones[i].sampleIndex] and feeding it as
 *      `sampleBuffer` to a Keymap zone descriptor
 */
function parse(input) {
  let view;
  if (input instanceof ArrayBuffer) {
    view = new DataView(input);
  } else if (input && input.buffer instanceof ArrayBuffer) {
    // Accept Uint8Array etc.
    view = new DataView(input.buffer, input.byteOffset || 0, input.byteLength);
  } else if (input instanceof DataView) {
    view = input;
  } else {
    throw new Error('EXS24Reader.parse: expected ArrayBuffer / DataView / TypedArray');
  }

  const littleEndian = sniffEndianness(view);

  const result = {
    name: '',
    version: 0,
    littleEndian,
    zones: [],
    groups: [],
    sampleRefs: [],
  };

  let offset = 0;
  let chunkCount = 0;
  const maxChunks = 1 << 16; // safety guard against malformed files

  while (offset + CHUNK_HEADER_SIZE <= view.byteLength && chunkCount < maxChunks) {
    const hdr = readChunkHeader(view, offset, littleEndian);
    if (!hdr) break;

    const payloadOffset = offset + CHUNK_HEADER_SIZE;
    const payloadSize = hdr.size;

    // Sanity: payload mustn't run past EOF
    if (payloadOffset + payloadSize > view.byteLength) {
      // Stop gracefully — partial trailing chunk
      break;
    }

    switch (hdr.type) {
      case CHUNK_TYPE_HEADER: {
        const h = parseHeaderPayload(view, payloadOffset, payloadSize, littleEndian, hdr.name);
        result.name = h.name || result.name;
        result.version = h.version;
        break;
      }
      case CHUNK_TYPE_ZONE: {
        const z = parseZonePayload(view, payloadOffset, payloadSize, littleEndian);
        z.name = hdr.name;
        result.zones.push(z);
        break;
      }
      case CHUNK_TYPE_GROUP: {
        const g = parseGroupPayload(view, payloadOffset, payloadSize, littleEndian, hdr.name);
        result.groups.push(g);
        break;
      }
      case CHUNK_TYPE_SAMPLE_REF: {
        const s = parseSampleRefPayload(view, payloadOffset, payloadSize, littleEndian, hdr.name);
        result.sampleRefs.push(s);
        break;
      }
      case CHUNK_TYPE_PARAMS:
      default:
        // Skip unknown / parameter chunks — tolerated by spec.
        break;
    }

    offset = payloadOffset + payloadSize;
    chunkCount++;
  }

  return result;
}

/**
 * Helper to BUILD a synthetic .exs blob for testing.
 *
 * Given a manifest of:
 *   { name, zones: [{name, ...zoneFields}], groups, sampleRefs }
 * returns an ArrayBuffer that round-trips through parse() with matching
 * field values. Used by the unit test to validate the reader without
 * shipping any Apple-licensed samples.
 *
 * (Helper, not a full writer — only encodes fields parse() reads back.)
 */
function buildSyntheticExs({ name = 'Test', version = 0x10000, zones = [], groups = [], sampleRefs = [] } = {}) {
  // Total size estimate — header + each chunk
  const HEADER_PAYLOAD = 16;
  const ZONE_PAYLOAD = 0x68;          // 104 bytes — covers all fields we read
  const GROUP_PAYLOAD = 0x40;         // 64 bytes
  const SAMPLE_PAYLOAD = 0x10C;       // header fields + 256-byte filename
  const totalSize =
      CHUNK_HEADER_SIZE + HEADER_PAYLOAD
    + zones.length * (CHUNK_HEADER_SIZE + ZONE_PAYLOAD)
    + groups.length * (CHUNK_HEADER_SIZE + GROUP_PAYLOAD)
    + sampleRefs.length * (CHUNK_HEADER_SIZE + SAMPLE_PAYLOAD);

  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  let offset = 0;

  function writeChunkHeader(type, payloadSize, id, chunkName) {
    view.setUint32(offset + 0x00, payloadSize, true);
    view.setUint8(offset + 0x04, type);
    view.setUint8(offset + 0x05, 0);          // flags
    view.setUint16(offset + 0x06, id, true);
    // Magic: 'SOBT' as raw bytes 'S' 'O' 'B' 'T' = 0x53 0x4F 0x42 0x54
    view.setUint8(offset + 0x08, 0x53);
    view.setUint8(offset + 0x09, 0x4F);
    view.setUint8(offset + 0x0A, 0x42);
    view.setUint8(offset + 0x0B, 0x54);
    view.setUint32(offset + 0x0C, 0, true);   // reserved
    // Name (64 bytes, NUL-padded)
    const enc = (typeof TextEncoder !== 'undefined') ? new TextEncoder() : null;
    const bytes = enc ? enc.encode(chunkName || '') : new Uint8Array((chunkName || '').split('').map(c => c.charCodeAt(0) & 0xFF));
    const writeLen = Math.min(63, bytes.length);
    for (let i = 0; i < writeLen; i++) view.setUint8(offset + 0x10 + i, bytes[i]);
    for (let i = writeLen; i < 64; i++) view.setUint8(offset + 0x10 + i, 0);
    offset += CHUNK_HEADER_SIZE;
  }

  // Header chunk
  writeChunkHeader(CHUNK_TYPE_HEADER, HEADER_PAYLOAD, 0, name);
  view.setUint32(offset + 0x00, version, true);
  view.setUint32(offset + 0x04, zones.length, true);
  view.setUint32(offset + 0x08, groups.length, true);
  view.setUint32(offset + 0x0C, sampleRefs.length, true);
  offset += HEADER_PAYLOAD;

  // Zones
  for (let i = 0; i < zones.length; i++) {
    const z = zones[i];
    writeChunkHeader(CHUNK_TYPE_ZONE, ZONE_PAYLOAD, i, z.name || `Zone ${i}`);
    let flags = 0;
    if (z.enabled !== false) flags |= 0x01;
    if (z.pitchTrack !== false) flags |= 0x02;
    if (z.oneShot) flags |= 0x04;
    if (z.reverse) flags |= 0x08;
    if (z.loopOn) flags |= 0x10;
    view.setUint8(offset + 0x00, flags);
    view.setUint8(offset + 0x01, z.rootNote ?? 60);
    view.setInt8(offset + 0x02, Math.max(-50, Math.min(50, z.fineTune ?? 0)));
    view.setInt8(offset + 0x03, Math.max(-64, Math.min(63, Math.round((z.pan ?? 0) * 64))));
    view.setUint8(offset + 0x04, Math.round((z.volume ?? 1) * 127));
    view.setInt8(offset + 0x05, z.coarseTune ?? 0);
    view.setUint8(offset + 0x06, (z.keyRange ?? [0, 127])[0]);
    view.setUint8(offset + 0x07, (z.keyRange ?? [0, 127])[1]);
    view.setUint8(offset + 0x08, (z.velRange ?? [1, 127])[0]);
    view.setUint8(offset + 0x09, (z.velRange ?? [1, 127])[1]);
    view.setUint32(offset + 0x0C, z.sampleStart ?? 0, true);
    view.setUint32(offset + 0x10, z.sampleEnd ?? 0, true);
    view.setUint32(offset + 0x14, z.loopStart ?? 0, true);
    view.setUint32(offset + 0x18, z.loopEnd ?? 0, true);
    view.setUint32(offset + 0x1C, z.loopCrossfade ?? 0, true);
    view.setInt32(offset + 0x40, z.sampleIndex ?? -1, true);
    view.setInt32(offset + 0x44, z.groupIndex ?? -1, true);
    offset += ZONE_PAYLOAD;
  }

  // Groups
  for (let i = 0; i < groups.length; i++) {
    const g = groups[i];
    writeChunkHeader(CHUNK_TYPE_GROUP, GROUP_PAYLOAD, i, g.name || `Group ${i}`);
    view.setUint32(offset + 0x00, g.polyphony ?? 0, true);
    view.setUint32(offset + 0x04, g.exclusiveGroup ?? 0, true);
    view.setInt8(offset + 0x08, g.volume ?? 0);
    view.setInt8(offset + 0x09, g.pan ?? 0);
    view.setUint8(offset + 0x0A, g.releaseTrig ? 1 : 0);
    offset += GROUP_PAYLOAD;
  }

  // Sample refs
  for (let i = 0; i < sampleRefs.length; i++) {
    const s = sampleRefs[i];
    writeChunkHeader(CHUNK_TYPE_SAMPLE_REF, SAMPLE_PAYLOAD, i, s.name || s.filename || `Sample ${i}`);
    view.setUint32(offset + 0x00, s.length ?? 0, true);
    view.setUint32(offset + 0x04, s.sampleRate ?? 44100, true);
    view.setUint32(offset + 0x08, s.bitDepth ?? 16, true);
    const enc = (typeof TextEncoder !== 'undefined') ? new TextEncoder() : null;
    const fname = s.filename || s.name || '';
    const fnBytes = enc ? enc.encode(fname) : new Uint8Array(fname.split('').map(c => c.charCodeAt(0) & 0xFF));
    const fnLen = Math.min(255, fnBytes.length);
    for (let j = 0; j < fnLen; j++) view.setUint8(offset + 0x0C + j, fnBytes[j]);
    for (let j = fnLen; j < 256; j++) view.setUint8(offset + 0x0C + j, 0);
    offset += SAMPLE_PAYLOAD;
  }

  return buf;
}

const EXS24Reader = {
  parse,
  buildSyntheticExs,
  // Constants exposed for callers / tests
  CHUNK_TYPE_HEADER,
  CHUNK_TYPE_ZONE,
  CHUNK_TYPE_GROUP,
  CHUNK_TYPE_SAMPLE_REF,
  CHUNK_TYPE_PARAMS,
};

export default EXS24Reader;
export { parse, buildSyntheticExs };
