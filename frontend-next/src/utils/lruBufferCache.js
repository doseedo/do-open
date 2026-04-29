/**
 * Byte-capped LRU cache for Web Audio AudioBuffers.
 *
 * Drop-in replacement for `new Map()`: same `.has / .get / .set / .delete /
 * .clear / .size / .keys()` surface, so existing call sites don't change.
 *
 * A 3-minute stereo 48kHz float32 AudioBuffer is ~69 MB. Without a cap,
 * a user syncing a day's worth of sessions OOMs the browser tab. Eviction
 * is LRU by access time (`.get` / `.has(url)` hits promote to MRU), and
 * `.set` evicts LRUs until total bytes fit under `maxBytes`.
 *
 * Key invariants:
 *   - Map iteration order is insertion order; `delete`+`set` moves to MRU.
 *   - Byte accounting uses `length × numberOfChannels × 4` (float32).
 *   - Oversized single buffers are accepted but log a warning; the cache
 *     evicts everything else to make room. This keeps scheduling alive
 *     even for pathological inputs; the alternative (refuse to cache)
 *     breaks playback entirely.
 */

const DEFAULT_MAX_BYTES = 512 * 1024 * 1024; // 512 MB

function _bufferBytes(buf) {
  if (!buf) return 0;
  // AudioBuffer: length (per-channel frames) × channels × 4 bytes (float32)
  const ch = Number(buf.numberOfChannels) || 1;
  const n = Number(buf.length) || 0;
  return n * ch * 4;
}

export class LRUBufferCache {
  constructor({ maxBytes = DEFAULT_MAX_BYTES, name = 'audioBuffers' } = {}) {
    this._map = new Map();
    this._bytes = 0;
    this.maxBytes = maxBytes;
    this.name = name;
    this._evictions = 0;
  }

  get size() { return this._map.size; }
  get bytes() { return this._bytes; }
  get pressure() { return this.maxBytes > 0 ? this._bytes / this.maxBytes : 0; }

  has(key) {
    if (!this._map.has(key)) return false;
    this._touch(key);
    return true;
  }

  get(key) {
    if (!this._map.has(key)) return undefined;
    const buf = this._map.get(key);
    this._touch(key);
    return buf;
  }

  set(key, buf) {
    if (this._map.has(key)) {
      this._bytes -= _bufferBytes(this._map.get(key));
      this._map.delete(key);
    }
    const size = _bufferBytes(buf);
    if (size > this.maxBytes) {
      console.warn(
        `[${this.name}] buffer ${(size / 1048576).toFixed(1)}MB exceeds cap ` +
        `${(this.maxBytes / 1048576).toFixed(0)}MB; caching anyway, evicting everything else`,
      );
    }
    this._map.set(key, buf);   // inserted at MRU end
    this._bytes += size;
    this._evictIfNeeded(key);
    return this;
  }

  delete(key) {
    if (!this._map.has(key)) return false;
    this._bytes -= _bufferBytes(this._map.get(key));
    return this._map.delete(key);
  }

  clear() {
    this._map.clear();
    this._bytes = 0;
  }

  keys() { return this._map.keys(); }
  values() { return this._map.values(); }
  entries() { return this._map.entries(); }
  forEach(fn) { this._map.forEach((v, k) => fn(v, k, this)); }
  [Symbol.iterator]() { return this._map[Symbol.iterator](); }

  /** Evict LRUs until `bytes <= maxBytes`, protecting `keepKey` (the
   *  just-inserted entry). Logs a summary on each eviction pass. */
  _evictIfNeeded(keepKey) {
    if (this._bytes <= this.maxBytes) return;
    const before = this._bytes;
    let dropped = 0;
    for (const k of this._map.keys()) {
      if (this._bytes <= this.maxBytes) break;
      if (k === keepKey) continue;
      this._bytes -= _bufferBytes(this._map.get(k));
      this._map.delete(k);
      dropped++;
      this._evictions++;
    }
    if (dropped > 0) {
      console.log(
        `[${this.name}] evicted ${dropped} entries ` +
        `(${((before - this._bytes) / 1048576).toFixed(1)}MB freed, ` +
        `${(this._bytes / 1048576).toFixed(1)}/${(this.maxBytes / 1048576).toFixed(0)}MB used)`,
      );
    }
  }

  /** Move `key` to the MRU end. Safe to call for absent keys. */
  _touch(key) {
    if (!this._map.has(key)) return;
    const v = this._map.get(key);
    this._map.delete(key);
    this._map.set(key, v);
  }
}

export default LRUBufferCache;
