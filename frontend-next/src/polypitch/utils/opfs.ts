/**
 * opfs — Origin Private File System caching for large model binaries.
 *
 * First call to `fetchWithOpfsCache` downloads from the URL, streams the
 * response to an OPFS file, and returns the bytes. Subsequent calls find the
 * file already present and read it back from disk without touching the network.
 *
 * OPFS is supported in Chrome/Edge 121+, Safari 18+, and Firefox 135+. When
 * unavailable the helper falls back to a plain `fetch` and emits a single
 * warn; behaviour stays correct, only the caching benefit is lost.
 *
 * This file is safe in Web Workers — OPFS is specifically designed to be used
 * from a worker (`navigator.storage.getDirectory()` exists on WorkerNavigator).
 */

export type ProgressCallback = (loaded: number, total: number) => void;

/**
 * Fetch a remote file, caching the bytes in OPFS under the given filename.
 *
 * @param url        — remote URL to fetch on cache miss.
 * @param filename   — filename within the OPFS "polypitch" directory.
 * @param onProgress — optional progress callback called with (bytesSoFar, totalBytes).
 *                     When the server omits `Content-Length`, `total` is 0.
 */
export async function fetchWithOpfsCache(
  url: string,
  filename: string,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  const root = await tryGetOpfsRoot();
  if (!root) {
    return fetchAsArrayBuffer(url, onProgress);
  }

  const dir = await root.getDirectoryHandle("polypitch", { create: true });

  // Cache hit: read back from OPFS.
  const existing = await tryGetFileHandle(dir, filename);
  if (existing) {
    const file = await existing.getFile();
    if (file.size > 0) {
      const buf = await file.arrayBuffer();
      onProgress?.(buf.byteLength, buf.byteLength);
      return buf;
    }
    // Zero-length files are treated as stale and overwritten.
  }

  // Cache miss: stream to OPFS while reporting progress.
  const handle = await dir.getFileHandle(filename, { create: true });
  const writable = await handle.createWritable();
  try {
    const bytes = await streamToWritable(url, writable, onProgress);
    return bytes;
  } catch (err) {
    // Remove any partial file so the next attempt isn't served stale junk.
    try {
      await dir.removeEntry(filename);
    } catch {
      /* ignore */
    }
    throw err;
  }
}

/**
 * Delete a cached file. Used by manual cache-invalidation flows (not wired to
 * any UI yet).
 */
export async function removeOpfsFile(filename: string): Promise<void> {
  const root = await tryGetOpfsRoot();
  if (!root) return;
  try {
    const dir = await root.getDirectoryHandle("polypitch", { create: false });
    await dir.removeEntry(filename);
  } catch {
    /* nothing to delete */
  }
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

async function tryGetOpfsRoot(): Promise<FileSystemDirectoryHandle | null> {
  try {
    if (typeof navigator === "undefined") return null;
    const storage = (navigator as Navigator & { storage?: StorageManager }).storage;
    if (!storage || typeof storage.getDirectory !== "function") {
      if (process.env.NODE_ENV === 'development') {
        // eslint-disable-next-line no-console
        console.warn("[opfs] OPFS unavailable; model downloads will not be cached.");
      }
      return null;
    }
    return await storage.getDirectory();
  } catch {
    return null;
  }
}

async function tryGetFileHandle(
  dir: FileSystemDirectoryHandle,
  filename: string,
): Promise<FileSystemFileHandle | null> {
  try {
    return await dir.getFileHandle(filename, { create: false });
  } catch {
    return null;
  }
}

async function streamToWritable(
  url: string,
  writable: FileSystemWritableFileStream,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  const resp = await fetch(url);
  if (!resp.ok) {
    await writable.close().catch(() => void 0);
    throw new Error(`fetch ${url} → HTTP ${resp.status} ${resp.statusText}`);
  }

  const total = parseContentLength(resp.headers.get("Content-Length"));
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  if (resp.body) {
    const reader = resp.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      await writable.write(value);
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.(loaded, total);
    }
  } else {
    // Some engines (older Safari in odd modes) hand back a fully-materialised
    // body. Fall back to ArrayBuffer.
    const buf = await resp.arrayBuffer();
    const view = new Uint8Array(buf);
    await writable.write(view);
    chunks.push(view);
    loaded = view.byteLength;
    onProgress?.(loaded, loaded);
  }
  await writable.close();

  // Assemble a single ArrayBuffer for the caller.
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

async function fetchAsArrayBuffer(
  url: string,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`fetch ${url} → HTTP ${resp.status} ${resp.statusText}`);
  }
  const total = parseContentLength(resp.headers.get("Content-Length"));
  if (!resp.body) {
    const buf = await resp.arrayBuffer();
    onProgress?.(buf.byteLength, total || buf.byteLength);
    return buf;
  }
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out.buffer;
}

function parseContentLength(v: string | null): number {
  if (!v) return 0;
  const n = parseInt(v, 10);
  return Number.isFinite(n) && n > 0 ? n : 0;
}
