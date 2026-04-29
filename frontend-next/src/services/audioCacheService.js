/**
 * Audio Cache Service - Caches audio files using IndexedDB
 *
 * This service caches audio blobs locally so they persist across page refreshes.
 * When loading audio:
 * 1. Check local IndexedDB cache first
 * 2. If not cached, fetch from URL (GCS or server)
 * 3. Cache the fetched audio for future use
 *
 * Cache keys are based on the audio URL, supporting both:
 * - GCS URLs (https://storage.googleapis.com/...)
 * - Local server URLs (/download/...)
 */

const DB_NAME = 'doseedo-audio-cache';
const DB_VERSION = 1;
const STORE_NAME = 'audio-files';
const MAX_CACHE_SIZE_MB = 500; // Maximum cache size in MB
const MAX_CACHE_AGE_DAYS = 7; // Maximum age of cached files

let db = null;

/**
 * Initialize the IndexedDB database
 */
async function initDB() {
  if (db) return db;

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      console.error('Failed to open audio cache database');
      reject(request.error);
    };

    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };

    request.onupgradeneeded = (event) => {
      const database = event.target.result;

      if (!database.objectStoreNames.contains(STORE_NAME)) {
        const store = database.createObjectStore(STORE_NAME, { keyPath: 'url' });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('size', 'size', { unique: false });
        console.log('Audio cache store created');
      }
    };
  });
}

/**
 * Generate a cache key from a URL
 * Normalizes URLs to handle both relative and absolute paths
 */
function getCacheKey(url) {
  // For GCS URLs, use the full URL
  if (url.startsWith('https://storage.googleapis.com/')) {
    return url;
  }

  // For relative URLs, normalize to full path
  if (url.startsWith('/')) {
    return `${window.location.origin}${url}`;
  }

  return url;
}

/**
 * Get cached audio blob
 * @param {string} url - Audio URL
 * @returns {Promise<Blob|null>} Cached blob or null
 */
export async function getCachedAudio(url) {
  try {
    await initDB();
    const cacheKey = getCacheKey(url);

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(cacheKey);

      request.onsuccess = () => {
        const result = request.result;
        if (result && result.blob) {
          // Check if cache is still valid
          const ageMs = Date.now() - result.timestamp;
          const maxAgeMs = MAX_CACHE_AGE_DAYS * 24 * 60 * 60 * 1000;

          if (ageMs < maxAgeMs) {
            resolve(result.blob);
          } else {
            console.log(`Cache expired for: ${url.substring(0, 50)}...`);
            resolve(null);
          }
        } else {
          resolve(null);
        }
      };

      request.onerror = () => {
        console.error('Error reading from cache:', request.error);
        resolve(null);
      };
    });
  } catch (error) {
    console.error('Cache read error:', error);
    return null;
  }
}

/**
 * Cache an audio blob
 * @param {string} url - Audio URL
 * @param {Blob} blob - Audio blob to cache
 */
export async function cacheAudio(url, blob) {
  try {
    await initDB();
    const cacheKey = getCacheKey(url);

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);

      const entry = {
        url: cacheKey,
        blob: blob,
        size: blob.size,
        timestamp: Date.now(),
        mimeType: blob.type
      };

      const request = store.put(entry);

      request.onsuccess = () => {
        resolve(true);
      };

      request.onerror = async () => {
        const err = request.error;
        console.error('Error caching audio:', err);
        // Storage quota is the common non-transient failure here.
        // Record it as a named product event so we see it in Sentry
        // without relying on a generic console-error hook.
        if (err && (err.name === 'QuotaExceededError' || /quota/i.test(String(err)))) {
          try {
            const { trackEvent, PRODUCT_EVENTS } = await import('../lib/telemetry');
            let quota_bytes = null;
            try {
              if (navigator?.storage?.estimate) {
                const est = await navigator.storage.estimate();
                quota_bytes = est?.quota ?? null;
              }
            } catch (_) {}
            trackEvent(PRODUCT_EVENTS.OPFS_QUOTA_EXCEEDED, {
              requested_bytes: blob?.size ?? null,
              quota_bytes,
            });
          } catch (_) { /* best-effort */ }
        }
        resolve(false);
      };
    });
  } catch (error) {
    console.error('Cache write error:', error);
    return false;
  }
}

// In-flight Promise registry: when N components ask for the same URL
// concurrently we share ONE fetch instead of hitting the network N times.
const _inflight = new Map();

/**
 * Fetch audio with caching
 * Checks cache first, fetches and caches if not found
 * @param {string} url - Audio URL
 * @returns {Promise<{blob: Blob, objectUrl: string, fromCache: boolean}>}
 */
export async function fetchAudioWithCache(url) {
  // Try cache first
  const cachedBlob = await getCachedAudio(url);

  if (cachedBlob) {
    const objectUrl = URL.createObjectURL(cachedBlob);
    return { blob: cachedBlob, objectUrl, fromCache: true };
  }

  // Dedupe concurrent requests for the same URL
  if (_inflight.has(url)) {
    return _inflight.get(url);
  }

  // Fetch from network

  const promise = (async () => {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch audio: ${response.status}`);
      }
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);

      // Cache in background (don't await)
      cacheAudio(url, blob).catch(err => {
        console.warn('Failed to cache audio:', err);
      });

      return { blob, objectUrl, fromCache: false };
    } catch (error) {
      console.error('Audio fetch error:', error);
      throw error;
    } finally {
      // Always remove the in-flight entry, even on error, so retries work
      _inflight.delete(url);
    }
  })();
  _inflight.set(url, promise);
  return promise;
}

/**
 * Clear expired cache entries
 */
export async function cleanupCache() {
  try {
    await initDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const index = store.index('timestamp');
      const maxAgeMs = MAX_CACHE_AGE_DAYS * 24 * 60 * 60 * 1000;
      const cutoffTime = Date.now() - maxAgeMs;

      const range = IDBKeyRange.upperBound(cutoffTime);
      const request = index.openCursor(range);
      let deletedCount = 0;

      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          store.delete(cursor.primaryKey);
          deletedCount++;
          cursor.continue();
        } else {
          if (deletedCount > 0) {
            console.log(`Cleaned up ${deletedCount} expired cache entries`);
          }
          resolve(deletedCount);
        }
      };

      request.onerror = () => {
        console.error('Error cleaning cache:', request.error);
        resolve(0);
      };
    });
  } catch (error) {
    console.error('Cache cleanup error:', error);
    return 0;
  }
}

/**
 * Get cache statistics
 */
export async function getCacheStats() {
  try {
    await initDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => {
        const entries = request.result || [];
        const totalSize = entries.reduce((sum, entry) => sum + (entry.size || 0), 0);

        resolve({
          count: entries.length,
          totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
          oldestTimestamp: entries.length > 0
            ? Math.min(...entries.map(e => e.timestamp))
            : null
        });
      };

      request.onerror = () => {
        resolve({ count: 0, totalSizeMB: 0, oldestTimestamp: null });
      };
    });
  } catch (error) {
    console.error('Error getting cache stats:', error);
    return { count: 0, totalSizeMB: 0, oldestTimestamp: null };
  }
}

/**
 * Clear all cached audio
 */
export async function clearCache() {
  try {
    await initDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.clear();

      request.onsuccess = () => {
        console.log('Audio cache cleared');
        resolve(true);
      };

      request.onerror = () => {
        console.error('Error clearing cache:', request.error);
        resolve(false);
      };
    });
  } catch (error) {
    console.error('Cache clear error:', error);
    return false;
  }
}

/**
 * Remove specific URL from cache
 */
export async function removeCachedAudio(url) {
  try {
    await initDB();
    const cacheKey = getCacheKey(url);

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(cacheKey);

      request.onsuccess = () => {
        console.log(`Removed from cache: ${url.substring(0, 50)}...`);
        resolve(true);
      };

      request.onerror = () => {
        console.error('Error removing from cache:', request.error);
        resolve(false);
      };
    });
  } catch (error) {
    console.error('Cache remove error:', error);
    return false;
  }
}

// Run cleanup on module load
if (typeof window !== 'undefined') {
  // Delay cleanup to not block initial load
  setTimeout(() => {
    cleanupCache().catch(console.error);
  }, 5000);
}
