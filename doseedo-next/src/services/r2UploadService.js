/**
 * R2 Upload Service
 *
 * POSTs to the Fly auth-service `/api/upload/r2` endpoint. Storage backend
 * is Cloudflare R2 (see auth-service/app/storage.py). The historical name
 * was `gcsUploadService` and the path `/api/upload/gcs` — both aliases are
 * still honored by the backend for one release so stale tabs don't break.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';

/**
 * Ask Clerk for a fresh JWT. Called per-request so a rotated Clerk session
 * (dev→prod swap, sign-out+sign-in) never sends a stale token. Returns
 * null for guests — caller should treat that as "don't attach Authorization".
 */
async function getAuthToken() {
  if (typeof window === 'undefined') return null;
  if (typeof window.__clerkGetToken === 'function') {
    try { return await window.__clerkGetToken(); }
    catch { /* fall through to legacy */ }
  }
  // Legacy JWT fallback for pre-Clerk users still in the same tab.
  return localStorage.getItem('token');
}

async function authHeaders(extra = {}) {
  const token = await getAuthToken();
  return {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  };
}

/**
 * Upload a file to R2.
 * @param {File} file
 * @param {string} contentType  session | loop | preset | midi | latent
 * @param {object} metadata
 * @returns {Promise<{success:true, url:string, path:string}>}
 */
export const uploadToR2 = async (file, contentType, metadata = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('content_type', contentType);
  formData.append('metadata', JSON.stringify(metadata));

  const response = await fetch(`${API_BASE}/api/upload/r2`, {
    method: 'POST',
    headers: await authHeaders(),
    body: formData,
  });

  const contentTypeHeader = response.headers.get('content-type') || '';
  if (contentTypeHeader.includes('text/html')) {
    throw new Error('Upload API endpoint not available — backend unreachable.');
  }
  if (!response.ok) {
    let msg = 'Upload failed';
    try { msg = (await response.json()).error || msg; } catch { /* non-json */ }
    throw new Error(msg);
  }

  const data = await response.json();
  return {
    success: true,
    url: data.url,
    path: data.path || data.gcs_path,
    fileName: file.name,
    fileSize: file.size,
  };
};

/**
 * Upload many files in parallel. One failure rejects the whole batch.
 */
export const uploadMultipleToR2 = async (files, contentType, metadata = {}) => {
  return Promise.all(files.map((f) => uploadToR2(f, contentType, metadata)));
};

export const deleteFromR2 = async (path) => {
  const response = await fetch(`${API_BASE}/api/upload/r2`, {
    method: 'DELETE',
    headers: await authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ path }),
  });
  if (!response.ok) {
    let msg = 'Delete failed';
    try { msg = (await response.json()).error || msg; } catch { /* non-json */ }
    throw new Error(msg);
  }
  return true;
};

export const getSignedUrl = async (filename, contentType = 'application/octet-stream') => {
  const response = await fetch(`${API_BASE}/api/upload/r2/signed-url`, {
    method: 'POST',
    headers: await authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ filename, content_type: contentType }),
  });
  if (!response.ok) {
    let msg = 'Failed to get signed URL';
    try { msg = (await response.json()).error || msg; } catch { /* non-json */ }
    throw new Error(msg);
  }
  const data = await response.json();
  return data.upload_url;
};

// Back-compat aliases — keep callers that still import the GCS names working
// until they migrate. Safe to delete once no grep hit remains.
export const uploadToGCS = uploadToR2;
export const uploadMultipleToGCS = uploadMultipleToR2;
export const deleteFromGCS = deleteFromR2;

export default {
  uploadToR2,
  uploadMultipleToR2,
  deleteFromR2,
  getSignedUrl,
  uploadToGCS,
  uploadMultipleToGCS,
  deleteFromGCS,
};
