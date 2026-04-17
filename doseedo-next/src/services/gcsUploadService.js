/**
 * GCS Upload Service
 * Handles file uploads to Google Cloud Storage via backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';

/**
 * Upload a file to GCS bucket
 * @param {File} file - The file to upload
 * @param {string} contentType - Type of content (session, loop, preset, midi)
 * @param {object} metadata - Additional metadata for the file
 * @returns {Promise<object>} Upload result with GCS URL
 */
export const uploadToGCS = async (file, contentType, metadata = {}) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('content_type', contentType);
    formData.append('metadata', JSON.stringify(metadata));

    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE}/api/upload/gcs`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      },
      body: formData
    });

    // Check content type to detect HTML error pages
    const contentTypeHeader = response.headers.get('content-type');
    if (contentTypeHeader && contentTypeHeader.includes('text/html')) {
      throw new Error('Upload API endpoint not available. Backend server may not be running.');
    }

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Upload failed');
    }

    const data = await response.json();
    return {
      success: true,
      gcsUrl: data.gcs_url,
      fileName: data.file_name,
      fileSize: data.file_size,
      uploadId: data.upload_id
    };
  } catch (error) {
    // Only log in development mode
    if (process.env.NODE_ENV === 'development') {
      console.warn('GCS upload error:', error.message);
    }
    throw error;
  }
};

/**
 * Upload multiple files to GCS
 * @param {File[]} files - Array of files to upload
 * @param {string} contentType - Type of content
 * @param {object} metadata - Shared metadata
 * @returns {Promise<object[]>} Array of upload results
 */
export const uploadMultipleToGCS = async (files, contentType, metadata = {}) => {
  try {
    const uploadPromises = files.map(file =>
      uploadToGCS(file, contentType, metadata)
    );
    return await Promise.all(uploadPromises);
  } catch (error) {
    // Only log in development mode
    if (process.env.NODE_ENV === 'development') {
      console.warn('Multiple GCS upload error:', error.message);
    }
    throw error;
  }
};

/**
 * Delete a file from GCS
 * @param {string} gcsUrl - The GCS URL of the file to delete
 * @returns {Promise<boolean>} Success status
 */
export const deleteFromGCS = async (gcsUrl) => {
  try {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE}/api/upload/gcs`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ gcs_url: gcsUrl })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Delete failed');
    }

    return true;
  } catch (error) {
    console.error('GCS delete error:', error);
    throw error;
  }
};

/**
 * Get signed URL for temporary access to GCS file
 * @param {string} gcsUrl - The GCS URL
 * @param {number} expiresIn - Expiration time in seconds (default: 3600)
 * @returns {Promise<string>} Signed URL
 */
export const getSignedUrl = async (gcsUrl, expiresIn = 3600) => {
  try {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE}/api/upload/gcs/signed-url`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        gcs_url: gcsUrl,
        expires_in: expiresIn
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to get signed URL');
    }

    const data = await response.json();
    return data.signed_url;
  } catch (error) {
    console.error('Get signed URL error:', error);
    throw error;
  }
};

export default {
  uploadToGCS,
  uploadMultipleToGCS,
  deleteFromGCS,
  getSignedUrl
};
