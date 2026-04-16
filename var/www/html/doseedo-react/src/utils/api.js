import axios from 'axios';

/**
 * API utility functions for communicating with the backend
 * Based on the API calls from doseedo2.html
 */

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

/**
 * Generate audio from the given parameters
 * @param {FormData} formData - Form data with file and generation parameters
 * @returns {Promise} - Response with generated audio or error
 */
export async function generateAudio(formData) {
  try {
    const response = await axios.post(`${API_BASE_URL}/generate-risers`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 minutes timeout
    });
    return response.data;
  } catch (error) {
    console.error('Generation error:', error);
    throw error;
  }
}

/**
 * Upload and process audio file
 * @param {File} file - Audio or MIDI file
 * @returns {Promise} - Processed file info
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
}

/**
 * Create generation payload from app state
 * @param {Object} state - Application state
 * @param {File} file - Optional uploaded file
 * @returns {FormData} - Formatted form data for API
 */
export function createGenerationPayload(state, file = null) {
  const formData = new FormData();

  // Add file if provided
  if (file) {
    formData.append('file', file);
  }

  // Add generation parameters
  const params = state.generationParams;
  Object.keys(params).forEach(key => {
    formData.append(key, params[key]);
  });

  // Add automation data if present
  if (state.automationWindow.points.length > 0) {
    formData.append('automationData', JSON.stringify({
      points: state.automationWindow.points,
      resolution: state.automationWindow.resolution
    }));
  }

  return formData;
}
