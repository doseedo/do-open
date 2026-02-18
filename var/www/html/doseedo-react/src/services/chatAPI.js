/**
 * Chat API Service
 * Handles communication with the AI chatbot backend
 */

const CHAT_API_BASE_URL = '/_chat/api';

/**
 * Send a chat message to the AI assistant
 * @param {Object} payload - Chat request payload
 * @param {string} payload.system_prompt - System prompt for the AI
 * @param {Object} payload.daw_context - Current DAW session context
 * @param {string} payload.message - User message
 * @param {Array} payload.conversation_history - Previous messages
 * @returns {Promise<Object>} - AI response
 */
export async function sendChatMessage(payload) {
  try {
    console.log('🤖 Sending chat message to AI:', payload.message);

    const response = await fetch(`${CHAT_API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));

      // Handle different error formats
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

      if (errorData.detail) {
        // If detail is an array of validation errors, format them
        if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail
            .map(err => `${err.loc ? err.loc.join('.') + ': ' : ''}${err.msg}`)
            .join(', ');
        } else if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else {
          errorMessage = JSON.stringify(errorData.detail);
        }
      }

      throw new Error(errorMessage);
    }

    const data = await response.json();
    console.log('✅ Received AI response');
    return data;

  } catch (error) {
    console.error('❌ Chat API error:', error);
    throw error;
  }
}

/**
 * Generate an image using DALL-E
 * @param {Object} payload
 * @param {string} payload.prompt - Image description
 * @param {string} [payload.size] - Image size (default 1024x1024)
 * @returns {Promise<Object>} - { url, revised_prompt }
 */
export async function generateImage(payload) {
  const response = await fetch(`${CHAT_API_BASE_URL}/images/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Search stock images via Unsplash
 * @param {Object} payload
 * @param {string} payload.query - Search query
 * @param {number} [payload.per_page] - Results per page (default 12)
 * @returns {Promise<Object>} - { results: [{ id, url, thumb, description, author }] }
 */
export async function searchImages(payload) {
  const response = await fetch(`${CHAT_API_BASE_URL}/images/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Send a message to the Backend Coder (codegen) chatbot
 * @param {Object} payload
 * @param {string} payload.system_prompt - System prompt for the DSP coder
 * @param {string} payload.message - User message
 * @param {Array} payload.conversation_history - Previous messages
 * @param {Object} [payload.ui_context] - Current UI state summary
 * @returns {Promise<Object>} - AI response
 */
export async function sendCodegenChatMessage(payload) {
  const response = await fetch(`${CHAT_API_BASE_URL}/codegen/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Generate JUCE C++ code from a DSP Language config
 * @param {Object} dspConfig - DSP Language JSON object
 * @returns {Promise<Object>} - { files: { "PluginProcessor.h": "...", ... } }
 */
export async function generatePluginCode(dspConfig) {
  const response = await fetch(`${CHAT_API_BASE_URL}/codegen/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dspConfig),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Check health of chat service
 * @returns {Promise<Object>} - Health status
 */
export async function checkChatHealth() {
  try {
    const response = await fetch(`${CHAT_API_BASE_URL}/chat/health`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('❌ Chat health check failed:', error);
    throw error;
  }
}
