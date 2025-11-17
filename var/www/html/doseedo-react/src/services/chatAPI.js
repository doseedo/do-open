/**
 * Chat API Service
 * Handles communication with the AI chatbot backend
 */

const CHAT_API_BASE_URL = 'http://localhost:8090/api';

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
