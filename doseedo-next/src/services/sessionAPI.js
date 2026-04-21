/**
 * Session Management API
 * Handles user sessions, uploads, and content management
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';

/**
 * Create a new session
 * @param {object} sessionData - Session data
 * @returns {Promise<object>} Created session
 */
export const createSession = async (sessionData) => {
  try {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE}/api/sessions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(sessionData)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to create session');
    }

    return await response.json();
  } catch (error) {
    console.error('Create session error:', error);
    throw error;
  }
};

/**
 * Get all user sessions
 * @param {object} filters - Optional filters (type, search, etc.)
 * @returns {Promise<object[]>} Array of sessions
 */
export const getUserSessions = async (filters = {}) => {
  try {
    const token = localStorage.getItem('token');
    const queryParams = new URLSearchParams(filters).toString();
    const url = `${API_BASE}/api/sessions${queryParams ? `?${queryParams}` : ''}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    // Check content type to detect HTML error pages
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('text/html')) {
      throw new Error('API endpoint not available. Backend server may not be running.');
    }

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to fetch sessions');
    }

    return await response.json();
  } catch (error) {
    // Only log in development mode - production should fail silently
    if (process.env.NODE_ENV === 'development') {
      console.warn('Get sessions error:', error.message);
    }
    throw error;
  }
};

/**
 * Get a specific session by ID
 * @param {string} sessionId - Session ID
 * @returns {Promise<object>} Session data
 */
export const getSession = async (sessionId) => {
  try {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to fetch session');
    }

    return await response.json();
  } catch (error) {
    console.error('Get session error:', error);
    throw error;
  }
};

/**
 * Update a session
 * @param {string} sessionId - Session ID
 * @param {object} updates - Session updates
 * @returns {Promise<object>} Updated session
 */
export const updateSession = async (sessionId, updates) => {
  try {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(updates)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to update session');
    }

    return await response.json();
  } catch (error) {
    console.error('Update session error:', error);
    throw error;
  }
};

/**
 * Delete a session
 * @param {string} sessionId - Session ID
 * @returns {Promise<boolean>} Success status
 */
export const deleteSession = async (sessionId) => {
  try {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to delete session');
    }

    return true;
  } catch (error) {
    console.error('Delete session error:', error);
    throw error;
  }
};

/**
 * Upload content to a session
 * @param {string} sessionId - Session ID
 * @param {object} contentData - Content data (files, metadata)
 * @returns {Promise<object>} Upload result
 */
export const uploadSessionContent = async (sessionId, contentData) => {
  try {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(contentData)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to upload content');
    }

    return await response.json();
  } catch (error) {
    console.error('Upload session content error:', error);
    throw error;
  }
};

/**
 * Search sessions
 * @param {string} query - Search query
 * @param {object} filters - Additional filters
 * @returns {Promise<object[]>} Search results
 */
export const searchSessions = async (query, filters = {}) => {
  try {
    const token = localStorage.getItem('token');
    const params = new URLSearchParams({
      q: query,
      ...filters
    }).toString();

    const response = await fetch(`${API_BASE}/api/sessions/search?${params}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Search failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Search sessions error:', error);
    throw error;
  }
};

/**
 * Get public sessions (for search/browse)
 * @param {object} filters - Filters (category, sort, etc.)
 * @returns {Promise<object[]>} Public sessions
 */
export const getPublicSessions = async (filters = {}) => {
  try {
    const queryParams = new URLSearchParams(filters).toString();
    const url = `${API_BASE}/api/sessions/public${queryParams ? `?${queryParams}` : ''}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to fetch public sessions');
    }

    return await response.json();
  } catch (error) {
    console.error('Get public sessions error:', error);
    throw error;
  }
};

export default {
  createSession,
  getUserSessions,
  getSession,
  updateSession,
  deleteSession,
  uploadSessionContent,
  searchSessions,
  getPublicSessions
};
