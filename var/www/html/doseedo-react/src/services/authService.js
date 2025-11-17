/**
 * Authentication Service
 * Handles user registration, login, logout, and session management
 * Uses HTTP-only cookies for secure authentication (production-ready)
 */

const API_BASE_URL = 'https://doseedo.com';

/**
 * Helper function to get a cookie value by name
 * @param {string} name - Cookie name
 * @returns {string|null} Cookie value or null if not found
 */
function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return null;
}

/**
 * Register a new user
 * @param {string} username
 * @param {string} email
 * @param {string} password
 * @returns {Promise<Object>} User data with subscription status
 */
export async function registerUser(username, email, password) {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('email', email);
  formData.append('password', password);

  const response = await fetch(`${API_BASE_URL}/register/`, {
    method: 'POST',
    credentials: 'include', // Include cookies in request
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || 'Registration failed');
  }

  const data = await response.json();

  // Backend sets cookies automatically (auth_token, username, ispro)
  // No need to use localStorage anymore!

  const subscriptionStatus = data.subscription ? "Pro+" : "Free";

  return {
    username,
    subscriptionStatus,
    isPro: data.subscription || false,
    picture: 'user.png'
  };
}

/**
 * Login user with email/password
 * @param {string} username - Email or username
 * @param {string} password
 * @returns {Promise<Object>} User data with subscription status
 */
export async function loginUser(username, password) {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);

  const response = await fetch(`${API_BASE_URL}/token/`, {
    method: 'POST',
    credentials: 'include', // Include cookies in request
    body: formData
  });

  if (!response.ok) {
    throw new Error('Incorrect username or password');
  }

  const data = await response.json();

  // Backend sets cookies automatically (auth_token, username, ispro)
  const subscriptionStatus = data.subscription ? "Pro+" : "Free";

  return {
    username,
    subscriptionStatus,
    isPro: data.subscription || false,
    picture: 'user.png'
  };
}

/**
 * Login user with Google OAuth
 * @param {Object} googleProfile - Google profile object with getEmail(), getId(), getPicture(), getName()
 * @returns {Promise<Object>} User data with subscription status
 */
export async function loginWithGoogle(googleProfile) {
  const email = googleProfile.getEmail();
  const googleId = googleProfile.getId();
  const picture = googleProfile.getPicture() || 'user.png';
  const name = googleProfile.getName();

  const formData = new FormData();
  formData.append('username', email);
  formData.append('password', googleId);

  try {
    const response = await fetch(`${API_BASE_URL}/token/`, {
      method: 'POST',
      credentials: 'include', // Include cookies
      body: formData
    });

    if (!response.ok) {
      // If login fails, try to register
      return await registerWithGoogle(googleProfile);
    }

    const data = await response.json();

    // Backend sets cookies automatically
    const subscriptionStatus = data.subscription ? "Pro+" : "Free";

    return {
      username: email,
      subscriptionStatus,
      isPro: data.subscription || false,
      picture
    };
  } catch (error) {
    // Fallback to registration if login fails
    return await registerWithGoogle(googleProfile);
  }
}

/**
 * Register user with Google OAuth
 * @param {Object} googleProfile - Google profile object
 * @returns {Promise<Object>} User data
 */
async function registerWithGoogle(googleProfile) {
  const email = googleProfile.getEmail();
  const googleId = googleProfile.getId();
  const picture = googleProfile.getPicture() || 'user.png';
  const name = googleProfile.getName();

  const formData = new FormData();
  formData.append('username', name || email.split('@')[0]);
  formData.append('email', email);
  formData.append('password', googleId);

  const response = await fetch(`${API_BASE_URL}/register/`, {
    method: 'POST',
    credentials: 'include', // Include cookies
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || 'Google registration failed');
  }

  const data = await response.json();

  // Backend sets cookies automatically
  const subscriptionStatus = data.subscription ? "Pro+" : "Free";

  return {
    username: email,
    subscriptionStatus,
    isPro: data.subscription || false,
    picture
  };
}

/**
 * Logout current user by calling backend to clear cookies
 */
export async function logoutUser() {
  try {
    await fetch(`${API_BASE_URL}/logout`, {
      method: 'POST',
      credentials: 'include' // Include cookies
    });
  } catch (error) {
    console.error('Logout failed:', error);
  }
  // Cookies are cleared by backend
}

/**
 * Check if user is authenticated by checking for auth_token cookie
 * @returns {boolean}
 */
export function isAuthenticated() {
  // Check if username cookie exists (auth_token is HTTP-only so we can't read it)
  return getCookie('username') !== null;
}

/**
 * Get current user data from cookies
 * @returns {Object|null} User data or null if not authenticated
 */
export function getCurrentUser() {
  if (!isAuthenticated()) {
    return null;
  }

  const username = getCookie('username');
  const ispro = getCookie('ispro');
  const userpic = getCookie('userpic') || 'user.png';

  return {
    id: username, // Use username as id for cloud save operations
    username,
    subscriptionStatus: ispro || 'Free',
    isPro: ispro === 'Pro+',
    picture: userpic
  };
}
