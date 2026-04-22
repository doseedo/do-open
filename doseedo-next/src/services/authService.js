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

  // Track sign-up in Google Analytics
  if (window.gtag) {
    window.gtag('event', 'sign_up', { method: 'email' });
  }

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

  // Track sign-up in Google Analytics
  if (window.gtag) {
    window.gtag('event', 'sign_up', { method: 'google' });
  }

  return {
    username: email,
    subscriptionStatus,
    isPro: data.subscription || false,
    picture
  };
}

/**
 * Claim a free plugin with just an email (no password).
 * Creates or finds user, sets auth cookies, creates purchase record.
 * @param {string} email - User's email
 * @param {string} pluginSlug - Plugin slug to claim
 * @returns {Promise<Object>} { download_url, plugin_name, username, is_new_user }
 */
export async function liteClaimPlugin(email, pluginSlug) {
  const response = await fetch(`${API_BASE_URL}/api/plugins/lite-claim`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, plugin_slug: pluginSlug }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to claim plugin');
  }

  const data = await response.json();

  if (window.gtag) {
    window.gtag('event', 'sign_up', { method: 'lite_email' });
  }

  return data;
}

/**
 * Logout current user. Clerk is the live auth system; we also expire the
 * legacy auth cookies (username/ispro/userpic/auth_token) that the
 * pre-Clerk backend used to set, otherwise `getCurrentUser` will re-read
 * them on reload and the UI insists the user is still signed in.
 */
export async function logoutUser() {
  // 1. Sign out of Clerk. `window.Clerk` is exposed by ClerkProvider in the
  //    Next shell; calling signOut here gives us the same behaviour as
  //    useClerk().signOut without having to hook-ify every caller.
  try {
    if (typeof window !== 'undefined' && window.Clerk?.signOut) {
      await window.Clerk.signOut();
    }
  } catch (error) {
    console.error('Clerk signOut failed:', error);
  }

  // 2. Expire legacy cookies on every plausible scope (host + apex + sub).
  //    Some old cookies were set with Domain=.doseedo.com, others without.
  if (typeof document !== 'undefined') {
    const past = 'Thu, 01 Jan 1970 00:00:00 GMT';
    for (const name of ['username', 'ispro', 'userpic', 'auth_token', 'access_token', 'clerk_token']) {
      document.cookie = `${name}=; expires=${past}; path=/`;
      document.cookie = `${name}=; expires=${past}; path=/; domain=.doseedo.com`;
      document.cookie = `${name}=; expires=${past}; path=/; domain=doseedo.com`;
    }
  }

  // 3. Drop any cached Clerk JWT snapshot from older ClerkTokenBridge
  //    builds + legacy `token` key. Current builds don't write these, but
  //    older tabs may still have them.
  try { window.localStorage?.removeItem('clerk_token'); } catch {}
  try { window.localStorage?.removeItem('token'); } catch {}
}

/**
 * Check if user is authenticated. Clerk is the source of truth when it's
 * loaded; the legacy-cookie check only runs when Clerk hasn't mounted yet.
 * Trusting the cookie first would flash the stale profile on logout for a
 * paint — which is the "logout shows same user" bug.
 */
export function isAuthenticated() {
  if (typeof window !== 'undefined' && window.Clerk) {
    if (window.Clerk.user) return true;
    if (window.Clerk.loaded) return false; // Clerk says no — don't fall back.
  }
  return getCookie('username') !== null;
}

/**
 * Get current user data. Clerk first; legacy cookies only as a pre-mount
 * fallback.
 */
export function getCurrentUser() {
  if (typeof window !== 'undefined' && window.Clerk?.user) {
    const u = window.Clerk.user;
    const primary = u.primaryEmailAddress?.emailAddress || '';
    const username = u.username || primary.split('@')[0] || 'user';
    return {
      id: u.id,
      username,
      email: primary,
      subscriptionStatus: 'Free',
      isPro: false,
      picture: u.imageUrl || 'user.png',
    };
  }
  if (typeof window !== 'undefined' && window.Clerk?.loaded && !window.Clerk.user) {
    return null;
  }
  if (!isAuthenticated()) return null;

  const username = getCookie('username');
  const ispro = getCookie('ispro');
  const userpic = getCookie('userpic') || 'user.png';

  return {
    id: username,
    username,
    subscriptionStatus: ispro || 'Free',
    isPro: ispro === 'Pro+',
    picture: userpic,
  };
}
