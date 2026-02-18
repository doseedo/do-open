/**
 * Login redirect script for React app
 * Handles authentication and redirects to React app
 */

// Check if already authenticated - but only redirect from login page
window.onload = function() {
  // Only check auth if we're on the login page
  if (window.location.pathname === '/login.html') {
    const isAuth = localStorage.getItem('isAuth') === 'true';
    const username = localStorage.getItem('username');

    if (isAuth && username) {
      // Already logged in, redirect to React app root
      window.location.href = '/';
    }
  }
};

// Override the existing loginUser and registerUser functions to redirect to React app
const originalLoginUser = window.loginUser;
const originalRegisterUser = window.registerUser;

// Wrap loginUser to redirect after successful login
if (typeof window.loginUser === 'function') {
  window.loginUser = function(googleProfile) {
    // If Google profile is provided, handle Google login
    if (googleProfile) {
      const email = googleProfile.getEmail();
      const googleId = googleProfile.getId();
      const picture = googleProfile.getPicture() || 'user.png';

      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', googleId);

      fetch('https://doseedo.com/token/', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          // Try to register if login fails
          return registerUserGoogle(googleProfile);
        }
        return response.json();
      })
      .then(data => {
        if (data) {
          const subscriptionStatus = data.subscription ? "Pro+" : "Free";
          localStorage.setItem('isAuth', 'true');
          localStorage.setItem('username', email);
          localStorage.setItem('ispro', subscriptionStatus);
          localStorage.setItem('userpic', picture);

          // Redirect to React app
          window.location.href = '/';
        }
      })
      .catch(error => {
        console.error('Login error:', error);
        document.querySelector('#signin-error-message').innerText = 'Login failed';
        document.querySelector('#signin-error-message').style.display = 'block';
      });
    } else {
      // Regular email/password login
      const usernameEl = document.getElementById('login-username');
      const passwordEl = document.getElementById('login-password');

      if (!usernameEl || !passwordEl) return;

      const username = usernameEl.value;
      const password = passwordEl.value;

      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);

      fetch('https://doseedo.com/token/', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          document.querySelector('#signin-error-message').innerText = 'Incorrect username or password';
          document.querySelector('#signin-error-message').style.display = 'block';
          throw new Error('Incorrect username or password');
        }
        return response.json();
      })
      .then(data => {
        const subscriptionStatus = data.subscription ? "Pro+" : "Free";
        localStorage.setItem('isAuth', 'true');
        localStorage.setItem('username', username);
        localStorage.setItem('ispro', subscriptionStatus);
        localStorage.setItem('userpic', 'user.png');

        // Redirect to React app
        window.location.href = '/';
      })
      .catch(error => {
        console.error('Login error:', error);
      });
    }
  };
}

function registerUserGoogle(googleProfile) {
  const email = googleProfile.getEmail();
  const googleId = googleProfile.getId();
  const picture = googleProfile.getPicture() || 'user.png';
  const name = googleProfile.getName();

  const formData = new FormData();
  formData.append('username', name || email.split('@')[0]);
  formData.append('email', email);
  formData.append('password', googleId);

  return fetch('https://doseedo.com/register/', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    const subscriptionStatus = data.subscription ? "Pro+" : "Free";
    localStorage.setItem('isAuth', 'true');
    localStorage.setItem('username', email);
    localStorage.setItem('ispro', subscriptionStatus);
    localStorage.setItem('userpic', picture);

    // Redirect to React app
    window.location.href = '/';
    return data;
  });
}

// Override registerUser function
if (typeof window.registerUser === 'function') {
  window.registerUser = function() {
    const usernameEl = document.getElementById('register-username');
    const emailEl = document.getElementById('register-email');
    const passwordEl = document.getElementById('register-password');

    if (!usernameEl || !emailEl || !passwordEl) return;

    const username = usernameEl.value;
    const email = emailEl.value;
    const password = passwordEl.value;

    const formData = new FormData();
    formData.append('username', username);
    formData.append('email', email);
    formData.append('password', password);

    fetch('https://doseedo.com/register/', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      const subscriptionStatus = data.subscription ? "Pro+" : "Free";
      localStorage.setItem('isAuth', 'true');
      localStorage.setItem('username', username);
      localStorage.setItem('ispro', subscriptionStatus);
      localStorage.setItem('userpic', 'user.png');

      // Redirect to React app
      window.location.href = '/';
    })
    .catch(error => {
      console.error('Registration error:', error);
      document.querySelector('#signup-error-message').innerText = 'Registration failed';
      document.querySelector('#signup-error-message').style.display = 'block';
    });
  };
}
