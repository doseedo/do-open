import React from 'react';
import ReactDOM from 'react-dom/client';
import { initSentry } from './lib/sentry';
import './styles/colors.css'; // Master color palette (legacy dark theme)
import './styles/theme-workbench.css'; // Master workbench theme — body.workbench-theme on /studio, /dashboard, /projects, ...
import './styles/theme-hifi-purple.css'; // Dev variant override (body.theme-hifi-purple on /studio)
import './styles/glass-theme-background.css'; // Glass theme backgrounds
import './styles/liquid-glass.css'; // Glass theme button & panel effects
import './assets/css/original-style5.css';
import './assets/css/App.css';
import App from './App';

// Sentry init BEFORE React mounts so early-boot errors are captured.
// No-ops if REACT_APP_SENTRY_DSN isn't set (local dev without DSN).
initSentry();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
