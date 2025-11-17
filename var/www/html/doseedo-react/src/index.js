import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/colors.css'; // Master color palette
import './styles/glass-theme-background.css'; // Glass theme backgrounds
import './assets/css/original-style5.css';
import './assets/css/App.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
