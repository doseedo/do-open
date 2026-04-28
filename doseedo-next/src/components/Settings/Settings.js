import React from 'react';
import { useNavigate } from 'react-router-dom';
import { logoutUser, getCurrentUser } from '../../services/authService';
import styles from './Settings.module.css';

/**
 * Settings — workbench-themed top-level settings page at /settings.
 *
 * Acts as a hub: lists each settings section with a short description and
 * a link to its dedicated page. Profile editing lives at /profile,
 * API keys at /settings/api-keys (proxied to the Fly auth-service),
 * billing at /plans.
 */

const SECTIONS = [
  {
    id: 'profile',
    title: 'Profile',
    desc: 'Display name, bio, avatar, and public profile links.',
    href: '/profile',
    cta: 'Edit profile',
    internal: true,
  },
  {
    id: 'api-keys',
    title: 'API Keys',
    desc: 'Generate and revoke keys for the doo desktop assistant and programmatic access.',
    href: '/settings/api-keys',
    cta: 'Manage keys',
    internal: false, // proxied to Fly via Vercel rewrite — full page navigation
  },
  {
    id: 'plans',
    title: 'Plan & Billing',
    desc: 'View your current plan, usage, and billing history.',
    href: '/plans',
    cta: 'View plan',
    internal: true,
  },
];

export default function Settings() {
  const navigate = useNavigate();
  const user = getCurrentUser();

  const handleNavigate = (section) => {
    if (section.internal) {
      navigate(section.href);
    } else {
      window.location.href = section.href;
    }
  };

  const handleSignOut = async () => {
    try {
      await logoutUser();
    } finally {
      window.location.href = '/';
    }
  };

  return (
    <div className={styles.settings}>
      <header className={styles.head}>
        <div>
          <div className={styles.crumb}>
            <span onClick={() => navigate('/profile')} role="button" tabIndex={0}>Profile</span>
            <span className={styles.crumbSep}>›</span>
            <span>Settings</span>
          </div>
          <h1 className={styles.title}>Settings</h1>
          <p className={styles.sub}>
            Account, keys, billing — for {user?.username ? <b>@{user.username}</b> : 'your account'}.
          </p>
        </div>
      </header>

      <div className={styles.sections}>
        {SECTIONS.map((s) => (
          <div key={s.id} className={styles.card}>
            <div className={styles.cardBody}>
              <div className={styles.cardLabel}>{s.title}</div>
              <div className={styles.cardDesc}>{s.desc}</div>
            </div>
            <button
              className={styles.cardCta}
              onClick={() => handleNavigate(s)}
              type="button"
            >
              {s.cta}
              <span className={styles.cardArrow}>→</span>
            </button>
          </div>
        ))}
      </div>

      <div className={styles.danger}>
        <div className={styles.dangerLabel}>Session</div>
        <button
          className={styles.signOutBtn}
          onClick={handleSignOut}
          type="button"
        >
          Sign out
        </button>
      </div>
    </div>
  );
}
