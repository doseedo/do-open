import React from 'react';
import styles from './Downloads.module.css';
import PageTopbar from '../Sidebar/PageTopbar';
import PageEyebrow from '../Sidebar/PageEyebrow';

// Direct download URL for the latest macOS desktop release. Points at
// the canonical GitHub Releases "latest" alias so the page never needs
// to be republished when a new build ships — drop the .dmg under that
// asset name on the next release and this link picks it up.
//
// If the artifact ships under a different name (e.g. signed/notarized
// build with a version suffix), update DESKTOP_MAC_DMG_URL to point at
// the actual asset URL.
const DESKTOP_MAC_DMG_URL =
  'https://github.com/doseedo/Do/releases/latest/download/Doseedo-mac.dmg';

const downloads = [
  {
    id: 'macos-desktop',
    title: 'Doseedo for macOS',
    date: 'Latest release',
    description:
      'Native desktop build with offline rendering, Audio Unit host, and a standalone local stem engine.',
    icon: 'fa-apple',
    brand: true,
    color: 'rgba(232, 223, 200, 0.18)',
    href: DESKTOP_MAC_DMG_URL,
    external: true,
    status: 'Available',
  },
  {
    id: 'ios-controller',
    title: 'Doseedo Controller (iOS)',
    date: 'Coming soon',
    description:
      'Companion iOS app for remote transport, track arming, and MIDI-over-Wi-Fi to your Doseedo session.',
    icon: 'fa-mobile-screen',
    color: 'rgba(186, 156, 255, 0.16)',
    status: 'Coming soon',
    disabled: true,
  },
];

const Downloads = () => {
  const handleClick = (item) => {
    if (item.disabled) return;
    if (item.external && item.href) {
      window.open(item.href, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <div className={styles.downloads}>
      <PageTopbar section="Info" title="Downloads" />
      <PageEyebrow section="Downloads" description="Renders and exports from your sessions" />
      <div className={styles.header}>
        <h1 className={`${styles.title} page-title`}>Downloads</h1>
      </div>

      <div className={styles.sessionsContainer}>
        <div className={styles.sessionsList}>
          <div className={styles.sessionsHeader}>
            <div className={styles.headerNumber}>#</div>
            <div className={styles.headerTitle}>Title</div>
            <div className={styles.headerDate}>Channel</div>
            <div className={styles.headerStatus}>Status</div>
          </div>

          {downloads.map((item, index) => (
            <div
              key={item.id}
              className={`${styles.sessionRow} ${item.disabled ? styles.disabled : ''}`}
              onClick={() => handleClick(item)}
            >
              <div className={styles.sessionNumber}>{index + 1}</div>
              <div className={styles.sessionTitle}>
                <div className={styles.paperIcon} style={{ background: item.color }}>
                  <i className={`${item.brand ? 'fa-brands' : 'fa-solid'} ${item.icon}`}></i>
                </div>
                <div className={styles.paperInfo}>
                  <div className={styles.sessionName}>{item.title}</div>
                  <div className={styles.paperDesc}>{item.description}</div>
                </div>
              </div>
              <div className={styles.sessionDate}>{item.date}</div>
              <div className={styles.sessionStatus}>
                <span
                  className={`${styles.statusBadge} ${
                    item.status === 'Available' ? styles.badgeBeta : styles.badgePlanned
                  }`}
                >
                  {item.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Downloads;
