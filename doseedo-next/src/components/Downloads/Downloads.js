import React from 'react';
import styles from './Downloads.module.css';
import PageTopbar from '../Sidebar/PageTopbar';
import PageEyebrow from '../Sidebar/PageEyebrow';

const downloads = [
  {
    id: 'ios-controller',
    title: 'Doseedo Controller (iOS)',
    date: 'TestFlight',
    description: 'Companion iOS app for remote transport, track arming, and MIDI-over-Wi-Fi to your Doseedo session.',
    icon: 'fa-mobile-screen',
    color: 'rgba(186, 156, 255, 0.2)',
    href: 'https://testflight.apple.com/join/doseedo-controller',
    external: true,
    status: 'Beta',
  },
  {
    id: 'macos-desktop',
    title: 'Doseedo for macOS',
    date: 'Coming soon',
    description: 'Native desktop build with offline rendering, Audio Unit host, and a standalone local stem engine.',
    icon: 'fa-display',
    color: 'rgba(102, 126, 234, 0.2)',
    status: 'Planned',
    disabled: true,
  },
  {
    id: 'windows-desktop',
    title: 'Doseedo for Windows',
    date: 'Coming soon',
    description: 'Native Windows build with VST3 host and local GPU-accelerated stem separation.',
    icon: 'fa-windows',
    color: 'rgba(88, 166, 255, 0.2)',
    status: 'Planned',
    disabled: true,
    brand: true,
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
                    item.status === 'Beta' ? styles.badgeBeta : styles.badgePlanned
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
