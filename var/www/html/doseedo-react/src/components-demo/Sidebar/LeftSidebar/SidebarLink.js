import React from 'react';
import styles from './LeftSidebar.module.css';

/**
 * SidebarLink Component
 * Reusable sidebar navigation link
 *
 * @param {string} href - Link URL
 * @param {string} icon - Font Awesome icon class
 * @param {string} label - Link text
 * @param {boolean} highlighted - Whether to show highlighted background
 * @param {boolean} active - Whether this link is currently active
 * @param {Function} onClick - Optional click handler
 */
const SidebarLink = React.memo(({ href, icon, label, highlighted, active, onClick }) => {
  const linkClass = highlighted ? styles.navLinkHighlighted : styles.navLink;

  return (
    <a
      href={href}
      className={linkClass}
      onClick={onClick}
      style={active ? {
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        color: 'white'
      } : undefined}
    >
      {icon && (
        <i
          className={icon}
          style={highlighted ? {
            backgroundColor: 'rgba(128, 128, 128, 0.21)',
            padding: '5px',
            borderRadius: '5px'
          } : undefined}
        />
      )}
      {label && <span className={styles.navLinkText}>{label}</span>}
    </a>
  );
});

SidebarLink.displayName = 'SidebarLink';

export default SidebarLink;
