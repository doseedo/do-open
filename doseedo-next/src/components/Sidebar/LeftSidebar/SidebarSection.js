import React from 'react';
import styles from './LeftSidebar.module.css';

/**
 * SidebarSection Component
 * Reusable sidebar section with optional title and divider
 *
 * @param {string} title - Section title
 * @param {React.ReactNode} children - Section content
 * @param {boolean} showDivider - Whether to show divider before section
 */
const SidebarSection = React.memo(({ title, children, showDivider }) => {
  return (
    <>
      {showDivider && <hr className={styles.divider} />}
      {title && <p className={styles.sectionTitle}>{title}</p>}
      {children}
    </>
  );
});

SidebarSection.displayName = 'SidebarSection';

export default SidebarSection;
