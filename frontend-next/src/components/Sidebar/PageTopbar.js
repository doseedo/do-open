import React from 'react';

/**
 * PageTopbar — shared breadcrumb strip for every dashboard subpage.
 *
 * Mirrors the visual rule already used on Home (Dashboard.module.css
 * `.topbar`) and Tools (inline-styled in Tools.js):
 *
 *   Dashboard  /  Create  /  <title>  ·  <meta>      <right side>
 *
 * Right side defaults to the canonical "engine · ready · sync · up to
 * date · ⌘K" status strip. Pass `right` to override.
 *
 * Drop-in usage: render as the FIRST child inside an existing dashboard
 * subpage wrapper. Negative margins cancel the wrapper's padding so the
 * topbar visually sits flush with the page edges (matching the look of
 * Home + Tools), while the wrapper's padding-top: 100 keeps the page
 * title's y baseline at the canonical position underneath.
 */
export default function PageTopbar({
  section = null,
  title,
  meta = null,
  right = null,
}) {
  const styles = {
    bar: {
      // Negative margins cancel the typical dashboard wrapper padding
      // (40px sides, 100px top) so the strip spans full width and sits
      // flush at y=0 of the page chrome.
      margin: '-100px -40px 32px -40px',
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      padding: '0 36px',
      height: 44,
      borderBottom: '1px solid var(--wb-rule, rgba(255,255,255,0.08))',
      background: 'var(--wb-surface, rgba(255,255,255,0.02))',
      fontFamily: 'var(--wb-font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)',
      fontSize: 10,
      letterSpacing: 0.6,
      textTransform: 'uppercase',
      color: 'var(--wb-ink-mute, rgba(255,255,255,0.5))',
      flexShrink: 0,
      flexWrap: 'wrap',
    },
    sep: { color: 'var(--wb-ink-faint, rgba(255,255,255,0.25))' },
    strong: { color: 'var(--wb-ink-soft, rgba(255,255,255,0.78))', fontWeight: 500 },
    spacer: { flex: 1 },
    kbd: {
      padding: '2px 6px',
      border: '1px solid var(--wb-rule, rgba(255,255,255,0.08))',
      borderRadius: 2,
      fontFamily: 'var(--wb-font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)',
      fontSize: 9,
      color: 'var(--wb-ink-soft, rgba(255,255,255,0.78))',
      letterSpacing: 0.3,
      textTransform: 'none',
    },
  };

  const defaultRight = (
    <>
      <span>engine · <strong style={styles.strong}>ready</strong></span>
      <span style={styles.sep}>·</span>
      <span>sync · <strong style={styles.strong}>up to date</strong></span>
      <span style={styles.sep}>·</span>
      <span style={styles.kbd}>⌘K</span>
    </>
  );

  return (
    <div style={styles.bar}>
      <span>Dashboard</span>
      {section && (
        <>
          <span style={styles.sep}>/</span>
          <span>{section}</span>
        </>
      )}
      <span style={styles.sep}>/</span>
      <span><strong style={styles.strong}>{title}</strong></span>
      {meta && (
        <>
          <span style={styles.sep}>·</span>
          <span>{meta}</span>
        </>
      )}
      <div style={styles.spacer} />
      {right ?? defaultRight}
    </div>
  );
}
