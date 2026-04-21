// doseedo · Hi-Fi Purple theme
// Drop-in tokens + primitives for a React studio in the Y8p style.
// No external deps. Works in plain React or with any styling approach
// (inline styles, Emotion, Tailwind arbitrary values, CSS vars).
//
// USAGE
//   import { theme, styles, Button, Card, Rule, Label, Title, Meta } from './theme-hifi-purple';
//
//   <div style={{ background: theme.bg, color: theme.ink, fontFamily: theme.sans }}>
//     <Title>Untitled</Title>
//     <Meta>autosaved · 2m</Meta>
//     <Button>Upload a file</Button>
//     <Button variant="secondary">Record</Button>
//   </div>
//
// Or inject as CSS variables once at the root:
//   <style>{cssVars}</style>
//   then use var(--bg), var(--ink), etc.

// ------------------------------------------------------------------
// 1. TOKENS
// ------------------------------------------------------------------
export const theme = {
  // Surfaces — warm near-black charcoal
  bg:       '#1f1a15',
  surface:  '#2a241d',
  surface2: '#36302a',

  // Ink — warm cream with three alphas
  ink:      '#e8dfc8',
  inkSoft:  'rgba(232,223,200,0.68)',
  inkMute:  'rgba(232,223,200,0.42)',

  // Hairline rules on dark
  rule:       'rgba(232,223,200,0.10)',
  ruleStrong: 'rgba(232,223,200,0.24)',

  // Accent — doseedo purple
  accent:     '#a88adc',
  accentDeep: '#6a4e9e',
  accentSoft: 'rgba(168,138,220,0.16)',

  // Track / clip palette — for color-coding parts
  tracks: {
    lead:    '#a88adc',  // accent
    rhodes:  '#e8c88a',  // wheat
    bass:    '#8ac8a0',  // sage
    drums:   '#e07556',  // rust
    strings: '#6aa8e8',  // sky
  },

  // Typography
  serif: '"Newsreader", Georgia, serif',
  sans:  '"Inter", system-ui, sans-serif',
  mono:  '"JetBrains Mono", ui-monospace, monospace',

  // Radii
  radius: { sm: 3, md: 4, lg: 6 },

  // Spacing scale (px)
  space: { 1: 4, 2: 8, 3: 12, 4: 16, 5: 20, 6: 24, 8: 32, 10: 40 },
};

// Google Fonts import string — paste into <head> or a CSS @import
export const fontImport =
  '@import url("https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap");';

// ------------------------------------------------------------------
// 2. CSS VARIABLES (optional)
// ------------------------------------------------------------------
export const cssVars = `
${fontImport}
:root {
  --bg: ${theme.bg};
  --surface: ${theme.surface};
  --surface-2: ${theme.surface2};
  --ink: ${theme.ink};
  --ink-soft: ${theme.inkSoft};
  --ink-mute: ${theme.inkMute};
  --rule: ${theme.rule};
  --rule-strong: ${theme.ruleStrong};
  --accent: ${theme.accent};
  --accent-deep: ${theme.accentDeep};
  --accent-soft: ${theme.accentSoft};
  --font-serif: ${theme.serif};
  --font-sans: ${theme.sans};
  --font-mono: ${theme.mono};
}
body { background: var(--bg); color: var(--ink); font-family: var(--font-sans); }
`;

// ------------------------------------------------------------------
// 3. STYLE RECIPES (reusable style objects)
// ------------------------------------------------------------------
export const styles = {
  // Layout
  page: {
    background: theme.bg,
    color: theme.ink,
    fontFamily: theme.sans,
    minHeight: '100vh',
  },
  panel: {
    background: theme.surface,
    borderBottom: `1px solid ${theme.rule}`,
  },
  panelInset: {
    background: theme.surface2,
    border: `1px solid ${theme.rule}`,
    borderRadius: theme.radius.lg,
  },

  // Typography
  title: {
    fontFamily: theme.serif,
    fontWeight: 400,
    letterSpacing: -0.3,
    color: theme.ink,
  },
  titleItalic: {
    fontFamily: theme.serif,
    fontStyle: 'italic',
    fontWeight: 400,
    letterSpacing: -0.1,
    color: theme.ink,
  },
  body: {
    fontFamily: theme.sans,
    fontSize: 14,
    lineHeight: 1.55,
    color: theme.ink,
  },
  secondary: {
    fontFamily: theme.sans,
    fontSize: 13,
    color: theme.inkSoft,
  },
  meta: {
    fontFamily: theme.mono,
    fontSize: 11,
    color: theme.inkMute,
  },
  label: {
    fontFamily: theme.mono,
    fontSize: 11,
    color: theme.inkMute,
    letterSpacing: 0.4,
    textTransform: 'uppercase',
  },

  // Buttons
  buttonPrimary: {
    background: theme.accentDeep,
    color: '#fff',
    border: 'none',
    padding: '11px 22px',
    borderRadius: theme.radius.md,
    fontFamily: theme.sans,
    fontSize: 13,
    fontWeight: 500,
    cursor: 'pointer',
  },
  buttonSecondary: {
    background: 'transparent',
    color: theme.ink,
    border: `1px solid ${theme.ruleStrong}`,
    padding: '11px 18px',
    borderRadius: theme.radius.md,
    fontFamily: theme.sans,
    fontSize: 13,
    fontWeight: 500,
    cursor: 'pointer',
  },
  buttonGhost: {
    background: 'transparent',
    color: theme.ink,
    border: `1px solid ${theme.rule}`,
    padding: '6px 12px',
    borderRadius: theme.radius.md,
    fontFamily: theme.sans,
    fontSize: 12,
    cursor: 'pointer',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
  },
  iconButton: {
    width: 28, height: 28, borderRadius: '50%',
    border: `1px solid ${theme.rule}`,
    background: theme.surface2,
    display: 'grid', placeItems: 'center',
    cursor: 'pointer',
    color: theme.ink,
  },

  // Cards / tiles
  card: {
    background: theme.surface2,
    border: `1px solid ${theme.rule}`,
    borderRadius: theme.radius.lg,
    padding: '14px 12px',
    cursor: 'pointer',
  },
  cardSelected: {
    background: theme.surface2,
    border: `1px solid ${theme.accent}55`,
    borderRadius: theme.radius.lg,
    padding: '14px 12px',
    cursor: 'pointer',
  },

  // Avatar / brand dot
  brandDot: {
    width: 22, height: 22, borderRadius: '50%',
    background: theme.accent,
  },
  avatar: {
    width: 28, height: 28, borderRadius: '50%',
    background: theme.accentDeep,
    color: '#fff',
    fontFamily: theme.serif,
    fontSize: 13,
    display: 'grid', placeItems: 'center',
  },

  // Tab underline
  tabActive: {
    borderBottom: `1.5px solid ${theme.accent}`,
    color: theme.ink,
    fontWeight: 500,
  },
  tabInactive: {
    borderBottom: '1.5px solid transparent',
    color: theme.inkSoft,
    fontWeight: 400,
  },

  // Hairline divider
  hrVertical: { width: 1, background: theme.rule, alignSelf: 'stretch' },
  hrHorizontal: { height: 1, background: theme.rule, width: '100%' },

  // Clip (timeline region) — call clipStyle(trackColor) to get the object
  clip: (trackColor) => ({
    background: trackColor + '22',
    border: `1px solid ${trackColor}55`,
    borderRadius: theme.radius.sm,
    color: trackColor,
    padding: 4,
    boxSizing: 'border-box',
  }),

  // Zebra row for alternating timeline lanes
  rowEven: { background: theme.surface },
  rowOdd:  { background: theme.surface2 },

  // Playhead
  playhead: {
    position: 'absolute',
    top: 0, bottom: 0,
    width: 1,
    background: theme.accent,
  },
};

// ------------------------------------------------------------------
// 4. PRIMITIVE COMPONENTS (optional — thin wrappers over styles)
// ------------------------------------------------------------------
// Remove the `React` import if you already have one in scope.
import React from 'react';

export function Title({ size = 22, italic = false, children, style, ...rest }) {
  const base = italic ? styles.titleItalic : styles.title;
  return <div style={{ ...base, fontSize: size, ...style }} {...rest}>{children}</div>;
}

export function Meta({ children, style, ...rest }) {
  return <div style={{ ...styles.meta, ...style }} {...rest}>{children}</div>;
}

export function Label({ children, style, ...rest }) {
  return <div style={{ ...styles.label, ...style }} {...rest}>{children}</div>;
}

export function Button({ variant = 'primary', children, style, ...rest }) {
  const map = {
    primary: styles.buttonPrimary,
    secondary: styles.buttonSecondary,
    ghost: styles.buttonGhost,
  };
  return <button style={{ ...map[variant], ...style }} {...rest}>{children}</button>;
}

export function Card({ selected = false, children, style, ...rest }) {
  return (
    <div style={{ ...(selected ? styles.cardSelected : styles.card), ...style }} {...rest}>
      {children}
    </div>
  );
}

export function Rule({ vertical = false, style }) {
  return <div style={{ ...(vertical ? styles.hrVertical : styles.hrHorizontal), ...style }} />;
}

export function Panel({ children, style, ...rest }) {
  return <div style={{ ...styles.panel, ...style }} {...rest}>{children}</div>;
}

// Clip + timeline row helpers
export function Clip({ color, children, style, ...rest }) {
  return <div style={{ ...styles.clip(color), ...style }} {...rest}>{children}</div>;
}

export function TimelineRow({ index = 0, children, style, ...rest }) {
  return (
    <div style={{ ...(index % 2 === 0 ? styles.rowEven : styles.rowOdd), ...style }} {...rest}>
      {children}
    </div>
  );
}

// ------------------------------------------------------------------
// 5. GUIDELINES (read once)
// ------------------------------------------------------------------
// Accent sparingly. On any given screen, aim for < 5 purple elements total:
//   brand dot, active tab underline, playhead, record icon, selected-card border.
// Everything else uses ink + rule. Purple should feel like punctuation.
//
// Type hierarchy:
//   serif (400/500, italic allowed) → titles, page headers, card titles
//   sans  (400/500)                 → body, buttons, nav
//   mono  (400/500, uppercase 0.4)  → numbers, timestamps, labels, track codes
//
// Hairlines over shadows. Everything is separated by 1px rules, not elevation.
// Don't add box-shadows except for spotlights on a dark canvas.
//
// Clip fills are ALWAYS derived from the track color:
//   bg = color + '22' (13% alpha) · border = color + '55' (33% alpha) · text = color
