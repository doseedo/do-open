/**
 * Clerk appearance — workbench cream theme.
 *
 * Matches the rest of the app: cream surfaces, dark ink, mono labels,
 * square-ish radii. Raw hex values (not var(--wb-*)) because Clerk's
 * runtime inlines these into its own <style> islands where custom
 * properties may or may not inherit, depending on shadow-root boundaries.
 * Keep in sync with styles/theme-workbench.css.
 */
export const doseedoClerkAppearance = {
  variables: {
    colorPrimary: '#15181c',        // wb-ink
    colorBackground: '#f2f0ea',     // wb-surface
    colorText: '#15181c',           // wb-ink
    colorTextSecondary: '#3a3d44',  // wb-ink-soft
    colorInputBackground: '#e8e6e1',// wb-bg
    colorInputText: '#15181c',      // wb-ink
    colorNeutral: '#15181c',
    colorDanger: '#c94f2c',         // wb-accent-warm
    colorSuccess: '#2f6b4e',        // wb-ok
    colorWarning: '#c94f2c',
    fontFamily:
      'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    fontFamilyButtons:
      '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace',
    fontSize: '13px',
    borderRadius: '4px',
    spacingUnit: '1rem',
  },
  elements: {
    card: {
      backgroundColor: '#f2f0ea',
      border: '1px solid #a5a29a',   // wb-rule-strong
      boxShadow: '0 8px 24px rgba(20, 22, 26, 0.10)',
      borderRadius: '0',
    },
    headerTitle: {
      color: '#15181c',
      fontWeight: 600,
      letterSpacing: '-0.3px',
    },
    headerSubtitle: {
      color: '#3a3d44',
    },
    footer: { display: 'none' },
    formButtonPrimary: {
      backgroundColor: '#15181c',    // wb-ink
      color: '#e8e6e1',              // wb-bg
      border: '1px solid #15181c',
      borderRadius: '0',
      fontFamily:
        '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace',
      fontSize: '11px',
      fontWeight: 500,
      letterSpacing: '0.8px',
      textTransform: 'uppercase',
      '&:hover, &:focus, &:active': {
        backgroundColor: '#000',
        color: '#e8e6e1',
      },
    },
    socialButtonsBlockButton: {
      backgroundColor: '#e8e6e1',
      border: '1px solid #c8c5bd',   // wb-rule
      color: '#15181c',
      borderRadius: '0',
      '&:hover': {
        backgroundColor: '#dcd9d1',  // wb-surface-2
        borderColor: '#a5a29a',
      },
    },
    formFieldLabel: {
      color: '#3a3d44',
      fontFamily:
        '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace',
      fontSize: '10px',
      letterSpacing: '0.7px',
      textTransform: 'uppercase',
    },
    formFieldInput: {
      backgroundColor: '#e8e6e1',
      border: '1px solid #c8c5bd',
      color: '#15181c',
      borderRadius: '0',
      '&:focus': { borderColor: '#15181c' },
    },
    dividerLine: { backgroundColor: '#c8c5bd' },
    dividerText: {
      color: '#7c7e85',              // wb-ink-mute
      fontSize: '0.78em',
      fontFamily:
        '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace',
      letterSpacing: '0.4px',
    },
    identityPreviewText: { color: '#15181c' },
    identityPreviewEditButton: { color: '#1d4c7a' }, // wb-accent
    formFieldAction: { color: '#1d4c7a' },
    footerActionText: { color: '#3a3d44' },
    footerActionLink: {
      color: '#1d4c7a',
      fontWeight: 500,
      '&:hover': { color: '#15181c' },
    },
  },
  layout: {
    socialButtonsPlacement: 'top',
    socialButtonsVariant: 'blockButton',
    privacyPageUrl: 'https://doseedo.com/privacy',
    termsPageUrl: 'https://doseedo.com/terms',
    helpPageUrl: 'https://doseedo.com/help',
  },
};
