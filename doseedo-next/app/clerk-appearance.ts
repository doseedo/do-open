import { dark } from '@clerk/themes';
import type { Appearance } from '@clerk/types';

/**
 * Minimal, Linear/Vercel-style dark aesthetic matching
 * auth-service/static/login/index.html (the desktop sign-in page).
 *
 * We start from Clerk's modern `dark` baseTheme and override only the
 * font + radius + a handful of accent colors. Less is more — Clerk's
 * default dark components are already modern; heavy overrides make
 * them look worse, not better.
 */
export const doseedoClerkAppearance: Appearance = {
  baseTheme: dark,
  variables: {
    colorPrimary: '#ffffff',
    colorBackground: '#0a0a0a',
    colorText: '#ffffff',
    colorInputBackground: '#111111',
    colorInputText: '#ffffff',
    colorNeutral: '#ffffff',
    colorDanger: '#ef4444',
    colorSuccess: '#10b981',
    colorWarning: '#f59e0b',
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    fontSize: '14px',
    borderRadius: '6px',
    spacingUnit: '1rem',
  },
  elements: {
    card: {
      backgroundColor: '#0a0a0a',
      border: '1px solid #1a1a1a',
      boxShadow: 'none',
    },
    footer: { display: 'none' },
    formButtonPrimary: {
      backgroundColor: '#e8e8e8',
      color: '#000',
      fontWeight: 600,
      textTransform: 'none',
      '&:hover, &:focus, &:active': {
        backgroundColor: '#ffffff',
        color: '#000',
      },
    },
    socialButtonsBlockButton: {
      backgroundColor: '#111',
      border: '1px solid #222',
      color: '#fff',
      '&:hover': {
        backgroundColor: '#1a1a1a',
        borderColor: '#333',
      },
    },
    formFieldInput: {
      backgroundColor: '#111',
      border: '1px solid #222',
      color: '#fff',
      '&:focus': { borderColor: '#444' },
    },
    dividerLine: { backgroundColor: '#1e1e1e' },
    dividerText: { color: '#444', fontSize: '0.78em' },
  },
  layout: {
    socialButtonsPlacement: 'top',
    socialButtonsVariant: 'blockButton',
    privacyPageUrl: 'https://doseedo.com/privacy',
    termsPageUrl: 'https://doseedo.com/terms',
    helpPageUrl: 'https://doseedo.com/help',
  },
};
