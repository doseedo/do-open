import type { Appearance } from '@clerk/types';

// Doseedo brand palette (from var/www/html/doseedo-react/src/styles/colors.css).
const BLUE = '#667eea';
const PURPLE = '#8b5cf6';

export const doseedoClerkAppearance: Appearance = {
  variables: {
    colorPrimary: BLUE,
    colorBackground: 'rgba(20, 20, 30, 0.88)',
    colorText: '#ffffff',
    colorTextSecondary: 'rgba(255, 255, 255, 0.72)',
    colorTextOnPrimaryBackground: '#ffffff',
    colorInputBackground: 'rgba(255, 255, 255, 0.05)',
    colorInputText: '#ffffff',
    colorNeutral: '#ffffff',
    colorDanger: '#ff6b6b',
    colorSuccess: '#4ade80',
    colorWarning: '#facc15',
    colorShimmer: 'rgba(102, 126, 234, 0.3)',
    borderRadius: '10px',
    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
    fontSize: '14px',
    spacingUnit: '1rem',
  },
  elements: {
    rootBox: {
      width: '100%',
      maxWidth: '420px',
    },
    card: {
      backgroundColor: 'rgba(20, 20, 30, 0.88)',
      backdropFilter: 'blur(24px) saturate(160%)',
      WebkitBackdropFilter: 'blur(24px) saturate(160%)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      boxShadow: '0 20px 60px rgba(0, 0, 0, 0.45), 0 0 1px rgba(255, 255, 255, 0.1) inset',
      padding: '2.25rem 2rem',
    },
    headerTitle: {
      color: '#ffffff',
      fontSize: '1.5rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    headerSubtitle: {
      color: 'rgba(255, 255, 255, 0.6)',
    },
    socialButtonsBlockButton: {
      backgroundColor: 'rgba(255, 255, 255, 0.06)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      color: '#ffffff',
      '&:hover': {
        backgroundColor: 'rgba(255, 255, 255, 0.12)',
        borderColor: 'rgba(255, 255, 255, 0.2)',
      },
    },
    socialButtonsBlockButtonText: {
      color: '#ffffff',
    },
    dividerLine: { backgroundColor: 'rgba(255, 255, 255, 0.1)' },
    dividerText: { color: 'rgba(255, 255, 255, 0.4)' },
    formFieldLabel: {
      color: 'rgba(255, 255, 255, 0.88)',
      fontSize: '0.85rem',
      fontWeight: 500,
    },
    formFieldInput: {
      backgroundColor: 'rgba(255, 255, 255, 0.05)',
      border: '1px solid rgba(255, 255, 255, 0.12)',
      color: '#ffffff',
      '&:focus': {
        borderColor: BLUE,
        boxShadow: `0 0 0 3px ${BLUE}33`,
      },
    },
    formButtonPrimary: {
      backgroundColor: BLUE,
      backgroundImage: `linear-gradient(135deg, ${BLUE} 0%, ${PURPLE} 100%)`,
      color: '#ffffff',
      fontWeight: 600,
      textTransform: 'none',
      letterSpacing: '0',
      boxShadow: `0 8px 24px ${BLUE}55`,
      '&:hover': {
        backgroundColor: PURPLE,
        filter: 'brightness(1.05)',
        boxShadow: `0 10px 28px ${PURPLE}66`,
      },
    },
    footer: { display: 'none' }, // hide "Secured by Clerk" badge
    footerAction: {
      color: 'rgba(255, 255, 255, 0.7)',
    },
    footerActionLink: {
      color: BLUE,
      '&:hover': { color: PURPLE },
    },
    identityPreview: {
      backgroundColor: 'rgba(255, 255, 255, 0.04)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
    },
    identityPreviewText: { color: '#ffffff' },
    identityPreviewEditButtonIcon: { color: BLUE },
    formResendCodeLink: { color: BLUE },
    otpCodeFieldInput: {
      backgroundColor: 'rgba(255, 255, 255, 0.05)',
      border: '1px solid rgba(255, 255, 255, 0.15)',
      color: '#ffffff',
    },
  },
  layout: {
    socialButtonsPlacement: 'top',
    socialButtonsVariant: 'blockButton',
    showOptionalFields: true,
    privacyPageUrl: 'https://doseedo.com/privacy',
    termsPageUrl: 'https://doseedo.com/terms',
    helpPageUrl: 'https://doseedo.com/help',
    logoPlacement: 'inside',
  },
};
