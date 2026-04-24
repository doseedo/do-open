import type { Metadata } from 'next';
import Script from 'next/script';
import { ClerkProvider } from '@clerk/nextjs';
import './globals.css';
// Workbench design tokens — defined at :root so they're available on every
// route (including auth pages that don't render AppShell). Body-level
// activation rules are scoped to body.workbench-theme, so importing here
// doesn't re-theme marketing/auth chrome by itself.
import '@/styles/theme-workbench.css';

export const metadata: Metadata = {
  title: 'Doseedo - Music Production Studio',
  icons: {
    icon: [
      { url: '/favicon/favicon.svg', type: 'image/svg+xml' },
      { url: '/favicon/favicon.ico' },
    ],
    apple: '/favicon/favicon.svg',
  },
  manifest: '/favicon/site.webmanifest',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter&display=swap"
          rel="stylesheet"
        />
        <Script
          src="https://kit.fontawesome.com/0de3cb0a73.js"
          strategy="afterInteractive"
          crossOrigin="anonymous"
        />
      </head>
      <body>
        <div id="root">{children}</div>
      </body>
    </html>
    </ClerkProvider>
  );
}
