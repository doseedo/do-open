import type { Metadata } from 'next';
import Script from 'next/script';
import { ClerkProvider } from '@clerk/nextjs';
import './globals.css';

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
        {/*
          Pre-React auth gate (ported from CRA public/index.html).
          Runs before the SPA hydrates. Redirects unauthenticated visitors
          from "/" to "/home" (Framer marketing site, proxied via rewrites).
        */}
        <Script id="auth-gate" strategy="beforeInteractive">{`
          (function () {
            try {
              var p = window.location.pathname;
              if (p !== '/' && p !== '/home') return;
              var hasAuth = /(?:^|;\\s*)username=/.test(document.cookie);
              if (!hasAuth && p === '/') {
                window.location.replace('/home');
              }
            } catch (e) {}
          })();
        `}</Script>
      </head>
      <body>
        <div id="root">{children}</div>
      </body>
    </html>
    </ClerkProvider>
  );
}
