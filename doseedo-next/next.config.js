/** @type {import('next').NextConfig} */

// Backend origins — matches the GCLB `bassify` url-map.
// Override per environment via Vercel project env vars if needed.
const AUTH     = process.env.NEXT_PUBLIC_AUTH_ORIGIN     || 'https://doseedo-api.fly.dev';
const CHATBOT  = process.env.NEXT_PUBLIC_CHATBOT_ORIGIN  || 'https://doseedo-chatbot-1028528394180.us-central1.run.app';
const MODAL    = process.env.NEXT_PUBLIC_MODAL_ORIGIN    || 'https://arlo--doseedo-stemphonic-stemphonic-wsgi.modal.run';
const BACKEND  = process.env.NEXT_PUBLIC_BACKEND_ORIGIN  || 'https://backend10-1028528394180.us-central1.run.app';
const FRAMER   = process.env.NEXT_PUBLIC_FRAMER_ORIGIN   || 'https://heartwarming-friday-546447.framer.app';

const nextConfig = {
  reactStrictMode: true,

  // CRA shim: components use ${process.env.PUBLIC_URL}/assets/... which in
  // CRA resolves to '' in prod. Next.js doesn't expose PUBLIC_URL by default,
  // so without this it'd stringify to 'undefined/assets/...' and 404.
  env: {
    PUBLIC_URL: '',
  },


  // CRA-compat: a few Node-only deps get bundled for the browser.
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    // Alias onnxruntime-web to its UMD build to avoid the ESM bundle's
    // `import.meta.url` worker pattern — Next.js's minifier chokes on it.
    // The app already sets env.wasm.wasmPaths to jsdelivr CDN, so we don't
    // need webpack to emit the wasm binaries as chunks.
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      'onnxruntime-web$': require('path').resolve(
        __dirname,
        'node_modules/onnxruntime-web/dist/ort.min.js'
      ),
    };
    return config;
  },

  async rewrites() {
    return [
      // Framer marketing site at /home
      { source: '/home',           destination: `${FRAMER}/` },
      { source: '/home/:path*',    destination: `${FRAMER}/:path*` },

      // Auth service (doseedo-auth Cloud Run)
      { source: '/token',              destination: `${AUTH}/token` },
      { source: '/register',           destination: `${AUTH}/register` },
      { source: '/register/:path*',    destination: `${AUTH}/register/:path*` },
      { source: '/verify',             destination: `${AUTH}/verify` },
      { source: '/verify-google-id-token/:path*', destination: `${AUTH}/verify-google-id-token/:path*` },
      { source: '/logout',             destination: `${AUTH}/logout` },
      { source: '/settings/api-keys',        destination: `${AUTH}/settings/api-keys` },
      { source: '/settings/api-keys/:path*', destination: `${AUTH}/settings/api-keys/:path*` },

      // Auth APIs (match GCLB url-map exactly)
      { source: '/api/sessions',           destination: `${AUTH}/api/sessions` },
      { source: '/api/sessions/:path*',    destination: `${AUTH}/api/sessions/:path*` },
      { source: '/api/upload/:path*',      destination: `${AUTH}/api/upload/:path*` },
      { source: '/api/plugins',            destination: `${AUTH}/api/plugins` },
      { source: '/api/plugins/:path*',     destination: `${AUTH}/api/plugins/:path*` },
      { source: '/api/admin/:path*',       destination: `${AUTH}/api/admin/:path*` },
      { source: '/api/creations',          destination: `${AUTH}/api/creations` },
      { source: '/api/creations/:path*',   destination: `${AUTH}/api/creations/:path*` },
      { source: '/api/profiles',           destination: `${AUTH}/api/profiles` },
      { source: '/api/profiles/:path*',    destination: `${AUTH}/api/profiles/:path*` },
      { source: '/api/me',                 destination: `${AUTH}/api/me` },
      { source: '/api/me/:path*',          destination: `${AUTH}/api/me/:path*` },
      { source: '/api/keys',               destination: `${AUTH}/api/keys` },
      { source: '/api/keys/:path*',        destination: `${AUTH}/api/keys/:path*` },

      // Chatbot
      { source: '/api/chat',         destination: `${CHATBOT}/api/chat` },
      { source: '/api/chat/:path*',  destination: `${CHATBOT}/api/chat/:path*` },

      // Modal (stemphonic)
      { source: '/api/generate-stemphonic',         destination: `${MODAL}/api/generate-stemphonic` },
      { source: '/api/generate-stemphonic/:path*',  destination: `${MODAL}/api/generate-stemphonic/:path*` },
      { source: '/download-stemphonic/:path*',      destination: `${MODAL}/download-stemphonic/:path*` },
      { source: '/api/encode-latents-bulk',         destination: `${MODAL}/api/encode-latents-bulk` },
      { source: '/health',                          destination: `${MODAL}/health` },
      { source: '/_chat/health',                    destination: `${MODAL}/_chat/health` },

      // Generic /api/* catch-all → backend10 (kept last so specifics win)
      { source: '/api/:path*', destination: `${BACKEND}/api/:path*` },
    ];
  },

  // Disable static optimization for the catch-all page so cookie-based auth works.
  async headers() {
    return [
      {
        source: '/',
        headers: [{ key: 'Cache-Control', value: 'no-store' }],
      },
    ];
  },
};

module.exports = nextConfig;
