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

  // Emit browser source maps in production so Sentry can un-minify stack
  // traces. The .map files are uploaded to Sentry in scripts/sentry-sourcemaps.sh
  // and should NOT be served alongside the bundle — the postbuild hook moves
  // them out of .next/static before Vercel publishes the assets.
  productionBrowserSourceMaps: true,

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

      // Clerk → legacy cookie bridge (called from /after-signin after Clerk sign-in)
      { source: '/api/auth/clerk-bridge',  destination: `${AUTH}/api/auth/clerk-bridge` },

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

      // Modal (stemphonic + all ML inference — the stemphonic_server WSGI
      // app exposes ~30 routes, most of which were previously behind the
      // dead backend10 catch-all; we list them explicitly now so they actually
      // route through to Modal instead of 404'ing on a nonexistent Cloud Run).
      { source: '/api/generate-stemphonic',         destination: `${MODAL}/api/generate-stemphonic` },
      { source: '/api/generate-stemphonic/:path*',  destination: `${MODAL}/api/generate-stemphonic/:path*` },
      { source: '/download-stemphonic/:path*',      destination: `${MODAL}/download-stemphonic/:path*` },
      { source: '/api/encode-latents-bulk',         destination: `${MODAL}/api/encode-latents-bulk` },
      { source: '/separate-stems',                  destination: `${MODAL}/separate-stems` },
      { source: '/separate-stems/:path*',           destination: `${MODAL}/separate-stems/:path*` },
      { source: '/api/encode-audio-latent',         destination: `${MODAL}/api/encode-audio-latent` },
      { source: '/api/latents/:path*',              destination: `${MODAL}/api/latents/:path*` },
      { source: '/api/upload-latent',               destination: `${MODAL}/api/upload-latent` },
      { source: '/api/extract-midi',                destination: `${MODAL}/api/extract-midi` },
      { source: '/api/extract-midi/:path*',         destination: `${MODAL}/api/extract-midi/:path*` },
      { source: '/api/classify-instrument',         destination: `${MODAL}/api/classify-instrument` },
      { source: '/api/detect-chords',               destination: `${MODAL}/api/detect-chords` },
      { source: '/api/repaint-meter',               destination: `${MODAL}/api/repaint-meter` },
      { source: '/api/regen-stem-for-chord',        destination: `${MODAL}/api/regen-stem-for-chord` },
      { source: '/api/generate-score-from-video',   destination: `${MODAL}/api/generate-score-from-video` },
      { source: '/api/generate-score-from-video/:path*', destination: `${MODAL}/api/generate-score-from-video/:path*` },
      { source: '/api/transcribe-vocals',           destination: `${MODAL}/api/transcribe-vocals` },
      { source: '/api/vae-version',                 destination: `${MODAL}/api/vae-version` },
      { source: '/health',                          destination: `${MODAL}/health` },
      { source: '/_chat/health',                    destination: `${MODAL}/_chat/health` },

      // NOTE: the following /api/* paths the frontend calls are NOT yet on
      // Modal (and the old A100 VM serving them is terminated): upload-audio,
      // audio-to-midi, generate-melody, render-chords, apply-fx, download-with-fx,
      // generate-do, generate-ace-step, drum-sampler/*. Any UI feature that
      // calls these will 404 until we port the handlers into modal_stemphonic.py.
      // Leaving them unrouted rather than silently 404ing via backend10 so
      // failures surface cleanly in the browser devtools.
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
