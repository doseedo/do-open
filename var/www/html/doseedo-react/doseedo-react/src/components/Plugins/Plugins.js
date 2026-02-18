import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import * as authService from '../../services/authService';
import styles from './Plugins.module.css';

// Client-side enrichment for known plugins (icons, colors, features)
const pluginMeta = {
  'brass-mute-1': {
    iconImg: '/assets/muted.png',
    color: 'rgba(173, 216, 255, 0.35)',
    features: [
      'Multiple mute types: Straight, Cup, Harmon, Plunger, Bucket',
      'Real-time acoustic modeling',
      'Low-latency processing suitable for live performance',
      'VST3 and AU formats included',
      'Works with any brass instrument input',
      'Adjustable mute depth and resonance',
    ],
  },
  'brass-mute-lite': {
    icon: 'fa-solid fa-trumpet',
    color: 'rgba(102, 126, 234, 0.2)',
    features: [
      'Straight and Cup mute types',
      'Lightweight and CPU-efficient',
      'VST3 and AU formats included',
      'Simple one-knob interface',
      'Works with any brass instrument input',
    ],
  },
};

const defaultMeta = {
  icon: 'fa-solid fa-puzzle-piece',
  color: 'rgba(150, 150, 150, 0.2)',
  features: [],
};

function enrich(apiPlugin) {
  const meta = pluginMeta[apiPlugin.slug] || defaultMeta;
  return {
    ...apiPlugin,
    price: apiPlugin.price_cents / 100,
    icon: meta.icon,
    iconImg: meta.iconImg,
    color: meta.color,
    features: meta.features,
  };
}

/**
 * Plugins page — product listing, detail pages, and download verification.
 * URL routing (same pattern as Research.js):
 *   /plugins            → product grid (fetched from API)
 *   /plugins/:slug      → product detail
 *   /plugins/download/:token → secure download page
 */
const Plugins = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [plugins, setPlugins] = useState([]);
  const [loadingPlugins, setLoadingPlugins] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/plugins').then(r => r.json()).then(data => {
      if (!cancelled) {
        setPlugins(data.map(enrich));
        setLoadingPlugins(false);
      }
    }).catch(() => {
      if (!cancelled) setLoadingPlugins(false);
    });
    return () => { cancelled = true; };
  }, []);

  const pathParts = location.pathname.split('/').filter(Boolean);
  const isDownloadPage = pathParts[1] === 'download' && pathParts[2];
  const downloadToken = isDownloadPage ? pathParts[2] : null;
  const pluginSlug = !isDownloadPage && pathParts.length > 1 ? pathParts[1] : null;

  if (isDownloadPage) {
    return <DownloadPage token={downloadToken} />;
  }

  if (pluginSlug) {
    const plugin = plugins.find(p => p.slug === pluginSlug);
    if (loadingPlugins) {
      return (
        <div className={styles.plugins}>
          <div className={styles.loadingState}>
            <i className="fa-solid fa-spinner fa-spin"></i>
            Loading...
          </div>
        </div>
      );
    }
    if (!plugin) {
      return (
        <div className={styles.plugins}>
          <button className={styles.backBtn} onClick={() => navigate('/plugins')}>
            <i className="fa-solid fa-arrow-left"></i> Back to Plugins
          </button>
          <p style={{ color: 'rgba(255,255,255,0.5)' }}>Plugin not found.</p>
        </div>
      );
    }
    return <ProductDetail plugin={plugin} onBack={() => navigate('/plugins')} />;
  }

  // Grid listing
  return (
    <div className={styles.plugins}>
      <div className={styles.header}>
        <h1 className={styles.title}>Plugins</h1>
      </div>
      {loadingPlugins ? (
        <div className={styles.loadingState}>
          <i className="fa-solid fa-spinner fa-spin"></i>
          Loading plugins...
        </div>
      ) : plugins.length === 0 ? (
        <p style={{ color: 'rgba(255,255,255,0.5)' }}>No plugins available yet.</p>
      ) : (
        <div className={styles.productGrid}>
          {plugins.map(plugin => (
            <div
              key={plugin.slug}
              className={styles.productCard}
              onClick={() => navigate(`/plugins/${plugin.slug}`)}
            >
              <div className={styles.productIcon} style={{ background: plugin.color }}>
                {plugin.iconImg ? (
                  <img src={plugin.iconImg} alt="" style={{ width: 32, height: 32, objectFit: 'contain' }} />
                ) : (
                  <i className={plugin.icon}></i>
                )}
              </div>
              <h2 className={styles.productName}>{plugin.name}</h2>
              <p className={styles.productDesc}>{plugin.description}</p>
              <div className={styles.productPrice}>${plugin.price}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * Product detail view with Buy Now button.
 */
const ProductDetail = ({ plugin, onBack }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleBuy = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const currentUrl = window.location.origin;
      const res = await fetch('/api/plugins/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          plugin_slug: plugin.slug,
          success_url: `${currentUrl}/plugins/download/{CHECKOUT_SESSION_ID}`,
          cancel_url: `${currentUrl}/plugins/${plugin.slug}`,
        }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Failed to start checkout');
      }
      const data = await res.json();
      window.location.href = data.checkout_url;
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }, [plugin.slug]);

  const user = authService.getCurrentUser();

  return (
    <div className={styles.plugins}>
      <button className={styles.backBtn} onClick={onBack}>
        <i className="fa-solid fa-arrow-left"></i> Back to Plugins
      </button>
      <div className={styles.detail}>
        <div className={styles.detailHeader}>
          <div className={styles.detailIcon} style={{ background: plugin.color }}>
            {plugin.iconImg ? (
              <img src={plugin.iconImg} alt="" style={{ width: 40, height: 40, objectFit: 'contain' }} />
            ) : (
              <i className={plugin.icon}></i>
            )}
          </div>
          <h1 className={styles.detailTitle}>{plugin.name}</h1>
          <div className={styles.detailPrice}>${plugin.price}</div>
          <p className={styles.detailDesc}>{plugin.description}</p>
        </div>

        {plugin.features.length > 0 && (
          <div className={styles.features}>
            <h3>Features</h3>
            <ul className={styles.featureList}>
              {plugin.features.map((feat, i) => (
                <li key={i}>
                  <i className="fa-solid fa-check"></i>
                  {feat}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className={styles.buySection}>
          <button
            className={styles.buyBtn}
            onClick={handleBuy}
            disabled={loading}
          >
            <i className={loading ? "fa-solid fa-spinner fa-spin" : "fa-solid fa-cart-shopping"}></i>
            {loading ? 'Redirecting to checkout...' : `Buy Now — $${plugin.price}`}
          </button>
          {!user && (
            <p className={styles.guestNote}>
              Guest checkout available. No account required.
            </p>
          )}
          {error && (
            <p style={{ color: 'rgba(255,100,100,0.9)', marginTop: 12, fontSize: 14 }}>
              {error}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Download page — verifies token via API, shows download button or error.
 * Stripe redirects here with the checkout session ID as the token.
 * The backend webhook creates the actual download token, so we first
 * exchange the session ID, then verify the download token.
 */
const DownloadPage = ({ token }) => {
  const [status, setStatus] = useState('loading'); // loading | ready | error
  const [downloadInfo, setDownloadInfo] = useState(null);
  const [error, setError] = useState('');
  const [downloadToken, setDownloadToken] = useState(token);

  useEffect(() => {
    let cancelled = false;
    const verify = async () => {
      try {
        const res = await fetch(`/api/plugins/verify-download/${token}`, {
          credentials: 'include',
        });
        const data = await res.json();
        if (!cancelled) {
          if (data.valid) {
            setDownloadInfo(data);
            setDownloadToken(data.download_token || token);
            setStatus('ready');
          } else {
            setError(data.error || 'Invalid or expired download link.');
            setStatus('error');
          }
        }
      } catch {
        if (!cancelled) {
          setError('Failed to verify download. Please try again.');
          setStatus('error');
        }
      }
    };
    verify();
    return () => { cancelled = true; };
  }, [token]);

  return (
    <div className={styles.plugins}>
      <div className={styles.downloadPage}>
        {status === 'loading' && (
          <div className={styles.loadingState}>
            <i className="fa-solid fa-spinner fa-spin"></i>
            Verifying your purchase...
          </div>
        )}

        {status === 'ready' && downloadInfo && (
          <>
            <div className={styles.downloadIcon}>
              <i className="fa-solid fa-check"></i>
            </div>
            <h2 className={styles.downloadTitle}>Purchase Confirmed</h2>
            <p className={styles.downloadSubtitle}>
              {downloadInfo.plugin_name} is ready to download.
            </p>
            <a
              href={`/api/plugins/download-file/${downloadToken}`}
              className={styles.downloadBtn}
              download
            >
              <i className="fa-solid fa-download"></i>
              Download Plugin
            </a>
          </>
        )}

        {status === 'error' && (
          <>
            <div className={`${styles.downloadIcon} ${styles.error}`}>
              <i className="fa-solid fa-xmark"></i>
            </div>
            <p className={styles.errorText}>{error}</p>
            <p className={styles.errorDetail}>
              If you believe this is a mistake, contact support@doseedo.com with your receipt.
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default Plugins;
