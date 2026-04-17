import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import * as authService from '../../services/authService';
import { listMyProjects, deleteProject, loadProject, generateCode } from '../../services/pluginProjectsAPI';
import { listCreations, getCreation, toggleLike, toggleFavorite, recordDownload, forkCreation, getMyFavorites } from '../../services/communityAPI';
import WebAudioDSPEngine from '../../audio/WebAudioDSPEngine';
import styles from './Plugins.module.css';
import PluginCreator from './PluginCreator/PluginCreator';

// Client-side enrichment for known plugins (icons, colors, features, filter flags)
const pluginMeta = {
  'brass-mute-1': {
    iconImg: '/assets/muted.png',
    color: 'rgba(173, 216, 255, 0.35)',
    heroImg: '/assets/brass-mute-1.png',
    demoYouTube: 'liKdv0GoP7E',
    features: [
      'Multiple mute types: Straight, Cup, Harmon, Plunger, Bucket',
      'Real-time acoustic modeling',
      'Low-latency processing suitable for live performance',
      'VST3 and AU formats included',
      'Works with any brass instrument input',
      'Adjustable mute depth and resonance',
    ],
    isOriginal: true,
    isFree: false,
    isNew: true,
  },
  'brass-mute-lite': {
    iconImg: '/assets/muted.png',
    color: 'rgba(102, 126, 234, 0.2)',
    heroImg: '/assets/brass-mute-1.png',
    demoYouTube: '2SuDG5lVUTc',
    features: [
      'Harmon mute preset',
      'Real-time acoustic modeling',
      'Low-latency processing suitable for live performance',
      'VST3 and AU formats included',
      'Works with any brass instrument input',
    ],
    isOriginal: true,
    isFree: true,
    isNew: true,
  },
};

const defaultMeta = {
  icon: 'fa-solid fa-puzzle-piece',
  color: 'rgba(150, 150, 150, 0.2)',
  features: [],
  isOriginal: false,
  isFree: false,
  isNew: false,
};

const FILTERS = [
  { id: 'trending',  label: 'Trending',       icon: 'fa-solid fa-fire' },
  { id: 'free',      label: 'Free',           icon: 'fa-solid fa-gift' },
  { id: 'user-made', label: 'User Made',      icon: 'fa-solid fa-user' },
  { id: 'originals', label: 'D\u00F8 Originals', icon: 'fa-solid fa-star' },
  { id: 'new',       label: 'New',            icon: 'fa-solid fa-bolt' },
];

// GA4 event helper
const trackEvent = (eventName, params = {}) => {
  if (window.gtag) {
    window.gtag('event', eventName, params);
  }
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
    heroImg: meta.heroImg,
    demoYouTube: meta.demoYouTube,
    isOriginal: meta.isOriginal ?? false,
    isFree: meta.isFree ?? (apiPlugin.price_cents === 0),
    isNew: meta.isNew ?? false,
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
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('trending');
  const [myPlugins, setMyPlugins] = useState([]);
  const [loadingMyPlugins, setLoadingMyPlugins] = useState(false);

  // ── Tabs: Store | My Creations | Community ──
  const [mainTab, setMainTab] = useState('store'); // 'store' | 'creations' | 'community'
  const [myCreations, setMyCreations] = useState([]);
  const [loadingCreations, setLoadingCreations] = useState(false);
  const [communityProjects, setCommunityProjects] = useState([]);
  const [communityTotal, setCommunityTotal] = useState(0);
  const [loadingCommunity, setLoadingCommunity] = useState(false);
  const [communitySearch, setCommunitySearch] = useState('');
  const [communitySort, setCommunitySort] = useState('newest');
  const [viewingProject, setViewingProject] = useState(null);
  const [projectCodePreview, setProjectCodePreview] = useState(null);
  const [generatingProjectCode, setGeneratingProjectCode] = useState(false);
  const [communityOffset, setCommunityOffset] = useState(0);
  const [loadingMoreCommunity, setLoadingMoreCommunity] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const [loadingPreviewId, setLoadingPreviewId] = useState(null);
  const [favoritesProjects, setFavoritesProjects] = useState([]);
  const [loadingFavorites, setLoadingFavorites] = useState(false);

  // Audio preview engine for community plugins
  const previewEngineRef = useRef(null);
  const [previewingId, setPreviewingId] = useState(null);

  // Cleanup preview engine on unmount or tab switch
  useEffect(() => {
    return () => {
      if (previewEngineRef.current) {
        previewEngineRef.current.stop();
        previewEngineRef.current.dispose();
        previewEngineRef.current = null;
      }
    };
  }, []);

  // Stop preview when switching tabs
  useEffect(() => {
    if (mainTab !== 'community' && previewEngineRef.current) {
      previewEngineRef.current.stop();
      previewEngineRef.current.dispose();
      previewEngineRef.current = null;
      setPreviewingId(null);
    }
  }, [mainTab]);

  const handlePreviewToggle = useCallback(async (projectId, dspConfig, e) => {
    if (e) e.stopPropagation();
    // If already previewing this one, stop
    if (previewingId === projectId) {
      if (previewEngineRef.current) {
        previewEngineRef.current.stop();
        previewEngineRef.current.dispose();
        previewEngineRef.current = null;
      }
      setPreviewingId(null);
      return;
    }
    // Stop any existing preview
    if (previewEngineRef.current) {
      previewEngineRef.current.stop();
      previewEngineRef.current.dispose();
      previewEngineRef.current = null;
    }
    // If we have dspConfig inline, use it directly
    if (dspConfig && dspConfig.dspChain?.length > 0) {
      try {
        const engine = new WebAudioDSPEngine(dspConfig);
        engine.loadTestTone('drums');
        engine.setLoop(true);
        engine.setMasterVolume(0.7);
        engine.play();
        previewEngineRef.current = engine;
        setPreviewingId(projectId);
      } catch (err) {
        console.error('Preview failed:', err);
        setPreviewingId(null);
      }
      return;
    }
    // Otherwise load the full project first
    try {
      setLoadingPreviewId(projectId);
      const fullProject = await loadProject(projectId);
      if (fullProject?.dsp_config?.dspChain?.length > 0) {
        const engine = new WebAudioDSPEngine(fullProject.dsp_config);
        engine.loadTestTone('drums');
        engine.setLoop(true);
        engine.setMasterVolume(0.7);
        engine.play();
        previewEngineRef.current = engine;
        setPreviewingId(projectId);
      }
    } catch (err) {
      console.error('Preview load failed:', err);
      setPreviewingId(null);
    } finally {
      setLoadingPreviewId(null);
    }
  }, [previewingId]);

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

  // Fetch user's purchased plugins
  useEffect(() => {
    const user = authService.getCurrentUser();
    if (!user) return;
    setLoadingMyPlugins(true);
    fetch('/api/plugins/my-plugins', { credentials: 'include' })
      .then(r => { if (!r.ok) throw new Error('not found'); return r.json(); })
      .catch(() =>
        fetch('/api/plugins/my-purchases', { credentials: 'include' })
          .then(r => { if (!r.ok) throw new Error('auth failed'); return r.json(); })
      )
      .then(data => {
        setMyPlugins(Array.isArray(data) ? data : []);
        setLoadingMyPlugins(false);
      })
      .catch(() => {
        setMyPlugins([]);
        setLoadingMyPlugins(false);
      });
  }, []);

  const filteredPlugins = useMemo(() => {
    let result = plugins;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(p =>
        p.name.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q)
      );
    }
    switch (activeFilter) {
      case 'free':
        result = result.filter(p => p.isFree);
        break;
      case 'user-made':
        result = result.filter(p => !p.isOriginal);
        break;
      case 'originals':
        result = result.filter(p => p.isOriginal);
        break;
      case 'new':
        result = result.filter(p => p.isNew);
        break;
      case 'trending':
      default:
        break;
    }
    return result;
  }, [plugins, searchQuery, activeFilter]);

  // Fetch my creations when tab opens
  useEffect(() => {
    if (mainTab !== 'creations') return;
    setLoadingCreations(true);
    listMyProjects().then(data => {
      setMyCreations(Array.isArray(data) ? data : []);
    }).catch(() => setMyCreations([])).finally(() => setLoadingCreations(false));
  }, [mainTab]);

  // Fetch community when tab opens or search/sort changes (DB-backed)
  useEffect(() => {
    if (mainTab !== 'community') return;
    setCommunityOffset(0);
    setCommunityProjects([]);
    setLoadingCommunity(true);
    listCreations({ type: 'plugin', search: communitySearch || undefined, sort: communitySort, limit: 24, offset: 0 })
      .then(data => {
        setCommunityProjects(data.creations || []);
        setCommunityTotal(data.total || 0);
      }).catch(() => {
        setCommunityProjects([]);
        setCommunityTotal(0);
      }).finally(() => setLoadingCommunity(false));
  }, [mainTab, communitySearch, communitySort]);

  // Fetch saved/bookmarked plugins
  useEffect(() => {
    if (mainTab !== 'saved') return;
    if (!authService.isAuthenticated()) return;
    setLoadingFavorites(true);
    getMyFavorites({ limit: 50 })
      .then(data => setFavoritesProjects(data.creations || data || []))
      .catch(() => setFavoritesProjects([]))
      .finally(() => setLoadingFavorites(false));
  }, [mainTab]);

  const handleDeleteCreation = useCallback((id, e) => {
    e.stopPropagation();
    setDeleteConfirmId(id);
  }, []);

  const confirmDelete = useCallback(async () => {
    if (!deleteConfirmId) return;
    try {
      await deleteProject(deleteConfirmId);
      setMyCreations(prev => prev.filter(p => p.id !== deleteConfirmId));
    } catch (err) { console.error('Delete failed:', err); }
    finally { setDeleteConfirmId(null); }
  }, [deleteConfirmId]);

  const handleLoadMore = useCallback(async () => {
    const nextOffset = communityOffset + 24;
    setLoadingMoreCommunity(true);
    try {
      const data = await listCreations({ type: 'plugin', search: communitySearch || undefined, sort: communitySort, limit: 24, offset: nextOffset });
      setCommunityProjects(prev => [...prev, ...(data.creations || [])]);
      setCommunityOffset(nextOffset);
    } catch (err) { console.error('Load more failed:', err); }
    finally { setLoadingMoreCommunity(false); }
  }, [communityOffset, communitySearch, communitySort]);

  const handleViewCommunityProject = useCallback(async (projectId) => {
    try {
      // Load full project data from GCS (for components/dsp_config) + DB metadata
      const [fullProject, creationMeta] = await Promise.all([
        loadProject(projectId).catch(() => null),
        getCreation(projectId).catch(() => null),
      ]);
      const merged = { ...(fullProject || {}), ...(creationMeta || {}), id: projectId };
      // Prefer full project data for dsp_config/components, but use DB for like/fav counts
      if (fullProject?.dsp_config) merged.dsp_config = fullProject.dsp_config;
      if (fullProject?.components) merged.components = fullProject.components;
      if (fullProject?.plugin_config) merged.plugin_config = fullProject.plugin_config;
      setViewingProject(merged);
    } catch (err) {
      console.error('Failed to load project:', err);
    }
  }, []);

  const handleDownloadCode = useCallback(async (project) => {
    if (!project.dsp_config) return;
    setGeneratingProjectCode(true);
    try {
      await recordDownload(project.id).catch(() => {});
      const uiLayout = { pluginConfig: project.plugin_config, components: project.components || [] };
      const result = await generateCode(project.dsp_config, uiLayout);
      setProjectCodePreview(result);
    } catch (err) {
      console.error('Code gen failed:', err);
    } finally {
      setGeneratingProjectCode(false);
    }
  }, []);

  const handleToggleLike = useCallback(async (creationId, e) => {
    if (e) e.stopPropagation();
    if (!authService.isAuthenticated()) return;
    try {
      const res = await toggleLike(creationId);
      setCommunityProjects(prev => prev.map(p =>
        p.id === creationId ? { ...p, user_liked: res.liked, like_count: res.like_count } : p
      ));
      if (viewingProject?.id === creationId) {
        setViewingProject(prev => ({ ...prev, user_liked: res.liked, like_count: res.like_count }));
      }
    } catch (err) { console.error('Like failed:', err); }
  }, [viewingProject]);

  const handleToggleFavorite = useCallback(async (creationId, e) => {
    if (e) e.stopPropagation();
    if (!authService.isAuthenticated()) return;
    try {
      const res = await toggleFavorite(creationId);
      setCommunityProjects(prev => prev.map(p =>
        p.id === creationId ? { ...p, user_favorited: res.favorited, favorite_count: res.favorite_count } : p
      ));
      if (viewingProject?.id === creationId) {
        setViewingProject(prev => ({ ...prev, user_favorited: res.favorited, favorite_count: res.favorite_count }));
      }
    } catch (err) { console.error('Favorite failed:', err); }
  }, [viewingProject]);

  const handleFork = useCallback(async (creationId) => {
    if (!authService.isAuthenticated()) return;
    try {
      const res = await forkCreation(creationId);
      if (res.forked_project_id) {
        navigate(`/plugins/create?project=${res.forked_project_id}`);
      }
    } catch (err) { console.error('Fork failed:', err); }
  }, [navigate]);

  const pathParts = location.pathname.split('/').filter(Boolean);
  const isDownloadPage = pathParts[1] === 'download' && pathParts[2];
  const isCreatePage = pathParts[1] === 'create';
  const downloadToken = isDownloadPage ? pathParts[2] : null;
  const pluginSlug = !isDownloadPage && !isCreatePage && pathParts.length > 1 ? pathParts[1] : null;

  if (isCreatePage) {
    return <PluginCreator />;
  }

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

  const user = authService.getCurrentUser();

  // Grid listing
  return (
    <div className={styles.plugins}>
      <div className={styles.header}>
        <h1 className={styles.title}>Plugins</h1>
      </div>

      {/* Main Tabs */}
      <div className={styles.mainTabs}>
        <button
          className={`${styles.mainTab} ${mainTab === 'store' ? styles.mainTabActive : ''}`}
          onClick={() => { setMainTab('store'); setViewingProject(null); }}
        >
          <i className="fa-solid fa-store"></i> Store
        </button>
        <button
          className={`${styles.mainTab} ${mainTab === 'creations' ? styles.mainTabActive : ''}`}
          onClick={() => { setMainTab('creations'); setViewingProject(null); }}
        >
          <i className="fa-solid fa-wand-magic-sparkles"></i> My Creations
        </button>
        <button
          className={`${styles.mainTab} ${mainTab === 'community' ? styles.mainTabActive : ''}`}
          onClick={() => { setMainTab('community'); setViewingProject(null); }}
        >
          <i className="fa-solid fa-users"></i> Community
        </button>
        <button
          className={`${styles.mainTab} ${mainTab === 'saved' ? styles.mainTabActive : ''}`}
          onClick={() => { setMainTab('saved'); setViewingProject(null); }}
        >
          <i className="fa-solid fa-bookmark"></i> Saved
        </button>
      </div>

      {/* ══════════ STORE TAB ══════════ */}
      {mainTab === 'store' && (
        <>
          {/* Search Bar */}
          <div className={styles.searchBarWrapper}>
            <i className="fa-solid fa-magnifying-glass"></i>
            <input
              type="text"
              className={styles.pluginSearchInput}
              placeholder="Search plugins..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button className={styles.clearSearchBtn} onClick={() => setSearchQuery('')}>
                <i className="fa-solid fa-xmark"></i>
              </button>
            )}
          </div>

          {/* Filter Chips */}
          <div className={styles.filterChips}>
            {FILTERS.map(filter => (
              <button
                key={filter.id}
                className={`${styles.filterChip} ${activeFilter === filter.id ? styles.filterChipActive : ''}`}
                onClick={() => setActiveFilter(filter.id)}
              >
                <i className={filter.icon}></i>
                <span>{filter.label}</span>
              </button>
            ))}
          </div>

          {/* Results */}
          {loadingPlugins ? (
            <div className={styles.loadingState}>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Loading plugins...
            </div>
          ) : filteredPlugins.length === 0 ? (
            <div className={styles.noResults}>
              <i className="fa-solid fa-puzzle-piece"></i>
              <p>No plugins found{searchQuery ? ` for "${searchQuery}"` : ' in this category'}.</p>
            </div>
          ) : (
            <div className={styles.productGrid}>
              {filteredPlugins.map(plugin => (
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
                  <div className={styles.productPrice}>
                    {plugin.isFree ? 'Free' : `$${plugin.price}`}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* My Purchased Plugins — only show for logged-in users */}
          {user && (
          <div className={styles.myPluginsSection}>
            <h2 className={styles.myPluginsTitle}>My Purchased Plugins</h2>
            <div className={styles.myPluginsBox}>
              {loadingMyPlugins ? (
                <div className={styles.loadingState}>
                  <i className="fa-solid fa-spinner fa-spin"></i>
                  Loading...
                </div>
              ) : myPlugins.length === 0 ? (
                <p className={styles.myPluginsEmpty}>
                  <i className="fa-solid fa-puzzle-piece"></i>
                  No purchased plugins yet.
                </p>
              ) : (
                <div className={styles.myPluginsGrid}>
                  {myPlugins.map(purchase => {
                    const enriched = plugins.find(p => p.slug === purchase.plugin_slug);
                    return (
                      <div
                        key={purchase.plugin_slug}
                        className={styles.myPluginCard}
                        onClick={() => navigate(`/plugins/${purchase.plugin_slug}`)}
                      >
                        <div
                          className={styles.myPluginIcon}
                          style={{ background: enriched?.color || 'rgba(150,150,150,0.2)' }}
                        >
                          {enriched?.iconImg ? (
                            <img src={enriched.iconImg} alt="" style={{ width: 24, height: 24, objectFit: 'contain' }} />
                          ) : (
                            <i className={enriched?.icon || 'fa-solid fa-puzzle-piece'}></i>
                          )}
                        </div>
                        <span className={styles.myPluginName}>{purchase.plugin_name}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
          )}
        </>
      )}

      {/* ══════════ MY CREATIONS TAB ══════════ */}
      {mainTab === 'creations' && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <p style={{ margin: 0, fontSize: 14, color: 'rgba(255,255,255,0.5)' }}>
              Plugin projects you've designed in the creator.
            </p>
            <button className={styles.createPluginBtn} onClick={() => navigate('/plugins/create')}>
              <i className="fa-solid fa-plus"></i> New Plugin
            </button>
          </div>

          {loadingCreations ? (
            <div className={styles.loadingState}>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Loading your creations...
            </div>
          ) : myCreations.length === 0 ? (
            <div className={styles.noResults}>
              <i className="fa-solid fa-wand-magic-sparkles"></i>
              <p>No plugin projects yet.</p>
              <button className={styles.createPluginBtn} onClick={() => navigate('/plugins/create')} style={{ marginTop: 12 }}>
                <i className="fa-solid fa-plus"></i> Create Your First Plugin
              </button>
            </div>
          ) : (
            <div className={styles.productGrid}>
              {myCreations.map(proj => (
                <div
                  key={proj.id}
                  className={styles.productCard}
                  onClick={() => navigate(`/plugins/create?project=${proj.id}`)}
                >
                  {proj.thumbnail_data ? (
                    <div className={styles.productIcon} style={{ background: 'transparent', overflow: 'hidden' }}>
                      <img src={proj.thumbnail_data} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 12 }} />
                    </div>
                  ) : (
                    <div className={styles.productIcon} style={{ background: 'rgba(102,126,234,0.15)' }}>
                      <i className="fa-solid fa-puzzle-piece" style={{ color: '#667eea' }}></i>
                    </div>
                  )}
                  <h2 className={styles.productName}>{proj.name}</h2>
                  <p className={styles.productDesc}>
                    {proj.description || 'No description'}
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 'auto' }}>
                    {proj.is_public && (
                      <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 20, background: 'rgba(76,175,80,0.15)', color: '#4caf50' }}>
                        <i className="fa-solid fa-globe" style={{ marginRight: 4 }}></i>Public
                      </span>
                    )}
                    <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)' }}>
                      {proj.updated_at ? new Date(proj.updated_at).toLocaleDateString() : ''}
                    </span>
                    <button
                      onClick={(e) => handleDeleteCreation(proj.id, e)}
                      style={{
                        marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer',
                        color: 'rgba(255,255,255,0.2)', fontSize: 13, padding: '4px 8px',
                      }}
                      title="Delete project"
                    >
                      <i className="fa-solid fa-trash"></i>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ══════════ COMMUNITY TAB ══════════ */}
      {mainTab === 'community' && !viewingProject && (
        <>
          <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
            <div className={styles.searchBarWrapper} style={{ flex: 1, minWidth: 200 }}>
              <i className="fa-solid fa-magnifying-glass"></i>
              <input
                type="text"
                className={styles.pluginSearchInput}
                placeholder="Search community plugins..."
                value={communitySearch}
                onChange={e => setCommunitySearch(e.target.value)}
              />
              {communitySearch && (
                <button className={styles.clearSearchBtn} onClick={() => setCommunitySearch('')}>
                  <i className="fa-solid fa-xmark"></i>
                </button>
              )}
            </div>
            <div className={styles.filterChips} style={{ margin: 0 }}>
              {[
                { id: 'newest', label: 'Newest', icon: 'fa-solid fa-clock' },
                { id: 'popular', label: 'Popular', icon: 'fa-solid fa-fire' },
                { id: 'liked', label: 'Most Liked', icon: 'fa-solid fa-heart' },
                { id: 'name', label: 'A-Z', icon: 'fa-solid fa-arrow-down-a-z' },
              ].map(s => (
                <button
                  key={s.id}
                  className={`${styles.filterChip} ${communitySort === s.id ? styles.filterChipActive : ''}`}
                  onClick={() => setCommunitySort(s.id)}
                >
                  <i className={s.icon}></i>
                  <span>{s.label}</span>
                </button>
              ))}
            </div>
          </div>

          {loadingCommunity ? (
            <div className={styles.loadingState}>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Loading community plugins...
            </div>
          ) : communityProjects.length === 0 ? (
            <div className={styles.noResults}>
              <i className="fa-solid fa-users"></i>
              <p>No community plugins yet. Be the first to publish!</p>
            </div>
          ) : (
            <>
              <p style={{ margin: '0 0 12px', fontSize: 13, color: 'rgba(255,255,255,0.35)' }}>
                {communityTotal} plugin{communityTotal !== 1 ? 's' : ''} shared by the community
              </p>
              <div className={styles.productGrid}>
                {communityProjects.map(proj => (
                  <div
                    key={proj.id}
                    className={styles.productCard}
                    onClick={() => handleViewCommunityProject(proj.id)}
                  >
                    {proj.thumbnail_data ? (
                      <div className={styles.productIcon} style={{ background: 'transparent', overflow: 'hidden' }}>
                        <img src={proj.thumbnail_data} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 12 }} />
                      </div>
                    ) : (
                      <div className={styles.productIcon} style={{ background: 'rgba(102,126,234,0.12)' }}>
                        <i className="fa-solid fa-puzzle-piece" style={{ color: '#667eea' }}></i>
                      </div>
                    )}
                    <h2 className={styles.productName}>{proj.name}</h2>
                    <p className={styles.productDesc}>
                      {proj.description || (proj.dsp_summary ? `${proj.dsp_summary.pluginType} - ${proj.dsp_summary.nodeCount} nodes` : 'Plugin project')}
                    </p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 'auto', fontSize: 12 }}>
                      <span
                        style={{ color: 'rgba(255,255,255,0.45)', cursor: 'pointer' }}
                        onClick={e => { e.stopPropagation(); navigate(`/profile/${proj.author?.username || proj.author_name || 'anonymous'}`); }}
                        title="View profile"
                      >
                        {proj.author?.avatar_url ? (
                          <img src={proj.author.avatar_url} alt="" style={{ width: 16, height: 16, borderRadius: '50%', marginRight: 4, verticalAlign: 'middle' }} />
                        ) : (
                          <i className="fa-solid fa-user" style={{ marginRight: 4 }}></i>
                        )}
                        {proj.author?.display_name || proj.author?.username || proj.author_name || 'Anonymous'}
                      </span>
                      <span style={{ color: 'rgba(255,255,255,0.2)' }}>
                        <i className="fa-solid fa-download" style={{ marginRight: 4 }}></i>{proj.download_count || 0}
                      </span>
                      {proj.tags?.length > 0 && (
                        <span style={{ color: 'rgba(186,156,255,0.5)' }}>
                          {proj.tags.slice(0, 2).join(', ')}
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                      <button
                        onClick={e => handleToggleLike(proj.id, e)}
                        style={{
                          background: 'none', border: 'none', cursor: 'pointer', padding: '2px 6px',
                          color: proj.user_liked ? '#ff6b8a' : 'rgba(255,255,255,0.25)', fontSize: 13,
                          display: 'flex', alignItems: 'center', gap: 4, transition: 'color 0.2s',
                        }}
                        title="Like"
                      >
                        <i className={proj.user_liked ? 'fa-solid fa-heart' : 'fa-regular fa-heart'}></i>
                        <span style={{ fontSize: 11 }}>{proj.like_count || 0}</span>
                      </button>
                      <button
                        onClick={e => handleToggleFavorite(proj.id, e)}
                        style={{
                          background: 'none', border: 'none', cursor: 'pointer', padding: '2px 6px',
                          color: proj.user_favorited ? '#fbbf24' : 'rgba(255,255,255,0.25)', fontSize: 13,
                          display: 'flex', alignItems: 'center', gap: 4, transition: 'color 0.2s',
                        }}
                        title="Save"
                      >
                        <i className={proj.user_favorited ? 'fa-solid fa-bookmark' : 'fa-regular fa-bookmark'}></i>
                      </button>
                      {(proj.dsp_summary?.nodeCount > 0 || proj.dsp_config?.dspChain?.length > 0) && (
                        <button
                          onClick={e => handlePreviewToggle(proj.id, proj.dsp_config, e)}
                          style={{
                            background: 'none', border: 'none', cursor: 'pointer', padding: '2px 6px',
                            color: loadingPreviewId === proj.id ? 'rgba(255,255,255,0.5)' : previewingId === proj.id ? '#00e5ff' : 'rgba(255,255,255,0.25)', fontSize: 13,
                            display: 'flex', alignItems: 'center', gap: 4, transition: 'color 0.2s',
                            marginLeft: 'auto',
                          }}
                          title={loadingPreviewId === proj.id ? 'Loading...' : previewingId === proj.id ? 'Stop preview' : 'Preview sound'}
                        >
                          <i className={`fa-solid ${loadingPreviewId === proj.id ? 'fa-spinner fa-spin' : previewingId === proj.id ? 'fa-stop' : 'fa-volume-high'}`}
                            style={previewingId === proj.id ? { animation: 'pulse 1s infinite' } : undefined}
                          ></i>
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              {communityProjects.length < communityTotal && (
                <div style={{ textAlign: 'center', marginTop: 16 }}>
                  <button
                    onClick={handleLoadMore}
                    disabled={loadingMoreCommunity}
                    style={{
                      padding: '10px 24px', borderRadius: 10, border: '1px solid rgba(186,156,255,0.2)',
                      background: 'rgba(186,156,255,0.06)', color: 'rgba(186,156,255,0.8)',
                      fontSize: 13, fontWeight: 600, cursor: loadingMoreCommunity ? 'wait' : 'pointer',
                      opacity: loadingMoreCommunity ? 0.6 : 1,
                    }}
                  >
                    {loadingMoreCommunity
                      ? <><i className="fa-solid fa-spinner fa-spin" style={{ marginRight: 6 }} />Loading...</>
                      : `Load More (${communityTotal - communityProjects.length} remaining)`
                    }
                  </button>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ══════════ SAVED TAB ══════════ */}
      {mainTab === 'saved' && (
        <>
          {!authService.isAuthenticated() ? (
            <div className={styles.noResults}>
              <i className="fa-solid fa-lock"></i>
              <p>Sign in to see your saved plugins.</p>
            </div>
          ) : loadingFavorites ? (
            <div className={styles.loadingState}>
              <i className="fa-solid fa-spinner fa-spin"></i> Loading saved plugins...
            </div>
          ) : favoritesProjects.length === 0 ? (
            <div className={styles.noResults}>
              <i className="fa-solid fa-bookmark"></i>
              <p>No saved plugins yet. Bookmark community plugins to find them here.</p>
            </div>
          ) : (
            <div className={styles.productGrid}>
              {favoritesProjects.map(proj => (
                <div
                  key={proj.id}
                  className={styles.productCard}
                  onClick={() => { setMainTab('community'); handleViewCommunityProject(proj.id || proj.creation_id); }}
                >
                  {proj.thumbnail_data ? (
                    <div className={styles.productIcon} style={{ background: 'transparent', overflow: 'hidden' }}>
                      <img src={proj.thumbnail_data} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 12 }} />
                    </div>
                  ) : (
                    <div className={styles.productIcon} style={{ background: 'rgba(102,126,234,0.12)' }}>
                      <i className="fa-solid fa-puzzle-piece" style={{ color: '#667eea' }}></i>
                    </div>
                  )}
                  <h2 className={styles.productName}>{proj.name}</h2>
                  <p className={styles.productDesc}>{proj.description || 'Plugin project'}</p>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ══════════ COMMUNITY PROJECT DETAIL VIEW ══════════ */}
      {mainTab === 'community' && viewingProject && (
        <CommunityProjectDetail
          project={viewingProject}
          onBack={() => { setViewingProject(null); setProjectCodePreview(null); }}
          onDownloadCode={handleDownloadCode}
          codePreview={projectCodePreview}
          generating={generatingProjectCode}
          navigate={navigate}
          onToggleLike={handleToggleLike}
          onToggleFavorite={handleToggleFavorite}
          onFork={handleFork}
          onPreviewToggle={handlePreviewToggle}
          previewingId={previewingId}
          loadingPreviewId={loadingPreviewId}
        />
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
        }} onClick={() => setDeleteConfirmId(null)}>
          <div onClick={e => e.stopPropagation()} style={{
            background: '#1e1e3a', border: '1px solid rgba(255,100,100,0.2)', borderRadius: 16,
            padding: 28, width: 360, maxWidth: '90vw', color: '#fff',
          }}>
            <h3 style={{ margin: '0 0 10px', fontSize: 16 }}>
              <i className="fa-solid fa-triangle-exclamation" style={{ color: '#f44336', marginRight: 8 }} />
              Delete Plugin?
            </h3>
            <p style={{ margin: '0 0 20px', fontSize: 13, color: 'rgba(255,255,255,0.5)' }}>
              This will permanently delete the plugin project. This cannot be undone.
            </p>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setDeleteConfirmId(null)} style={{
                padding: '8px 16px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)',
                background: 'transparent', color: 'rgba(255,255,255,0.6)', fontSize: 13, cursor: 'pointer',
              }}>
                Cancel
              </button>
              <button onClick={confirmDelete} style={{
                padding: '8px 20px', borderRadius: 8, border: 'none',
                background: 'rgba(244,67,54,0.85)', color: '#fff', fontSize: 13, fontWeight: 600, cursor: 'pointer',
              }}>
                <i className="fa-solid fa-trash" style={{ marginRight: 6 }} />Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Community project detail view
 */
const CommunityProjectDetail = ({ project, onBack, onDownloadCode, codePreview, generating, navigate, onToggleLike, onToggleFavorite, onFork, onPreviewToggle, previewingId, loadingPreviewId }) => {
  const [activeFile, setActiveFile] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (codePreview?.files) {
      setActiveFile(Object.keys(codePreview.files)[0] || null);
    }
  }, [codePreview]);

  return (
    <>
      <button className={styles.backBtn} onClick={onBack}>
        <i className="fa-solid fa-arrow-left"></i> Back to Community
      </button>

      <div style={{ display: 'flex', gap: 24, marginTop: 16, flexWrap: 'wrap' }}>
        {/* Info */}
        <div style={{ flex: '1 1 300px', minWidth: 300 }}>
          <h2 style={{ margin: '0 0 8px', fontSize: 22, fontWeight: 600 }}>{project.name}</h2>
          <div style={{ margin: '0 0 12px', fontSize: 13, color: 'rgba(255,255,255,0.4)', display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span
              style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
              onClick={() => navigate(`/profile/${project.author?.username || project.author_name || 'anonymous'}`)}
              title="View profile"
            >
              {project.author?.avatar_url ? (
                <img src={project.author.avatar_url} alt="" style={{ width: 20, height: 20, borderRadius: '50%' }} />
              ) : (
                <i className="fa-solid fa-user"></i>
              )}
              <span style={{ color: 'rgba(186,156,255,0.8)' }}>
                {project.author?.display_name || project.author?.username || project.author_name || 'Anonymous'}
              </span>
            </span>
            <span>&middot; {project.download_count || 0} downloads</span>
            <button
              onClick={() => onToggleLike(project.id)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer', padding: '2px 8px',
                color: project.user_liked ? '#ff6b8a' : 'rgba(255,255,255,0.35)', fontSize: 14,
                display: 'flex', alignItems: 'center', gap: 5,
              }}
            >
              <i className={project.user_liked ? 'fa-solid fa-heart' : 'fa-regular fa-heart'}></i>
              {project.like_count || 0}
            </button>
            <button
              onClick={() => onToggleFavorite(project.id)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer', padding: '2px 8px',
                color: project.user_favorited ? '#fbbf24' : 'rgba(255,255,255,0.35)', fontSize: 14,
                display: 'flex', alignItems: 'center', gap: 5,
              }}
            >
              <i className={project.user_favorited ? 'fa-solid fa-bookmark' : 'fa-regular fa-bookmark'}></i>
              Save
            </button>
          </div>
          {project.description && (
            <p style={{ margin: '0 0 16px', fontSize: 14, color: 'rgba(255,255,255,0.6)', lineHeight: 1.5 }}>
              {project.description}
            </p>
          )}
          {project.tags?.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 16 }}>
              {project.tags.map(tag => (
                <span key={tag} style={{
                  padding: '3px 10px', borderRadius: 20, fontSize: 11,
                  background: 'rgba(186,156,255,0.1)', color: 'rgba(186,156,255,0.7)',
                }}>
                  {tag}
                </span>
              ))}
            </div>
          )}
          {project.dsp_config && (
            <div style={{ marginBottom: 16, padding: 12, borderRadius: 10, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.35)', marginBottom: 6 }}>DSP Chain</div>
              <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)' }}>
                {project.dsp_config.pluginType || 'effect'} &middot;{' '}
                {project.dsp_config.dspChain?.length || 0} nodes &middot;{' '}
                {project.dsp_config.parameters?.length || 0} parameters
              </div>
              {project.dsp_config.dspChain?.length > 0 && (
                <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.3)', marginTop: 4 }}>
                  {project.dsp_config.dspChain.map(n => n.type).join(' → ')}
                </div>
              )}
            </div>
          )}

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {project.dsp_config?.dspChain?.length > 0 && onPreviewToggle && (
              <button
                onClick={() => onPreviewToggle(project.id, project.dsp_config)}
                style={{
                  padding: '10px 20px', borderRadius: 10,
                  border: previewingId === project.id ? '1px solid rgba(0,229,255,0.4)' : '1px solid rgba(0,229,255,0.25)',
                  background: previewingId === project.id ? 'rgba(0,229,255,0.12)' : 'transparent',
                  color: previewingId === project.id ? '#00e5ff' : 'rgba(0,229,255,0.7)',
                  fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8,
                  transition: 'all 0.2s',
                }}
              >
                <i className={`fa-solid ${loadingPreviewId === project.id ? 'fa-spinner fa-spin' : previewingId === project.id ? 'fa-stop' : 'fa-volume-high'}`}></i>
                {loadingPreviewId === project.id ? 'Loading...' : previewingId === project.id ? 'Stop Preview' : 'Preview Sound'}
              </button>
            )}
            {project.dsp_config && (
              <button
                onClick={() => onDownloadCode(project)}
                disabled={generating}
                style={{
                  padding: '10px 20px', borderRadius: 10, border: 'none',
                  background: 'linear-gradient(135deg, #667eea, #764ba2)', color: '#fff',
                  fontSize: 14, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8,
                  opacity: generating ? 0.6 : 1,
                }}
              >
                <i className={`fa-solid ${generating ? 'fa-spinner fa-spin' : 'fa-code'}`}></i>
                {generating ? 'Generating...' : 'Get JUCE Code'}
              </button>
            )}
            <button
              onClick={() => navigate(`/plugins/create?project=${project.id}`)}
              style={{
                padding: '10px 20px', borderRadius: 10,
                border: '1px solid rgba(186,156,255,0.3)', background: 'transparent',
                color: 'rgba(186,156,255,0.8)', fontSize: 14, cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 8,
              }}
            >
              <i className="fa-solid fa-eye"></i> View in Creator
            </button>
            <button
              onClick={() => {
                if (!authService.isAuthenticated()) {
                  alert('Sign in to fork this plugin to your creations.');
                  return;
                }
                onFork(project.id);
              }}
              title="Copy this plugin to your own creations so you can edit and modify it"
              style={{
                padding: '10px 20px', borderRadius: 10,
                border: '1px solid rgba(102,126,234,0.3)', background: 'transparent',
                color: 'rgba(102,126,234,0.8)', fontSize: 14, cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 8,
              }}
            >
              <i className="fa-solid fa-code-fork"></i> Fork (Edit Your Own Copy)
            </button>
          </div>
        </div>

        {/* Preview / Thumbnail */}
        <div style={{ flex: '0 0 auto', width: 300 }}>
          {project.thumbnail_data ? (
            <img src={project.thumbnail_data} alt={project.name} style={{
              width: '100%', borderRadius: 12, border: '1px solid rgba(255,255,255,0.06)',
            }} />
          ) : (
            <div style={{
              width: '100%', height: 200, borderRadius: 12, display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
              color: 'rgba(255,255,255,0.15)', fontSize: 14,
            }}>
              <i className="fa-solid fa-image" style={{ marginRight: 8 }}></i> No preview
            </div>
          )}
        </div>
      </div>

      {/* Code Preview */}
      {codePreview && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ margin: '0 0 8px', fontSize: 15, fontWeight: 600 }}>
            <i className="fa-solid fa-code" style={{ marginRight: 8 }}></i>Generated Code
          </h3>
          <div style={{ display: 'flex', gap: 4, marginBottom: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {Object.keys(codePreview.files).map(fname => (
              <button
                key={fname}
                onClick={() => setActiveFile(fname)}
                style={{
                  padding: '4px 10px', borderRadius: 6, border: 'none', fontSize: 12, cursor: 'pointer',
                  background: activeFile === fname ? 'rgba(102,126,234,0.3)' : 'rgba(255,255,255,0.06)',
                  color: activeFile === fname ? '#667eea' : 'rgba(255,255,255,0.5)',
                }}
              >
                {fname}
              </button>
            ))}
            <button
              onClick={() => {
                navigator.clipboard.writeText(codePreview.files[activeFile] || '');
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              }}
              style={{
                marginLeft: 'auto', padding: '4px 10px', borderRadius: 6, fontSize: 11, cursor: 'pointer',
                border: '1px solid rgba(255,255,255,0.12)', background: 'rgba(255,255,255,0.05)',
                color: copied ? '#4caf50' : 'rgba(255,255,255,0.5)', display: 'flex', alignItems: 'center', gap: 4,
              }}
            >
              <i className={`fa-solid ${copied ? 'fa-check' : 'fa-clipboard'}`}></i>
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <pre style={{
            overflow: 'auto', maxHeight: 500, padding: 16, borderRadius: 10,
            background: 'rgba(0,0,0,0.4)', fontSize: 12, lineHeight: 1.5,
            color: 'rgba(255,255,255,0.8)', margin: 0, whiteSpace: 'pre-wrap',
            border: '1px solid rgba(255,255,255,0.06)',
          }}>
            <code>{codePreview.files[activeFile] || ''}</code>
          </pre>
        </div>
      )}
    </>
  );
};

/**
 * Inline auth modal — email-only signup + Google sign-in for free plugin download.
 */
const AuthModal = ({ open, onClose, onSuccess, pluginSlug = 'brass-mute-lite' }) => {
  const [email, setEmail] = useState('');
  const [authError, setAuthError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const googleBtnRef = useRef(null);

  // Render Google sign-in button when modal opens
  useEffect(() => {
    if (!open || !googleBtnRef.current) return;
    const interval = setInterval(() => {
      if (window.google?.accounts?.id) {
        clearInterval(interval);
        window.google.accounts.id.initialize({
          client_id: '1028528394180-t25p4ross097jh8018sh42p8ddqgam9r.apps.googleusercontent.com',
          callback: handleGoogleResponse,
        });
        window.google.accounts.id.renderButton(googleBtnRef.current, {
          type: 'standard',
          size: 'large',
          theme: 'outline',
          text: 'continue_with',
          shape: 'rectangular',
          width: 300,
        });
      }
    }, 100);
    return () => clearInterval(interval);
  }, [open]);

  const handleGoogleResponse = async (response) => {
    setAuthError('');
    setSubmitting(true);
    try {
      const decoded = JSON.parse(atob(response.credential.split('.')[1]));
      const googleEmail = decoded.email;
      const googleProfile = {
        getEmail: () => googleEmail,
        getId: () => decoded.sub,
        getName: () => decoded.name || googleEmail.split('@')[0],
        getPicture: () => decoded.picture || 'user.png',
      };
      await authService.loginWithGoogle(googleProfile);
      const result = await authService.liteClaimPlugin(googleEmail, pluginSlug);
      if (window.gtag) {
        window.gtag('event', 'sign_up', { method: 'google' });
      }
      onSuccess(result);
    } catch (err) {
      setAuthError(err.message || 'Google sign-in failed');
    }
    setSubmitting(false);
  };

  const handleEmailSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim()) return;
    setAuthError('');
    setSubmitting(true);
    try {
      const result = await authService.liteClaimPlugin(email.trim(), pluginSlug);
      onSuccess(result);
    } catch (err) {
      setAuthError(err.message || 'Something went wrong');
    }
    setSubmitting(false);
  };

  if (!open) return null;

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={e => e.stopPropagation()}>
        <button className={styles.modalClose} onClick={onClose}>
          <i className="fa-solid fa-xmark"></i>
        </button>

        <img src="/assets/muted.png" alt="" style={{ width: 40, height: 40, marginBottom: 12, opacity: 0.8 }} />
        <h2 className={styles.modalTitle}>Get the free version</h2>
        <p className={styles.modalSubtitle}>Enter your email to download Brass Mute Lite</p>

        <form onSubmit={handleEmailSubmit} className={styles.modalForm}>
          <input
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            placeholder="Enter your email"
            autoFocus
            required
            className={styles.modalInput}
          />
          <button type="submit" className={styles.modalSubmitBtn} disabled={submitting}>
            {submitting ? 'Setting up...' : 'Get Free Download'}
          </button>
        </form>

        {authError && <p className={styles.modalError}>{authError}</p>}

        <div className={styles.modalDivider}>
          <hr /><span>OR</span><hr />
        </div>

        <div ref={googleBtnRef} className={styles.googleBtnWrapper}></div>

        <p className={styles.modalTerms}>
          By continuing, you agree to our <a href="/terms">Terms</a> and <a href="/privacy">Privacy Policy</a>.
        </p>
      </div>
    </div>
  );
};

/**
 * Product detail view with Buy Now button.
 */
const ProductDetail = ({ plugin, onBack }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Track product page view
  useEffect(() => {
    trackEvent('view_item', {
      currency: 'USD',
      value: plugin.price,
      items: [{ item_id: plugin.slug, item_name: plugin.name, price: plugin.price }],
    });
  }, [plugin.slug, plugin.name, plugin.price]);

  const handleBuy = useCallback(async () => {
    setLoading(true);
    setError(null);
    trackEvent('begin_checkout', {
      currency: 'USD',
      value: plugin.price,
      items: [{ item_id: plugin.slug, item_name: plugin.name, price: plugin.price }],
    });
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
      trackEvent('checkout_redirect', { plugin: plugin.slug, plugin_name: plugin.name });
      window.location.href = data.checkout_url;
    } catch (err) {
      trackEvent('checkout_error', { plugin: plugin.slug, error: err.message });
      setError(err.message);
      setLoading(false);
    }
  }, [plugin.slug, plugin.name, plugin.price]);

  const handleDownloadFree = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // lite-claim works for both new and existing users; backend uses JWT email if authenticated
      const result = await authService.liteClaimPlugin('', plugin.slug);
      trackEvent('file_download', { plugin: plugin.slug, plugin_name: plugin.name });
      window.location.href = result.download_url;
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  }, [plugin.slug, plugin.name]);

  const [showAuthModal, setShowAuthModal] = useState(false);
  const navigate = useNavigate();
  const detailLocation = useLocation();
  const user = authService.getCurrentUser();

  // Auto-trigger download if navigated here with a download URL (from brass-mute-1 page)
  useEffect(() => {
    if (detailLocation.state?.autoDownloadUrl) {
      trackEvent('file_download', { plugin: plugin.slug, plugin_name: plugin.name });
      window.location.href = detailLocation.state.autoDownloadUrl;
      window.history.replaceState({}, document.title);
    }
  }, [detailLocation.state, plugin.slug, plugin.name]);

  return (
    <div className={styles.plugins}>
      <AuthModal
        open={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        pluginSlug={plugin.isFree ? plugin.slug : 'brass-mute-lite'}
        onSuccess={(result) => {
          setShowAuthModal(false);
          if (plugin.isFree) {
            // On lite page — trigger download directly
            trackEvent('file_download', { plugin: plugin.slug, plugin_name: plugin.name });
            window.location.href = result.download_url;
          } else {
            // On paid page — navigate to lite page with auto-download
            navigate('/plugins/brass-mute-lite', { state: { autoDownloadUrl: result.download_url } });
          }
        }}
      />
      <button className={styles.backBtn} onClick={onBack}>
        <i className="fa-solid fa-arrow-left"></i> Back to Plugins
      </button>
      <div className={styles.detailLayout}>
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
            <div className={styles.detailPrice}>
              {plugin.isFree ? 'Free' : `$${plugin.price}`}
              {!plugin.isFree && (
                <span className={styles.tryFreeNote}> - <button onClick={() => setShowAuthModal(true)} className={styles.tryFreeLink}>get the free version</button></span>
              )}
            </div>
            <p className={styles.detailDesc}>{plugin.description}</p>
          </div>

          {/* Mobile image — shown under description on small screens */}
          <div className={styles.mediaMobileImage}>
            {plugin.heroImg ? (
              <img src={plugin.heroImg} alt={plugin.name} className={styles.mediaImage} />
            ) : (
              <div className={styles.mediaPlaceholder}>
                <i className="fa-solid fa-image"></i>
                <span>Plugin Image</span>
              </div>
            )}
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
            {plugin.isFree ? (
              <>
                <button
                  className={styles.downloadFreeBtn}
                  onClick={() => {
                    if (!user) {
                      setShowAuthModal(true);
                    } else {
                      handleDownloadFree();
                    }
                  }}
                  disabled={loading}
                >
                  <i className={loading ? "fa-solid fa-spinner fa-spin" : "fa-solid fa-download"}></i>
                  {loading ? 'Downloading...' : 'Download Free'}
                </button>
                {!user && (
                  <p className={styles.guestNote}>
                    Enter your email to download.
                  </p>
                )}
              </>
            ) : (
              <>
                <div className={styles.buyActions}>
                  <button
                    className={styles.buyBtn}
                    onClick={handleBuy}
                    disabled={loading}
                  >
                    <i className={loading ? "fa-solid fa-spinner fa-spin" : "fa-solid fa-cart-shopping"}></i>
                    {loading ? 'Redirecting to checkout...' : `Buy Now — $${plugin.price}`}
                  </button>
                  <button
                    className={styles.freeVersionLink}
                    onClick={() => user ? navigate('/plugins/brass-mute-lite') : setShowAuthModal(true)}
                  >
                    <i className="fa-solid fa-gift"></i>
                    Free version — {user ? 'Download' : 'Sign up free'}
                  </button>
                </div>
                {!user && (
                  <p className={styles.guestNote}>
                    Guest checkout available. No account required.
                  </p>
                )}
              </>
            )}
            {error && (
              <p style={{ color: 'rgba(255,100,100,0.9)', marginTop: 12, fontSize: 14 }}>
                {error}
              </p>
            )}
            {!plugin.isFree && (
              <p className={styles.legalNote}>
                By purchasing, you agree to our{' '}
                <a href="/terms">Terms of Service</a> and{' '}
                <a href="/privacy">Privacy Policy</a>.
                All sales are final.
              </p>
            )}
          </div>

          {/* Mobile video — shown under buy section on small screens */}
          <div className={styles.mediaMobileVideo}>
            {plugin.demoYouTube ? (
              <div className={styles.youtubeWrapper}>
                <iframe
                  src={`https://www.youtube.com/embed/${plugin.demoYouTube}`}
                  title="Plugin demo"
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
            ) : (
              <div className={styles.mediaPlaceholder}>
                <i className="fa-solid fa-play"></i>
                <span>Plugin Video</span>
              </div>
            )}
          </div>
        </div>

        {/* Desktop media sidebar — hidden on mobile */}
        <div className={styles.detailMedia}>
          {plugin.heroImg ? (
            <img src={plugin.heroImg} alt={plugin.name} className={styles.mediaImage} />
          ) : (
            <div className={styles.mediaPlaceholder}>
              <i className="fa-solid fa-image"></i>
              <span>Plugin Image</span>
            </div>
          )}
          {plugin.demoYouTube ? (
            <div className={styles.youtubeWrapper}>
              <iframe
                src={`https://www.youtube.com/embed/${plugin.demoYouTube}`}
                title="Plugin demo"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          ) : (
            <div className={styles.mediaPlaceholder}>
              <i className="fa-solid fa-play"></i>
              <span>Plugin Video</span>
            </div>
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
  const [status, setStatus] = useState('email'); // email | loading | ready | error
  const [downloadInfo, setDownloadInfo] = useState(null);
  const [error, setError] = useState('');
  const [downloadToken, setDownloadToken] = useState(token);
  const [email, setEmail] = useState('');
  const [emailError, setEmailError] = useState('');
  const [pluginName, setPluginName] = useState('');
  const [verifying, setVerifying] = useState(false);

  // Initial check to get plugin name and confirm token exists
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`/api/plugins/verify-download/${token}`, {
          credentials: 'include',
        });
        const data = await res.json();
        if (!cancelled) {
          if (data.needs_email) {
            setPluginName(data.plugin_name || 'Plugin');
            setStatus('email');
          } else if (data.valid) {
            setDownloadInfo(data);
            setDownloadToken(data.download_token || token);
            trackEvent('purchase', {
              currency: 'USD',
              transaction_id: token,
              items: [{ item_name: data.plugin_name }],
            });
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
    check();
    return () => { cancelled = true; };
  }, [token]);

  const handleEmailSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim()) return;
    setEmailError('');
    setVerifying(true);
    try {
      const res = await fetch(
        `/api/plugins/verify-download/${token}?email=${encodeURIComponent(email.trim())}`,
        { credentials: 'include' }
      );
      const data = await res.json();
      if (data.valid) {
        setDownloadInfo(data);
        setDownloadToken(data.download_token || token);
        trackEvent('purchase', {
          currency: 'USD',
          transaction_id: token,
          items: [{ item_name: data.plugin_name }],
        });
        setStatus('ready');
      } else {
        setEmailError(data.error || 'Verification failed.');
      }
    } catch {
      setEmailError('Failed to verify. Please try again.');
    }
    setVerifying(false);
  };

  return (
    <div className={styles.plugins}>
      <div className={styles.downloadPage}>
        {status === 'email' && (
          <>
            <div className={styles.downloadIcon}>
              <i className="fa-solid fa-envelope"></i>
            </div>
            <h2 className={styles.downloadTitle}>Verify Your Purchase</h2>
            <p className={styles.downloadSubtitle}>
              {pluginName ? `Enter the email you used to purchase ${pluginName}.` : 'Enter the email you used at checkout.'}
            </p>
            <form onSubmit={handleEmailSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, width: '100%', maxWidth: 340 }}>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="your@email.com"
                autoFocus
                required
                style={{
                  padding: '12px 16px', fontSize: 15, borderRadius: 10, width: '100%',
                  border: '1px solid rgba(186,156,255,0.3)', background: 'rgba(255,255,255,0.06)',
                  color: '#fff', outline: 'none',
                }}
              />
              <button
                type="submit"
                disabled={verifying}
                className={styles.downloadBtn}
                style={{ width: '100%', justifyContent: 'center', opacity: verifying ? 0.6 : 1 }}
              >
                <i className={`fa-solid ${verifying ? 'fa-spinner fa-spin' : 'fa-arrow-right'}`}></i>
                {verifying ? 'Verifying...' : 'Continue'}
              </button>
              {emailError && <p style={{ color: 'rgba(248,113,113,0.9)', fontSize: 14, margin: 0 }}>{emailError}</p>}
            </form>
          </>
        )}

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
              href={`/api/plugins/download-file/${downloadToken}?email=${encodeURIComponent(email.trim())}`}
              className={styles.downloadBtn}
              download
              onClick={() => trackEvent('file_download', { plugin: downloadInfo?.plugin_name })}
            >
              <i className="fa-solid fa-download"></i>
              Download Plugin
            </a>
            <p style={{ marginTop: 24, fontSize: 13, color: 'rgba(255,255,255,0.4)' }}>
              Questions or issues? Reach out to <a href="mailto:support@doseedo.com" style={{ color: 'rgba(186,156,255,0.7)', textDecoration: 'none' }}>support@doseedo.com</a>
            </p>
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
