import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './Models.module.css';
import PageTopbar from '../Sidebar/PageTopbar';

// Placeholder catalog — replace with /api/models backend once wired.
const MODEL_CATALOG = [
  {
    id: 'basic-pitch',
    name: 'Basic Pitch',
    tagline: 'Polyphonic note transcription',
    description: 'Converts audio to MIDI with polyphonic pitch + onset detection. Runs in-browser via ONNX.',
    category: 'transcription',
    vendor: 'Spotify · ported',
    size: '6.9 MB',
    runtime: 'onnx',
    latency: '< 200 ms',
    license: 'CC-BY-4.0',
    isFree: true,
    isNew: false,
    featured: false,
  },
  {
    id: 'latent-pitch',
    name: 'Latent Pitch',
    tagline: 'Phase-aware pitch shifter',
    description: 'Neural pitch shifting with preserved formants and transient integrity. 48 kHz, 128-frame windows.',
    category: 'synthesis',
    vendor: 'Doseedo',
    size: '18.2 MB',
    runtime: 'onnx',
    latency: '< 40 ms',
    license: 'Commercial',
    isFree: false,
    isNew: true,
    featured: true,
  },
  {
    id: 'stemphonic',
    name: 'Stemphonic',
    tagline: '4-stem source separation',
    description: 'Vocals / drums / bass / other separation. Modal-hosted A100 inference for full-length tracks.',
    category: 'separation',
    vendor: 'Doseedo',
    size: 'hosted',
    runtime: 'modal',
    latency: '30-90 s / track',
    license: 'Commercial',
    isFree: false,
    isNew: false,
    featured: true,
  },
  {
    id: 'polypitch',
    name: 'Polypitch',
    tagline: 'Per-stem polyphonic pitch map',
    description: 'Latent-mask pitch detection that tracks multiple voices per stem. Drives the studio pitch grid.',
    category: 'transcription',
    vendor: 'Doseedo',
    size: 'hosted',
    runtime: 'modal',
    latency: '10-40 s / stem',
    license: 'Commercial',
    isFree: false,
    isNew: true,
    featured: false,
  },
  {
    id: 'chord-detect',
    name: 'ChordNet',
    tagline: 'Chord & key detection',
    description: 'Real-time chord extraction with key + tempo inference. Feeds the Studio chord lane.',
    category: 'analysis',
    vendor: 'Doseedo',
    size: '12.4 MB',
    runtime: 'onnx',
    latency: '< 500 ms',
    license: 'MIT',
    isFree: true,
    isNew: false,
    featured: false,
  },
  {
    id: 'drum-sampler',
    name: 'Drum Sampler v2',
    tagline: 'Neural drum synthesis',
    description: 'Generates drum one-shots from textual prompts. Kick, snare, hat, perc categories.',
    category: 'generation',
    vendor: 'Doseedo',
    size: 'hosted',
    runtime: 'modal',
    latency: '2-6 s',
    license: 'Commercial',
    isFree: false,
    isNew: true,
    featured: false,
  },
  {
    id: 'voice-clone',
    name: 'Voice Clone',
    tagline: 'Zero-shot voice conversion',
    description: 'Transfer a source vocal onto a reference voice with minimal artifacts. 24 kHz output.',
    category: 'synthesis',
    vendor: 'Doseedo · research',
    size: 'hosted',
    runtime: 'modal',
    latency: '5-15 s',
    license: 'Research preview',
    isFree: false,
    isNew: true,
    featured: false,
  },
  {
    id: 'melody-gen',
    name: 'MelodyGen',
    tagline: 'Controllable melody generation',
    description: 'Conditions on chord track + style tag to generate MIDI melodies. Rolling 8-bar context.',
    category: 'generation',
    vendor: 'Doseedo · research',
    size: 'hosted',
    runtime: 'modal',
    latency: '1-3 s / bar',
    license: 'Research preview',
    isFree: true,
    isNew: false,
    featured: false,
  },
];

const CATEGORIES = [
  { id: 'all', label: 'All' },
  { id: 'transcription', label: 'Transcription' },
  { id: 'separation', label: 'Separation' },
  { id: 'synthesis', label: 'Synthesis' },
  { id: 'generation', label: 'Generation' },
  { id: 'analysis', label: 'Analysis' },
];

const FILTERS = [
  { id: 'trending', label: 'Trending', icon: 'fa-solid fa-fire' },
  { id: 'free', label: 'Free', icon: 'fa-solid fa-gift' },
  { id: 'new', label: 'New', icon: 'fa-solid fa-bolt' },
  { id: 'featured', label: 'Featured', icon: 'fa-solid fa-star' },
];

function Models() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('all');
  const [activeFilter, setActiveFilter] = useState('trending');
  const [selectedModel, setSelectedModel] = useState(null);

  const filtered = useMemo(() => {
    let r = MODEL_CATALOG;
    if (activeCategory !== 'all') {
      r = r.filter(m => m.category === activeCategory);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      r = r.filter(m =>
        m.name.toLowerCase().includes(q) ||
        m.tagline.toLowerCase().includes(q) ||
        m.description.toLowerCase().includes(q)
      );
    }
    switch (activeFilter) {
      case 'free':     r = r.filter(m => m.isFree); break;
      case 'new':      r = r.filter(m => m.isNew); break;
      case 'featured': r = r.filter(m => m.featured); break;
      case 'trending':
      default: break;
    }
    return r;
  }, [searchQuery, activeCategory, activeFilter]);

  const featured = MODEL_CATALOG.find(m => m.featured);
  const counts = {
    total: MODEL_CATALOG.length,
    free: MODEL_CATALOG.filter(m => m.isFree).length,
    updated: MODEL_CATALOG.filter(m => m.isNew).length,
  };

  return (
    <div className={styles.models}>
      <PageTopbar section="Create" title="Models" meta={`${counts.total} models`} />
      {/* Header */}
      <div className={styles.head}>
        <div>
          <h1 className={`${styles.title} page-title`}>Models</h1>
          <div className={styles.sub}>
            <b>{counts.total} models</b> &nbsp;·&nbsp; browse · benchmark · deploy
          </div>
        </div>
        <div className={styles.counts}>
          <div><b>{counts.total}</b>catalog</div>
          <div><b>{counts.free}</b>free</div>
          <div><b>{counts.updated}</b>new</div>
        </div>
      </div>

      {/* Featured hero */}
      {featured && (
        <div className={styles.hero}>
          <div className={styles.heroLeft}>
            <div className={styles.heroKicker}>Featured · {featured.vendor}</div>
            <h2 className={styles.heroTitle}>{featured.name}</h2>
            <div className={styles.heroBy}>
              <b>{featured.tagline}</b>
            </div>
            <p className={styles.heroDesc}>{featured.description}</p>
            <div className={styles.heroMeta}>
              <div><b>{featured.runtime}</b>runtime</div>
              <div><b>{featured.latency}</b>latency</div>
              <div><b>{featured.size}</b>size</div>
              <div><b>{featured.license}</b>license</div>
            </div>
            <div className={styles.heroActions}>
              <button className={`${styles.btn} ${styles.btnPrimary}`} onClick={() => setSelectedModel(featured)}>
                Open details
              </button>
              <button className={styles.btn} onClick={() => navigate('/studio')}>
                Use in Studio
              </button>
            </div>
          </div>
          <div className={styles.heroRight}>
            <div className={styles.rack}>
              <div className={styles.rackLcd}>{featured.name.toUpperCase()}</div>
              <div className={styles.rackStats}>
                <span>{featured.runtime}</span>
                <span className={styles.sep}>·</span>
                <span>{featured.latency}</span>
              </div>
              <div className={styles.rackMeters}>
                <div className={styles.meter}>
                  <span className={styles.meterLabel}>L</span>
                  <div className={styles.meterBar}><div className={styles.meterFill} style={{ width: '72%' }} /></div>
                </div>
                <div className={styles.meter}>
                  <span className={styles.meterLabel}>R</span>
                  <div className={styles.meterBar}><div className={styles.meterFill} style={{ width: '64%' }} /></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search */}
      <div className={styles.search}>
        <i className="fa-solid fa-magnifying-glass" />
        <input
          type="text"
          placeholder="Search models, vendors, or capabilities..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <button className={styles.searchClear} onClick={() => setSearchQuery('')}>
            <i className="fa-solid fa-xmark" />
          </button>
        )}
        <span className={styles.searchHint}>
          <span className={styles.searchKbd}>⌘ K</span>
        </span>
      </div>

      {/* Category pills */}
      <div className={styles.pills}>
        {CATEGORIES.map(c => {
          const n = c.id === 'all'
            ? MODEL_CATALOG.length
            : MODEL_CATALOG.filter(m => m.category === c.id).length;
          return (
            <button
              key={c.id}
              className={`${styles.pill} ${activeCategory === c.id ? styles.pillActive : ''}`}
              onClick={() => setActiveCategory(c.id)}
            >
              <span>{c.label}</span>
              <span className={styles.pillCount}>{n}</span>
            </button>
          );
        })}
        <div className={styles.pillDivider} />
        {FILTERS.map(f => (
          <button
            key={f.id}
            className={`${styles.pill} ${activeFilter === f.id ? styles.pillActive : ''}`}
            onClick={() => setActiveFilter(f.id)}
          >
            <i className={f.icon} />
            <span>{f.label}</span>
          </button>
        ))}
      </div>

      {/* Grid title */}
      <div className={styles.gridTitle}>
        <h2>Catalog</h2>
        <span className={styles.gridSub}>{filtered.length} {filtered.length === 1 ? 'model' : 'models'}</span>
        <span className={styles.spacer} />
      </div>

      {/* Grid */}
      {filtered.length === 0 ? (
        <div className={styles.empty}>
          <i className="fa-solid fa-folder-open" />
          <p>No models match your filters.</p>
        </div>
      ) : (
        <div className={styles.grid}>
          {filtered.map(m => (
            <article key={m.id} className={styles.card}>
              <div className={styles.cardCover}>
                <div className={styles.cardLogo}>{m.name.split(' ').map(w => w[0]).join('').slice(0, 2)}</div>
                <div className={styles.cardBadges}>
                  {m.isNew && <span className={`${styles.cardBadge} ${styles.cardBadgeNew}`}>New</span>}
                  {m.featured && <span className={`${styles.cardBadge} ${styles.cardBadgeFeat}`}>Featured</span>}
                </div>
                <div className={`${styles.cardPrice} ${m.isFree ? styles.cardPriceFree : ''}`}>
                  {m.isFree ? 'Free' : 'Licensed'}
                </div>
              </div>
              <div className={styles.cardBody}>
                <div className={styles.cardRow}>
                  <h3 className={styles.cardName}>{m.name}</h3>
                  <span className={styles.cardType}>{m.category}</span>
                </div>
                <div className={styles.cardBy}>
                  by <b>{m.vendor}</b>
                </div>
                <p className={styles.cardDesc}>{m.description}</p>
              </div>
              <div className={styles.cardStats}>
                <span><b>{m.runtime}</b></span>
                <span>{m.latency}</span>
                <span>{m.size}</span>
              </div>
              <div className={styles.cardActions}>
                <button className={styles.cardBtn} onClick={() => setSelectedModel(m)}>
                  Details
                </button>
                <button className={`${styles.cardBtn} ${styles.cardBtnPrimary}`} onClick={() => navigate('/studio')}>
                  Use
                </button>
              </div>
            </article>
          ))}
        </div>
      )}

      {/* Detail drawer (basic modal) */}
      {selectedModel && (
        <div className={styles.modalOverlay} onClick={() => setSelectedModel(null)}>
          <div className={styles.modal} onClick={e => e.stopPropagation()}>
            <div className={styles.modalHead}>
              <div>
                <div className={styles.modalKicker}>{selectedModel.vendor}</div>
                <h2 className={styles.modalTitle}>{selectedModel.name}</h2>
              </div>
              <button className={styles.modalClose} onClick={() => setSelectedModel(null)}>
                <i className="fa-solid fa-xmark" />
              </button>
            </div>
            <p className={styles.modalTagline}>{selectedModel.tagline}</p>
            <p className={styles.modalDesc}>{selectedModel.description}</p>
            <div className={styles.modalMeta}>
              <div className={styles.modalMetaRow}><span>Category</span><b>{selectedModel.category}</b></div>
              <div className={styles.modalMetaRow}><span>Runtime</span><b>{selectedModel.runtime}</b></div>
              <div className={styles.modalMetaRow}><span>Latency</span><b>{selectedModel.latency}</b></div>
              <div className={styles.modalMetaRow}><span>Size</span><b>{selectedModel.size}</b></div>
              <div className={styles.modalMetaRow}><span>License</span><b>{selectedModel.license}</b></div>
            </div>
            <div className={styles.modalActions}>
              <button className={`${styles.btn} ${styles.btnPrimary}`} onClick={() => { setSelectedModel(null); navigate('/studio'); }}>
                Use in Studio
              </button>
              <button className={styles.btn} onClick={() => setSelectedModel(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Models;
