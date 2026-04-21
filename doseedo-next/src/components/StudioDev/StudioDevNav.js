/*
 * StudioDevNav — themed left nav rail for /studio-dev.
 *
 * Mirrors the production LeftSidebar: Home / Search / Projects / Plugins /
 * DO1 / Research / What's New / More (Help / About / Feedback), plus the
 * Generation / Browse / Chat toggles that live in the collapsed toolbar
 * today. Collapsible — click the chevron to expand for labels.
 */
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const NAV_ITEMS = [
  { icon: 'fa-house',               label: 'Home',       to: '/dashboard' },
  { icon: 'fa-magnifying-glass',    label: 'Search',     to: '/search' },
  { icon: 'fa-folder-closed',       label: 'Projects',   to: '/projects' },
  { icon: 'fa-puzzle-piece',        label: 'Plugins',    to: '/plugins' },
  { icon: 'fa-bolt',                label: 'DO1',        to: '/DO1' },
];
const INFO_ITEMS = [
  { icon: 'fa-flask',     label: 'Research',  to: '/research' },
  { icon: 'fa-newspaper', label: "What's New", to: '/whats-new' },
];
const MORE_ITEMS = [
  { icon: 'fa-circle-question', label: 'Help',     to: '/help' },
  { icon: 'fa-info-circle',     label: 'About',    to: '/about' },
  { icon: 'fa-message',         label: 'Feedback', to: '/feedback' },
];

export default function StudioDevNav({
  expanded, onToggleExpanded,
  activePanel,              // 'generate' | 'browse' | 'chat' | null
  onOpenGenerate,
  onOpenBrowse,
  onOpenChat,
}) {
  const navigate = useNavigate();
  const [moreOpen, setMoreOpen] = useState(false);

  const NavItem = ({ icon, label, onClick, active }) => (
    <button
      className={`sd-nav-item ${active ? 'active' : ''}`}
      onClick={onClick}
      title={label}
    >
      <i className={`fa-solid ${icon}`} />
      {expanded && <span className="sd-nav-label">{label}</span>}
    </button>
  );

  return (
    <aside className={`sd-nav ${expanded ? 'expanded' : ''}`}>
      {/* Collapse / expand — always on top */}
      <button className="sd-nav-toggle" onClick={onToggleExpanded} title={expanded ? 'Collapse' : 'Expand'}>
        <i className={`fa-solid ${expanded ? 'fa-chevron-left' : 'fa-bars'}`} />
      </button>

      {/* Mode switch:
          COLLAPSED → tool toggles (chat / browse / generate) only
          EXPANDED  → wayfinding nav (home / search / projects / ... / more) */}
      {!expanded ? (
        <div className="sd-nav-section">
          <NavItem
            icon="fa-comments"
            label="Chat"
            onClick={onOpenChat}
            active={activePanel === 'chat'}
          />
          <NavItem
            icon="fa-folder-open"
            label="Browse MIDI"
            onClick={onOpenBrowse}
            active={activePanel === 'browse'}
          />
          <NavItem
            icon="fa-wand-magic-sparkles"
            label="Generate"
            onClick={onOpenGenerate}
            active={activePanel === 'generate'}
          />
        </div>
      ) : (
        <>
          {/* Primary nav */}
          <div className="sd-nav-section">
            {NAV_ITEMS.map((n) => (
              <NavItem key={n.to} icon={n.icon} label={n.label} onClick={() => navigate(n.to)} />
            ))}
          </div>

          <div className="sd-nav-divider" />

          {/* Info */}
          <div className="sd-nav-section">
            {INFO_ITEMS.map((n) => (
              <NavItem key={n.to} icon={n.icon} label={n.label} onClick={() => navigate(n.to)} />
            ))}
            <div style={{ position: 'relative' }}>
              <NavItem icon="fa-ellipsis" label="More" onClick={() => setMoreOpen((v) => !v)} active={moreOpen} />
              {moreOpen && (
                <div className="sd-nav-more" onMouseLeave={() => setMoreOpen(false)}>
                  {MORE_ITEMS.map((m) => (
                    <button key={m.to} className="sd-nav-more-item" onClick={() => { setMoreOpen(false); navigate(m.to); }}>
                      <i className={`fa-solid ${m.icon}`} />
                      <span>{m.label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </aside>
  );
}
