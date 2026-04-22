import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { getCurrentUser } from '../../services/authService';
import * as sessionService from '../../services/sessionService';
import * as dashboardService from '../../services/dashboardService';
import styles from './Dashboard.module.css';

/* =====================================================================
 * Waveform generator + seed helpers
 *   Deterministic 0..1 pseudo-random per seed. Used to give every
 *   session card a stable waveform + track swatch across renders.
 * ===================================================================== */
const seedRand = (s) => {
  const x = Math.sin(s) * 10000;
  return x - Math.floor(x);
};

const stringHash = (str = '') => {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  }
  return Math.abs(h) || 1;
};

const TRACK_SWATCHES = ['#c94f2c', '#7a5d3a', '#2f6b4e', '#4a3d6b', '#1d4c7a', '#5c7a8a'];

const makeWave = ({ seed, bars, envPad = 0.05, envFloor = 0.35, noiseFloor = 0.55 }) => {
  const out = new Array(bars);
  for (let i = 0; i < bars; i++) {
    const t = i / bars;
    const env = envFloor + (1 - envFloor) * Math.pow(
      Math.sin(Math.PI * (t * (1 - 2 * envPad) + envPad)), 2
    );
    const noise = noiseFloor + seedRand(seed * 13 + i * 3.1) * (1 - noiseFloor);
    out[i] = (env * noise * 100).toFixed(1) + '%';
  }
  return out;
};

const Wave = ({ seed, bars = 44, envFloor = 0.35, noiseFloor = 0.55, envPad = 0.05, className }) => {
  const heights = useMemo(
    () => makeWave({ seed, bars, envFloor, noiseFloor, envPad }),
    [seed, bars, envFloor, noiseFloor, envPad]
  );
  return (
    <div className={className}>
      {heights.map((h, i) => (<i key={i} style={{ height: h }} />))}
    </div>
  );
};

/* =====================================================================
 * Session card  (Jump Back In rail)
 * ===================================================================== */
const SessionCard = ({ session, onLoad, onMore }) => {
  const seed = useMemo(() => stringHash(session.name), [session.name]);
  const swatch = TRACK_SWATCHES[seed % TRACK_SWATCHES.length];
  const tracks = session.trackCount || 0;
  const meta = `${session.daw || 'Doseedo'} · ${tracks} track${tracks === 1 ? '' : 's'} · ${session.time || '—'}`;
  const mark = (session.daw || 'DSD').slice(0, 3).toUpperCase();

  return (
    <div
      className={styles.sessCard}
      style={{ '--swatch': swatch }}
      onClick={() => onLoad(session.name)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onLoad(session.name); }}
    >
      <div className={styles.sessCover}>
        <div className={styles.sessSwatch} />
        <div className={styles.sessMark}>{mark}</div>
        <Wave seed={seed} className={styles.sessWave} />
      </div>
      <div className={styles.sessBody}>
        <div className={styles.sessName}>{session.name}</div>
        <div className={styles.sessMeta}>{meta}</div>
      </div>
      <div className={styles.sessActions}>
        <button
          className={`${styles.chip} ${styles.chipPrimary}`}
          onClick={(e) => { e.stopPropagation(); onLoad(session.name); }}
        >
          <svg width="7" height="7" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4l13 8-13 8z" /></svg>
          Resume
        </button>
        <button
          className={`${styles.chip} ${styles.chipMini}`}
          onClick={(e) => { e.stopPropagation(); onMore(session.name); }}
          title="More"
        >···</button>
      </div>
    </div>
  );
};

/* =====================================================================
 * Layout helpers
 * ===================================================================== */
const initials = (name = '') => name.split(/[\s._-]+/).filter(Boolean).slice(0, 1)
  .map(s => s[0].toUpperCase()).join('') || 'A';

const weekRangeLabel = () => {
  const now = new Date();
  const day = now.getDay();
  const mondayOffset = (day === 0 ? -6 : 1 - day);
  const monday = new Date(now); monday.setDate(now.getDate() + mondayOffset);
  const sunday = new Date(monday); sunday.setDate(monday.getDate() + 6);
  const fmt = (d) => d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return `${fmt(monday)} — ${fmt(sunday)}`;
};

// Deterministic sparkline heights per cell (so every render lines up).
const sparkHeights = (seed) => {
  return Array.from({ length: 7 }, (_, i) => {
    const r = seedRand(seed * 17 + i * 2.3);
    return Math.max(10, Math.round(r * 100));
  });
};

// A synth timecode ("03:12") for trending cards — real durations aren't in the
// community-creations API surface yet, so derive one deterministically from
// the item id so the LCD panel isn't empty. Swap for real data when available.
const synthDuration = (seed) => {
  const total = 90 + Math.floor(seedRand(seed * 29) * 300); // 90s..390s
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
};

const FILTERS = [
  { key: 'trending', label: 'Trending' },
  { key: 'new', label: 'New' },
  { key: 'most_forked', label: 'Most Forked' },
  { key: 'your_genre', label: 'Your Genre' },
];

/* =====================================================================
 * Dashboard
 * ===================================================================== */
const Dashboard = ({ onCreateNew, onLoadProject }) => {
  const { state, dispatch } = useApp();

  const [projects, setProjects] = useState([]);            // raw list for dedup in create-new
  const [recent, setRecent] = useState([]);                // Jump Back In
  const [activity, setActivity] = useState([]);            // Activity feed
  const [liveUsers, setLiveUsers] = useState([]);          // Live Now
  const [trending, setTrending] = useState([]);            // Trending
  const [weekStats, setWeekStats] = useState([]);          // Your Week (4 stats)
  const [trendingFilter, setTrendingFilter] = useState('trending');
  const [newProjectName, setNewProjectName] = useState('');
  const [showNameInput, setShowNameInput] = useState(false);

  const user = useMemo(() => getCurrentUser(), []);

  // ── Initial + ongoing fetch ───────────────────────────────────
  const reloadAll = useCallback(async () => {
    setProjects(sessionService.getProjects());
    setWeekStats(dashboardService.getWeekStats());
    try { setRecent(await dashboardService.getRecentSessions()); } catch { setRecent([]); }
    try { setActivity(await dashboardService.getActivityFeed()); } catch { setActivity([]); }
    try { setLiveUsers(await dashboardService.getLiveUsers()); } catch { setLiveUsers([]); }
  }, []);

  useEffect(() => { reloadAll(); }, [reloadAll]);

  // Trending re-fetches whenever the filter changes
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const items = await dashboardService.getTrending(trendingFilter);
        if (!cancelled) setTrending(items);
      } catch {
        if (!cancelled) setTrending([]);
      }
    })();
    return () => { cancelled = true; };
  }, [trendingFilter]);

  // ── Session create / load / rename / delete ───────────────────
  const handleCreateNew = useCallback(() => {
    let baseName = (newProjectName || '').trim() || 'Untitled';
    let projectName = baseName;
    let counter = 2;
    while (projects.includes(projectName)) {
      projectName = `${baseName} (${counter})`;
      counter++;
    }
    dispatch({ type: 'RESET_SESSION', payload: { projectName } });
    sessionService.setActiveProject(projectName);
    sessionService.saveSession(projectName, { ...state, projectName, buses: [] });
    setNewProjectName('');
    setShowNameInput(false);
    reloadAll();
    if (onCreateNew) onCreateNew(projectName);
  }, [newProjectName, projects, state, dispatch, onCreateNew, reloadAll]);

  const handleLoadProject = useCallback((projectName) => {
    const sessionData = sessionService.loadSession(projectName);
    if (!sessionData) {
      alert('Could not load project');
      return;
    }
    sessionService.setActiveProject(projectName);
    dispatch({ type: 'LOAD_SESSION', payload: sessionData.state });
    if (onLoadProject) onLoadProject(projectName);
  }, [dispatch, onLoadProject]);

  // Minimal more-menu (rename / delete) — the original ProjectCard kebab
  // lived on the table rows; those rows are gone. Prompts keep the
  // functionality reachable until a proper menu component lands.
  const handleMore = useCallback((projectName) => {
    const action = window.prompt(
      `Project "${projectName}" — type "rename" or "delete" (or Cancel):`,
      ''
    );
    if (!action) return;
    const a = action.trim().toLowerCase();
    if (a === 'delete') {
      if (window.confirm(`Delete "${projectName}"? This cannot be undone.`)) {
        sessionService.deleteProject(projectName);
        reloadAll();
      }
    } else if (a === 'rename') {
      const newName = window.prompt('New name:', projectName);
      if (newName && newName.trim() && newName !== projectName && !projects.includes(newName.trim())) {
        sessionService.renameProject(projectName, newName.trim());
        reloadAll();
      }
    }
  }, [projects, reloadAll]);

  // ── Derived values ────────────────────────────────────────────
  const totalSessions = projects.length;
  const weekStat = (i) => weekStats[i] || { value: '—', label: '' };
  const sessionsEdited = parseInt(weekStat(0).value, 10) || 0;
  const stemsGenerated = parseInt(weekStat(1).value, 10) || 0;
  const collabsActive = parseInt(weekStat(2).value, 10) || 0;
  const studioTime = weekStat(3).value;

  const liveCount = liveUsers.filter(u => u.state === 'live').length;

  return (
    <div className={styles.dashboard}>

      {/* ===================== Top bar ===================== */}
      <div className={styles.topbar}>
        <span>Dashboard</span>
        <span className={styles.sep}>/</span>
        <span><strong>Home</strong></span>
        <span className={styles.sep}>·</span>
        <span>{totalSessions} session{totalSessions === 1 ? '' : 's'}</span>
        <div className={styles.spacer} />
        <span>engine · <strong>ready</strong></span>
        <span className={styles.sep}>·</span>
        <span>sync · <strong>up to date</strong></span>
        <span className={styles.sep}>·</span>
        <span className={styles.kbd}>⌘K</span>
      </div>

      <div className={styles.scroll}>

        {showNameInput && (
          <div className={styles.sessionInputContainer}>
            <input
              type="text"
              placeholder="Session name (optional) — Enter to create, Esc to cancel"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreateNew();
                if (e.key === 'Escape') { setNewProjectName(''); setShowNameInput(false); }
              }}
              className={styles.sessionInput}
              autoFocus
            />
          </div>
        )}

        {/* ===================== Jump Back In ===================== */}
        <section className={styles.sect}>
          <div className={styles.sectHead}>
            <h2 className={styles.sectTitle}>Jump Back In</h2>
            <span className={styles.sectCount}>last touched</span>
            <div className={styles.sectSpacer} />
          </div>

          <div className={styles.jbi}>
            <button
              type="button"
              className={`${styles.sessCard} ${styles.newCard}`}
              onClick={() => setShowNameInput(true)}
            >
              <div className={styles.newCardPlus}>＋</div>
              <div className={styles.newCardTitle}>Start a new session</div>
              <div className={styles.newCardSub}>blank · or from a memo</div>
            </button>

            {recent.map((s) => (
              <SessionCard
                key={s.id || s.name}
                session={s}
                onLoad={handleLoadProject}
                onMore={handleMore}
              />
            ))}
          </div>
        </section>

        {/* ===================== Activity + Live Now ===================== */}
        <section className={styles.midrow}>

          <div className={styles.panel}>
            <div className={styles.panelHead}>
              <strong>Activity</strong>
              <span>community · last 24h</span>
            </div>
            <div className={styles.panelBody}>
              {activity.length === 0 ? (
                <div className={styles.panelEmpty}>No community activity yet</div>
              ) : (
                activity.slice(0, 5).map((a, i) => {
                  const dotCls = styles[`actDot${i % 6}`] || '';
                  return (
                    <div className={styles.act} key={a.id || i}>
                      <div className={`${styles.actDot} ${dotCls}`} />
                      <div><b>{a.who}</b> {a.action}</div>
                      <div className={styles.actTime}>{a.time}</div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          <div className={styles.panel}>
            <div className={styles.panelHead}>
              <span className={`${styles.statusDot} ${liveCount === 0 ? styles.statusDotIdle : ''}`} />
              <strong>Live Now</strong>
              <span>{liveCount} · collaborators</span>
            </div>
            <div className={`${styles.panelBody} ${styles.live}`}>
              <div className={styles.liveBody}>
                <div className={styles.livePings}>
                  {Array.from({ length: 5 }, (_, i) => {
                    const pct = Math.round(seedRand(i + 1) * 100);
                    return (
                      <div
                        className={styles.livePing}
                        key={i}
                        style={{ background: `linear-gradient(90deg, var(--wb-surface-2) ${pct}%, var(--wb-bg) ${pct}%)` }}
                      />
                    );
                  })}
                </div>

                {liveUsers.length === 0 ? (
                  <div className={styles.panelEmpty}>
                    No collaborators online
                  </div>
                ) : (
                  <div className={styles.liveList}>
                    {liveUsers.map((u) => {
                      const seed = stringHash(u.name || '');
                      const color = TRACK_SWATCHES[seed % TRACK_SWATCHES.length];
                      return (
                        <div className={styles.liveRow} key={u.name}>
                          <div className={styles.liveAvatar} style={{ background: color }}>{initials(u.name)}</div>
                          <div>
                            <div className={styles.liveName}>{u.name}</div>
                            <div className={styles.liveWhere}>{u.where || ''}</div>
                          </div>
                          <div className={`${styles.liveState} ${u.state === 'live' ? styles.liveStateOn : ''}`}>
                            {u.state === 'live' ? '● live' : 'idle'}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className={styles.liveInvite}>
                <span>Need another set of ears?</span>
                <button className={`${styles.chip} ${styles.chipMini}`}>Invite +</button>
              </div>
            </div>
          </div>

        </section>

        {/* ===================== Trending ===================== */}
        <section className={styles.sect}>
          <div className={styles.sectHead}>
            <h2 className={styles.sectTitle}>Trending in doseedo</h2>
            <span className={styles.sectCount}>this week</span>
            <div className={styles.sectSpacer} />
            <div className={styles.filters}>
              {FILTERS.map((f) => (
                <button
                  key={f.key}
                  className={`${styles.filter} ${trendingFilter === f.key ? styles.filterActive : ''}`}
                  onClick={() => setTrendingFilter(f.key)}
                >{f.label}</button>
              ))}
            </div>
          </div>

          {trending.length === 0 ? (
            <div className={styles.panelEmpty} style={{ border: '1px dashed var(--wb-rule)' }}>
              Nothing trending yet — be the first to publish.
            </div>
          ) : (
            <div className={styles.trending}>
              {trending.slice(0, 3).map((t, i) => {
                const seed = stringHash(t.id || t.name || String(i));
                const swatch = TRACK_SWATCHES[seed % TRACK_SWATCHES.length];
                const genre = Array.isArray(t.tags) && t.tags.length > 0 ? t.tags[0] : (t.daw || '—');
                const lcd = synthDuration(seed);
                return (
                  <article className={styles.trend} style={{ '--swatch': swatch }} key={t.id || i}>
                    <div className={styles.trendHead}>
                      <span className={styles.trendRank}>{String(i + 1).padStart(2, '0')}</span>
                      <h3 className={styles.trendTitle}>{t.name}</h3>
                      <span className={styles.trendGenre}>{genre}</span>
                    </div>
                    <div className={styles.trendAuthor}>by {t.creator || 'anonymous'}</div>
                    <Wave seed={seed} bars={80} envFloor={0.30} noiseFloor={0.50} envPad={0.025} className={styles.trendWave} />
                    <div className={styles.trendFoot}>
                      <span className={styles.lcd}>{lcd}</span>
                      <span>▲ {t.plays || 0} · ⎘ {t.forks || 0}</span>
                      <span className={styles.trendActions}>
                        <button>Play</button>
                        <button>Fork</button>
                      </span>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        {/* ===================== Your Week ===================== */}
        <section className={styles.week}>
          <div className={styles.weekLabel}>Your Week · {weekRangeLabel()}</div>

          <WeekCell
            label="Sessions edited"
            value={sessionsEdited}
            note={sessionsEdited > 0 ? `${totalSessions} total on record` : 'No edits this week yet'}
            sparkSeed={1}
            peakIdx={3}
          />

          <WeekCell
            label="Stems generated"
            value={stemsGenerated}
            note={stemsGenerated > 0 ? `across ${sessionsEdited} session${sessionsEdited === 1 ? '' : 's'}` : '—'}
            sparkSeed={2}
            peakIdx={5}
          />

          <WeekCell
            label="Collaborators active"
            value={collabsActive}
            valueEm={collabsActive === 0 ? '/ invite someone' : null}
            note={collabsActive === 0 ? 'Solo week — share a session to start.' : `${collabsActive} across your sessions`}
            sparkSeed={3}
            muteAll={collabsActive === 0}
          />

          <WeekCell
            label="Time in studio"
            value={studioTime || '0m'}
            note="estimate · 30m per session"
            sparkSeed={4}
            peakIdx={5}
          />

          <div className={styles.wcell}>
            <span className={styles.wcellK}>Storage · {user?.subscriptionStatus || 'Free'} plan</span>
            <span className={styles.wcellV}>{totalSessions}<small>{' sessions'}</small></span>
            <div className={styles.wcellMeter}>
              <div
                className={styles.wcellMeterFill}
                style={{ width: `calc(${Math.min(100, totalSessions * 2)}% - 2px)` }}
              />
              <div className={styles.wcellMeterTicks} />
            </div>
            <div className={styles.wcellMeterFoot}>
              <span>{totalSessions} session{totalSessions === 1 ? '' : 's'} · {stemsGenerated} stem{stemsGenerated === 1 ? '' : 's'}</span>
              <a href="/plans">Upgrade →</a>
            </div>
          </div>

        </section>

      </div>
    </div>
  );
};

/* =====================================================================
 * Your Week — cell with sparkline + day row
 * ===================================================================== */
const WeekCell = ({ label, value, valueEm, note, sparkSeed, peakIdx, muteAll }) => {
  const heights = useMemo(() => sparkHeights(sparkSeed), [sparkSeed]);
  return (
    <div className={styles.wcell}>
      <span className={styles.wcellK}>{label}</span>
      <span className={styles.wcellV}>{value}{valueEm && <em>{valueEm}</em>}</span>
      <span className={styles.wcellNote}>{note}</span>
      <div className={styles.wcellSpark}>
        {heights.map((h, i) => {
          const cls = muteAll ? styles.sparkMute : (i === peakIdx ? styles.sparkWarm : '');
          return <i key={i} style={{ height: `${muteAll ? 10 : h}%` }} className={cls} />;
        })}
      </div>
      <div className={styles.wcellDays}>
        <span>M</span><span>T</span><span>W</span><span>T</span><span>F</span><span><b>S</b></span><span>S</span>
      </div>
    </div>
  );
};

export default Dashboard;
