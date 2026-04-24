/*
 * sessionHistory — pure data model + localStorage persistence for a
 * git-shaped commit DAG over doseedo session state.
 *
 * Mental model (closely parallels git):
 *   - A commit = { id, parentIds[], author, timestamp, label, actionType,
 *                  snapshot } where snapshot is a pruned project state
 *     (buses, bpm, masterGain, etc. — heavy per-track metadata already
 *     stripped by sessionService.stripHeavyTrackMetadata).
 *   - A ref = a named pointer (branch name → commit id). The "HEAD" is
 *     always the commit at refs[currentBranch]; we don't store HEAD as
 *     its own field, but we cache `head` for O(1) reads.
 *   - Revert = write a new commit with parent = current HEAD and
 *     snapshot = target commit's snapshot. Old tip stays reachable via
 *     its own commit id.
 *   - Branch = create a new ref pointing at a commit, switch current-
 *     Branch to it. Old branch tip stays where it was.
 *   - Preview (not a persisted concept) = load a commit's snapshot into
 *     live state without writing a commit. UI-side flag in AppContext.
 */

const PREFIX = 'sesshist-v1:';
const SOFT_MAX_COMMITS = 120; // GC-visible ceiling; branch tips always kept
const MAX_LABEL_LEN = 96;

// ---------- author registry -----------------------------------------
// Components that know the Clerk user (e.g. AuthorBridge mounted inside
// AppProvider) call setCurrentAuthor on mount/change; the commit builder
// consults this whenever it writes. Decouples the reducer from Clerk.
let _currentAuthor = { userId: null, username: null, avatarUrl: null };
export function setCurrentAuthor(author) {
  if (!author) return;
  _currentAuthor = {
    userId: author.userId ?? null,
    username: author.username ?? null,
    avatarUrl: author.avatarUrl ?? null,
  };
}
export function getCurrentAuthor() { return _currentAuthor; }

// ---------- storage ---------------------------------------------------
export function storageKey(sessionKey) { return PREFIX + (sessionKey || 'default'); }

export function loadHistory(sessionKey) {
  try {
    const raw = localStorage.getItem(storageKey(sessionKey));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return null;
    if (!parsed.commits || typeof parsed.commits !== 'object') return null;
    return parsed;
  } catch {
    return null;
  }
}

export function saveHistory(sessionKey, history) {
  if (!sessionKey) return false;
  const key = storageKey(sessionKey);
  const write = (h) => {
    try { localStorage.setItem(key, JSON.stringify(h)); return true; }
    catch { return false; }
  };
  if (write(history)) return true;
  // Quota hit — prune aggressively and retry once.
  const pruned = pruneHistory(history, Math.floor(SOFT_MAX_COMMITS / 2));
  return write(pruned);
}

// ---------- initial + commit factories ------------------------------
export function initialHistory() {
  return {
    commits: {},
    refs: { main: null },
    head: null,
    currentBranch: 'main',
  };
}

export function makeCommitId() {
  return `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createCommit({ parentId, label, actionType, snapshot, actionPayload }) {
  return {
    id: makeCommitId(),
    parentIds: parentId ? [parentId] : [],
    author: { ...getCurrentAuthor() },
    timestamp: Date.now(),
    label: truncateLabel(label || actionType || 'edit'),
    actionType: actionType || null,
    actionPayload: actionPayload ?? null,
    snapshot,
  };
}

function truncateLabel(s) {
  if (typeof s !== 'string') return String(s);
  return s.length > MAX_LABEL_LEN ? s.slice(0, MAX_LABEL_LEN - 1) + '…' : s;
}

// Append commit to history on the CURRENT branch, advancing HEAD. Pure.
export function recordCommit(history, commit) {
  const currentBranch = history.currentBranch || 'main';
  return {
    ...history,
    commits: { ...history.commits, [commit.id]: commit },
    refs: { ...history.refs, [currentBranch]: commit.id },
    head: commit.id,
  };
}

// New branch pointing at commitId. currentBranch switches to it; HEAD
// stays on commitId (we're "checking out" the new branch's tip, which
// is the same commit it was created from).
export function createBranch(history, commitId, branchName) {
  if (!history.commits[commitId]) return history;
  const name = branchName || nextBranchName(history);
  return {
    ...history,
    refs: { ...history.refs, [name]: commitId },
    head: commitId,
    currentBranch: name,
  };
}

export function checkoutBranch(history, branchName) {
  const commitId = history.refs?.[branchName];
  if (!commitId) return history;
  return { ...history, currentBranch: branchName, head: commitId };
}

export function nextBranchName(history, base = 'branch') {
  const used = new Set(Object.keys(history.refs || {}));
  let n = 1;
  while (used.has(`${base}-${n}`)) n++;
  return `${base}-${n}`;
}

// ---------- queries --------------------------------------------------
export function listCommitsDesc(history) {
  return Object.values(history.commits || {}).sort((a, b) => b.timestamp - a.timestamp);
}

export function refsAtCommit(history, commitId) {
  if (!history?.refs) return [];
  return Object.entries(history.refs)
    .filter(([, id]) => id === commitId)
    .map(([name]) => name);
}

export function isOnCurrentBranch(history, commitId) {
  // Walk back from the current-branch tip; the commit is "on" the
  // current branch if we find it in the parent chain.
  const tip = history.refs?.[history.currentBranch];
  let id = tip;
  const seen = new Set();
  while (id && !seen.has(id)) {
    if (id === commitId) return true;
    seen.add(id);
    id = history.commits?.[id]?.parentIds?.[0] || null;
  }
  return false;
}

// ---------- GC -------------------------------------------------------
// Keep branch tips + their ancestor chains up to `max` total commits.
// Commits outside any branch ancestry are dropped first.
export function pruneHistory(history, max = SOFT_MAX_COMMITS) {
  const all = Object.keys(history.commits || {});
  if (all.length <= max) return history;

  const keep = new Set();
  const frontier = Array.from(new Set(Object.values(history.refs || {}).filter(Boolean)));
  while (frontier.length && keep.size < max) {
    const id = frontier.shift();
    if (!id || keep.has(id)) continue;
    keep.add(id);
    const c = history.commits[id];
    if (c?.parentIds) for (const p of c.parentIds) frontier.push(p);
  }
  // If we still have headroom, fill with the most-recent stragglers.
  if (keep.size < max) {
    const remaining = all
      .filter((id) => !keep.has(id))
      .sort((a, b) => (history.commits[b]?.timestamp || 0) - (history.commits[a]?.timestamp || 0));
    for (const id of remaining) {
      if (keep.size >= max) break;
      keep.add(id);
    }
  }
  const pruned = {};
  for (const id of keep) pruned[id] = history.commits[id];
  return { ...history, commits: pruned };
}

// ---------- human labels from dispatch action ------------------------
// Shapes the commit's one-line summary. We intentionally avoid leaking
// internal action shapes into the UI; labels are for humans.
export function labelForAction(actionType, payload) {
  const p = payload || {};
  switch (actionType) {
    case 'ADD_TRACK':      return `Add track: ${p.track?.name || p.track?.id || 'track'}`;
    case 'ADD_TRACKS_BULK':return `Add ${p.tracks?.length || 0} tracks`;
    case 'REMOVE_TRACK':   return `Remove track`;
    case 'REPLACE_TRACK':  return `Replace track`;
    case 'PASTE_TRACK':    return `Paste track`;
    case 'UPDATE_TRACK':   return `Edit track${p.updates?.audioUrl ? ' (audio)' : p.updates?.metadata?.versions ? ' (version)' : ''}`;
    case 'UPDATE_TRACK_MIDI_DATA': return `Edit MIDI`;
    case 'CREATE_BUS':     return `Create bus: ${p.name || p.type || 'bus'}`;
    case 'ADD_BUS':        return `Add bus`;
    case 'REMOVE_BUS':     return `Remove bus`;
    case 'CLEAR_BUS':      return `Clear bus`;
    case 'UPDATE_BPM':     return `Set BPM: ${p}`;
    default:               return actionType || 'edit';
  }
}
