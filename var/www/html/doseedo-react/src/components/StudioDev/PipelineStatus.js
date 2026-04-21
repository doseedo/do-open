import React, { useEffect, useRef, useState } from 'react';
import { subscribe } from '../../services/pipelineStatus';

/**
 * PipelineStatus — docked status bar at the bottom of the right sidebar.
 * Shows the most recent pipeline event; click to expand the last N
 * entries. Auto-collapses after idle.
 *
 * Stage/msg come from services/pipelineStatus.js. Kinds:
 *   info   blue-ish, animated pulse (in progress)
 *   ok     green, solid (done)
 *   warn   orange
 *   error  red
 *
 * The bar hides itself when there's nothing to show. Once events are
 * present it stays visible — the user can collapse/expand the log with
 * the chevron.
 */
export default function PipelineStatus() {
  const [latest, setLatest] = useState(null);
  const [all, setAll] = useState([]);
  const [expanded, setExpanded] = useState(false);
  const logRef = useRef(null);

  useEffect(() => {
    return subscribe((evt, events) => {
      if (evt) setLatest(evt);
      setAll([...events]);
    });
  }, []);

  // Auto-scroll the expanded log to the top as new events arrive (newest-first).
  useEffect(() => {
    if (expanded && logRef.current) logRef.current.scrollTop = 0;
  }, [all.length, expanded]);

  if (!latest && all.length === 0) return null;

  const shown = latest || all[all.length - 1];
  const kind = shown.kind || 'info';

  return (
    <div className="sd-pipeline-status" data-kind={kind}>
      <button
        type="button"
        className="sd-pipeline-current"
        onClick={() => setExpanded((v) => !v)}
        title="Toggle pipeline log"
      >
        <span className="sd-pipeline-dot" data-kind={kind} />
        <span className="sd-pipeline-stage">{shown.stage}</span>
        <span className="sd-pipeline-msg">{shown.msg}</span>
        <span className="sd-pipeline-toggle">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <ul className="sd-pipeline-log" ref={logRef}>
          {all.slice().reverse().map((e, i) => (
            <li key={`${e.t}-${i}`} data-kind={e.kind || 'info'}>
              <span className="sd-pipeline-log-stage">{e.stage}</span>
              <span className="sd-pipeline-log-msg">{e.msg}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
