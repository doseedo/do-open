/**
 * StructureView — Transparent overlay for structure mode.
 * Supports:
 * - Panel resize with proportional scaling (snapshots at drag start)
 * - Panel-to-panel drag swap (each takes the other's bounds)
 * - Child component drag swap (positions swap within a section)
 * - Visual row drag swap (entire rows of components swap vertically)
 */

import React, { useState, useCallback, useMemo, useRef } from 'react';
import {
  buildSectionMap, detectGridStructure,
  adjustAdjacentPanels, applySectionUpdates,
  reflowSectionChildren,
  adjustAdjacentChildren, clampChildToPanel,
  MIN_PANEL_WIDTH, MIN_PANEL_HEIGHT,
  detectChildRows, swapSectionPositions, swapChildPositions, swapChildRowPositions,
} from './sectionDetector';

// ── Type icons ──────────────────────────────────────────────────────────

const TYPE_ICONS = {
  knob: 'fa-solid fa-circle-dot',
  slider: 'fa-solid fa-sliders',
  button: 'fa-solid fa-toggle-on',
  label: 'fa-solid fa-font',
  led: 'fa-solid fa-circle',
  dropdown: 'fa-solid fa-caret-down',
  image: 'fa-solid fa-image',
  panel: 'fa-solid fa-square',
  meter: 'fa-solid fa-signal',
  waveform: 'fa-solid fa-wave-square',
  'xy-pad': 'fa-solid fa-arrows-up-down-left-right',
};

const DRAG_DEADZONE = 5;

// ── Edge/Corner resize handles ──────────────────────────────────────────

const HANDLE_SIZE = 8;
const EDGE_THICKNESS = 6;

function PanelResizeHandles({ panel, onResizeStart }) {
  const handles = [
    { id: 'nw', cursor: 'nw-resize', style: { left: -HANDLE_SIZE / 2, top: -HANDLE_SIZE / 2, width: HANDLE_SIZE, height: HANDLE_SIZE } },
    { id: 'ne', cursor: 'ne-resize', style: { right: -HANDLE_SIZE / 2, top: -HANDLE_SIZE / 2, width: HANDLE_SIZE, height: HANDLE_SIZE } },
    { id: 'sw', cursor: 'sw-resize', style: { left: -HANDLE_SIZE / 2, bottom: -HANDLE_SIZE / 2, width: HANDLE_SIZE, height: HANDLE_SIZE } },
    { id: 'se', cursor: 'se-resize', style: { right: -HANDLE_SIZE / 2, bottom: -HANDLE_SIZE / 2, width: HANDLE_SIZE, height: HANDLE_SIZE } },
    { id: 'n', cursor: 'n-resize', style: { left: HANDLE_SIZE, right: HANDLE_SIZE, top: -EDGE_THICKNESS / 2, height: EDGE_THICKNESS } },
    { id: 's', cursor: 's-resize', style: { left: HANDLE_SIZE, right: HANDLE_SIZE, bottom: -EDGE_THICKNESS / 2, height: EDGE_THICKNESS } },
    { id: 'w', cursor: 'w-resize', style: { left: -EDGE_THICKNESS / 2, top: HANDLE_SIZE, bottom: HANDLE_SIZE, width: EDGE_THICKNESS } },
    { id: 'e', cursor: 'e-resize', style: { right: -EDGE_THICKNESS / 2, top: HANDLE_SIZE, bottom: HANDLE_SIZE, width: EDGE_THICKNESS } },
  ];

  return handles.map(h => (
    <div
      key={h.id}
      onMouseDown={(e) => { e.stopPropagation(); onResizeStart(panel, h.id, e); }}
      style={{
        position: 'absolute',
        ...h.style,
        cursor: h.cursor,
        zIndex: 10,
        ...(h.id.length === 2 ? {
          background: 'rgba(186, 156, 255, 0.9)',
          border: '1px solid #fff',
          borderRadius: 2,
        } : {
          background: 'transparent',
        }),
      }}
    />
  ));
}

// ── Child component box ─────────────────────────────────────────────────

function ChildResizeHandles({ onResizeStart }) {
  const S = 6;
  const E = 4;
  const handles = [
    { id: 'se', cursor: 'se-resize', style: { right: -S / 2, bottom: -S / 2, width: S, height: S } },
    { id: 'sw', cursor: 'sw-resize', style: { left: -S / 2, bottom: -S / 2, width: S, height: S } },
    { id: 'ne', cursor: 'ne-resize', style: { right: -S / 2, top: -S / 2, width: S, height: S } },
    { id: 'nw', cursor: 'nw-resize', style: { left: -S / 2, top: -S / 2, width: S, height: S } },
    { id: 'e', cursor: 'e-resize', style: { right: -E / 2, top: S, bottom: S, width: E } },
    { id: 'w', cursor: 'w-resize', style: { left: -E / 2, top: S, bottom: S, width: E } },
    { id: 's', cursor: 's-resize', style: { left: S, right: S, bottom: -E / 2, height: E } },
    { id: 'n', cursor: 'n-resize', style: { left: S, right: S, top: -E / 2, height: E } },
  ];
  return handles.map(h => (
    <div
      key={h.id}
      onMouseDown={(e) => { e.stopPropagation(); onResizeStart(h.id, e); }}
      style={{
        position: 'absolute', ...h.style, cursor: h.cursor, zIndex: 10,
        ...(h.id.length === 2 ? {
          background: 'rgba(186, 156, 255, 0.8)',
          border: '1px solid #fff',
          borderRadius: 2,
        } : { background: 'transparent' }),
      }}
    />
  ));
}

function ChildBox({ comp, isSelected, isDropTarget, onMouseDown, onResizeStart, panelBounds }) {
  const icon = TYPE_ICONS[comp.type] || 'fa-solid fa-cube';
  const isLabel = comp.type === 'label';
  const isLed = comp.type === 'led';

  const relX = comp.x - (panelBounds?.x || 0);
  const relY = comp.y - (panelBounds?.y || 0);

  return (
    <div
      onMouseDown={(e) => { e.stopPropagation(); onMouseDown(comp.id, e); }}
      style={{
        position: 'absolute',
        left: relX,
        top: relY,
        width: comp.width,
        height: comp.height,
        border: isDropTarget
          ? '2px solid rgba(100, 255, 150, 0.9)'
          : isSelected
            ? '2px solid rgba(186, 156, 255, 0.9)'
            : '1px solid rgba(186, 156, 255, 0.25)',
        background: isDropTarget
          ? 'rgba(100, 255, 150, 0.15)'
          : isSelected
            ? 'rgba(186, 156, 255, 0.12)'
            : 'transparent',
        borderRadius: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 3,
        fontSize: isLed ? 7 : isLabel ? 8 : 9,
        color: isDropTarget
          ? 'rgba(100, 255, 150, 0.9)'
          : isSelected ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.5)',
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        cursor: 'grab',
        zIndex: 3,
        transition: 'border-color 0.1s, background 0.1s',
        ...(isDropTarget && { boxShadow: '0 0 8px rgba(100, 255, 150, 0.3)' }),
      }}
    >
      <i className={icon} style={{ fontSize: isLed ? 6 : 8, flexShrink: 0 }} />
      {comp.label && comp.width > 30 && (
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{comp.label}</span>
      )}
      {isSelected && !isDropTarget && (
        <ChildResizeHandles onResizeStart={(handle, e) => onResizeStart(comp.id, handle, e)} />
      )}
    </div>
  );
}

// ── Section panel box ───────────────────────────────────────────────────

function SectionBox({ section, isSelected, isDropTarget, onMouseDown, onResizeStart, children }) {
  const { panel, label } = section;

  return (
    <div
      onMouseDown={(e) => { onMouseDown(panel.id, e); }}
      style={{
        position: 'absolute',
        left: panel.x,
        top: panel.y,
        width: panel.width,
        height: panel.height,
        border: isDropTarget
          ? '2px solid rgba(100, 255, 150, 0.8)'
          : isSelected
            ? '2px solid rgba(186, 156, 255, 0.8)'
            : '2px dashed rgba(186, 156, 255, 0.35)',
        background: isDropTarget
          ? 'rgba(100, 255, 150, 0.08)'
          : isSelected
            ? 'rgba(186, 156, 255, 0.06)'
            : 'transparent',
        borderRadius: 6,
        cursor: 'grab',
        zIndex: 1,
        transition: 'border-color 0.15s, background 0.15s',
        ...(isDropTarget && { boxShadow: '0 0 12px rgba(100, 255, 150, 0.2)' }),
      }}
    >
      {/* Section label badge */}
      <div style={{
        position: 'absolute',
        top: -1,
        left: 8,
        padding: '2px 8px',
        background: isDropTarget
          ? 'rgba(100, 255, 150, 0.25)'
          : isSelected ? 'rgba(186, 156, 255, 0.25)' : 'rgba(186, 156, 255, 0.12)',
        borderRadius: '0 0 4px 4px',
        fontSize: 10,
        fontWeight: 600,
        color: isDropTarget
          ? 'rgba(100, 255, 150, 0.9)'
          : isSelected ? '#d4c0ff' : 'rgba(255,255,255,0.6)',
        letterSpacing: '0.05em',
        zIndex: 5,
        pointerEvents: 'none',
      }}>
        {label}
      </div>

      {/* Dimensions */}
      <div style={{
        position: 'absolute',
        bottom: 2,
        right: 6,
        fontSize: 8,
        color: 'rgba(255,255,255,0.25)',
        pointerEvents: 'none',
      }}>
        {Math.round(panel.width)}x{Math.round(panel.height)}
      </div>

      {/* Resize handles (only when selected, not during drop target) */}
      {isSelected && !isDropTarget && <PanelResizeHandles panel={panel} onResizeStart={onResizeStart} />}

      {children}
    </div>
  );
}

// ── Structure Tree (sidebar) ────────────────────────────────────────────

export function StructureTreePanel({ sections, orphans, selectedIds, onSelect, pluginConfig }) {
  return (
    <div style={{
      padding: '8px 6px',
      fontSize: 11,
      color: 'rgba(255,255,255,0.8)',
      overflowY: 'auto',
      maxHeight: '100%',
    }}>
      <div style={{
        fontSize: 10, fontWeight: 600, marginBottom: 6,
        color: 'rgba(255,255,255,0.5)', letterSpacing: '0.06em',
      }}>
        STRUCTURE
      </div>
      <div style={{
        fontSize: 9, color: 'rgba(255,255,255,0.3)', marginBottom: 8,
      }}>
        {pluginConfig?.name || 'Plugin'} ({pluginConfig?.width}x{pluginConfig?.height})
      </div>

      {sections.map((sec) => {
        const isPanelSelected = selectedIds.includes(sec.panel.id);

        return (
          <div key={sec.panel.id} style={{ marginBottom: 4 }}>
            <div
              onClick={() => onSelect(sec.panel.id, false)}
              style={{
                padding: '3px 6px',
                borderRadius: 4,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                background: isPanelSelected ? 'rgba(186, 156, 255, 0.15)' : 'transparent',
                fontWeight: 600,
              }}
            >
              <i className="fa-solid fa-square" style={{ fontSize: 8, color: 'rgba(186, 156, 255, 0.6)' }} />
              <span>{sec.label}</span>
              <span style={{ marginLeft: 'auto', fontSize: 8, color: 'rgba(255,255,255,0.3)' }}>
                {sec.children.length}
              </span>
            </div>

            {sec.children.map(child => {
              const isChildSelected = selectedIds.includes(child.id);
              return (
                <div
                  key={child.id}
                  onClick={(e) => { e.stopPropagation(); onSelect(child.id, e.shiftKey); }}
                  style={{
                    padding: '2px 6px 2px 20px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                    fontSize: 10,
                    borderRadius: 3,
                    background: isChildSelected ? 'rgba(186, 156, 255, 0.1)' : 'transparent',
                    color: isChildSelected ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.5)',
                  }}
                >
                  <i className={TYPE_ICONS[child.type] || 'fa-solid fa-cube'} style={{ fontSize: 8, width: 12, textAlign: 'center' }} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {child.label || child.type}
                  </span>
                </div>
              );
            })}
          </div>
        );
      })}

      {orphans.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 9, fontWeight: 600, color: 'rgba(255,170,0,0.7)', marginBottom: 4 }}>
            <i className="fa-solid fa-triangle-exclamation" style={{ marginRight: 4 }} />
            ORPHANS ({orphans.length})
          </div>
          {orphans.map(c => (
            <div
              key={c.id}
              onClick={() => onSelect(c.id, false)}
              style={{
                padding: '2px 6px 2px 12px', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 5,
                fontSize: 10, color: 'rgba(255,170,0,0.5)',
              }}
            >
              <i className={TYPE_ICONS[c.type] || 'fa-solid fa-cube'} style={{ fontSize: 8 }} />
              <span>{c.label || c.type}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main StructureView overlay ──────────────────────────────────────────

// ── Alignment guide detection ─────────────────────────────────────────────
const SNAP_THRESHOLD = 6;

function detectAlignmentGuides(movingBounds, allSections, excludeId) {
  const guides = [];
  const snaps = { x: null, y: null };
  const movingEdges = {
    left: movingBounds.x,
    right: movingBounds.x + movingBounds.width,
    centerX: movingBounds.x + movingBounds.width / 2,
    top: movingBounds.y,
    bottom: movingBounds.y + movingBounds.height,
    centerY: movingBounds.y + movingBounds.height / 2,
  };

  let bestDistX = SNAP_THRESHOLD + 1, bestDistY = SNAP_THRESHOLD + 1;

  for (const sec of allSections) {
    if (sec.panel.id === excludeId) continue;
    const p = sec.panel;
    const targets = {
      left: p.x, right: p.x + p.width, centerX: p.x + p.width / 2,
      top: p.y, bottom: p.y + p.height, centerY: p.y + p.height / 2,
    };

    // Horizontal alignment (vertical guide lines)
    for (const mKey of ['left', 'right', 'centerX']) {
      for (const tKey of ['left', 'right', 'centerX']) {
        const dist = Math.abs(movingEdges[mKey] - targets[tKey]);
        if (dist < SNAP_THRESHOLD && dist < bestDistX) {
          bestDistX = dist;
          snaps.x = { edge: mKey, snapTo: targets[tKey] };
          guides.push({ type: 'vertical', position: targets[tKey] });
        }
      }
    }

    // Vertical alignment (horizontal guide lines)
    for (const mKey of ['top', 'bottom', 'centerY']) {
      for (const tKey of ['top', 'bottom', 'centerY']) {
        const dist = Math.abs(movingEdges[mKey] - targets[tKey]);
        if (dist < SNAP_THRESHOLD && dist < bestDistY) {
          bestDistY = dist;
          snaps.y = { edge: mKey, snapTo: targets[tKey] };
          guides.push({ type: 'horizontal', position: targets[tKey] });
        }
      }
    }
  }

  return { guides, snaps };
}

const StructureView = ({
  config,
  components,
  selectedIds,
  onSelect,
  onDeselect,
  onUpdateComponents,
  onCommitResize,
}) => {
  const canvasRef = useRef(null);
  const resizeStateRef = useRef(null);
  const dropTargetRef = useRef(null);
  const [dropTarget, setDropTarget] = useState(null);
  const [dragGhost, setDragGhost] = useState(null);
  const [snapGuides, setSnapGuides] = useState([]);

  const { sections, orphans } = useMemo(
    () => buildSectionMap(components),
    [components]
  );

  const getCanvasPos = useCallback((e) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  // ── Panel resize logic (existing) ──────────────────────────────────────

  const handleResizeStart = useCallback((panel, handle, e) => {
    e.preventDefault();

    const snapshotSections = sections.map(s => ({
      ...s,
      panel: { ...s.panel },
      children: s.children.map(c => ({ ...c })),
    }));
    const snapshotGrid = detectGridStructure(snapshotSections);
    const snapshotComponents = components.map(c => ({ ...c }));

    const startBounds = { x: panel.x, y: panel.y, width: panel.width, height: panel.height };
    const section = snapshotSections.find(s => s.panel.id === panel.id);

    resizeStateRef.current = {
      panelId: panel.id,
      handle,
      startX: e.clientX,
      startY: e.clientY,
      startBounds,
      section,
      snapshotSections,
      snapshotGrid,
      snapshotComponents,
    };

    const handleMouseMove = (me) => {
      const state = resizeStateRef.current;
      if (!state) return;

      const dx = me.clientX - state.startX;
      const dy = me.clientY - state.startY;
      const sb = state.startBounds;
      let newBounds = { ...sb };

      if (state.handle.includes('e')) newBounds.width = Math.max(MIN_PANEL_WIDTH, sb.width + dx);
      if (state.handle.includes('w')) {
        newBounds.width = Math.max(MIN_PANEL_WIDTH, sb.width - dx);
        newBounds.x = sb.x + (sb.width - newBounds.width);
      }
      if (state.handle.includes('s')) newBounds.height = Math.max(MIN_PANEL_HEIGHT, sb.height + dy);
      if (state.handle.includes('n')) {
        newBounds.height = Math.max(MIN_PANEL_HEIGHT, sb.height - dy);
        newBounds.y = sb.y + (sb.height - newBounds.height);
      }

      // Snap to alignment guides
      const { guides, snaps } = detectAlignmentGuides(newBounds, state.snapshotSections, state.panelId);
      setSnapGuides(guides);
      if (snaps.x) {
        const edge = snaps.x.edge;
        if (edge === 'right') newBounds.width = snaps.x.snapTo - newBounds.x;
        else if (edge === 'left') { const right = newBounds.x + newBounds.width; newBounds.x = snaps.x.snapTo; newBounds.width = right - newBounds.x; }
        else if (edge === 'centerX') newBounds.x = snaps.x.snapTo - newBounds.width / 2;
      }
      if (snaps.y) {
        const edge = snaps.y.edge;
        if (edge === 'bottom') newBounds.height = snaps.y.snapTo - newBounds.y;
        else if (edge === 'top') { const bottom = newBounds.y + newBounds.height; newBounds.y = snaps.y.snapTo; newBounds.height = bottom - newBounds.y; }
        else if (edge === 'centerY') newBounds.y = snaps.y.snapTo - newBounds.height / 2;
      }

      const reflowedChildren = reflowSectionChildren(state.section, newBounds);

      const adjacentUpdates = adjustAdjacentPanels(
        state.section, newBounds, state.snapshotSections, state.snapshotGrid
      );

      const allUpdates = [{
        panelId: state.panelId,
        newPanelBounds: newBounds,
        updatedChildren: reflowedChildren,
      }];

      for (const adj of adjacentUpdates) {
        const adjReflowed = reflowSectionChildren(adj.section, adj.newBounds);
        allUpdates.push({
          panelId: adj.section.panel.id,
          newPanelBounds: adj.newBounds,
          updatedChildren: adjReflowed,
        });
      }

      // Direct above-panel shift — bypass adjustAdjacentPanels for north drag
      const deltaY = newBounds.y - sb.y;
      if (deltaY !== 0) {
        for (const sec of state.snapshotSections) {
          if (sec.panel.id === state.panelId) continue;
          // Already handled by adjustAdjacentPanels? Check if in allUpdates
          const alreadyHandled = allUpdates.some(u => u.panelId === sec.panel.id);
          const secBottom = sec.panel.y + sec.panel.height;
          if (secBottom <= sb.y && !alreadyHandled) {
            const shiftedBounds = { ...sec.panel, y: sec.panel.y + deltaY };
            const shifted = reflowSectionChildren(sec, shiftedBounds);
            allUpdates.push({
              panelId: sec.panel.id,
              newPanelBounds: shiftedBounds,
              updatedChildren: shifted,
            });
          }
        }
      }

      const updated = applySectionUpdates(state.snapshotComponents, allUpdates);
      onUpdateComponents(updated);
    };

    const handleMouseUp = () => {
      resizeStateRef.current = null;
      setSnapGuides([]);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      if (onCommitResize) onCommitResize();
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [sections, components, onUpdateComponents, onCommitResize]);

  // ── Child component resize (with neighbor adjustment) ───────────────────

  const handleChildResizeStart = useCallback((childId, handle, e) => {
    e.preventDefault();
    e.stopPropagation();

    const comp = components.find(c => c.id === childId);
    if (!comp) return;

    // Snapshot at drag start (same pattern as panel resize)
    const snapshotSections = sections.map(s => ({
      ...s, panel: { ...s.panel }, children: s.children.map(c => ({ ...c })),
    }));
    const snapshotComponents = components.map(c => ({ ...c }));

    // Find which section contains this child
    const parentSection = snapshotSections.find(s =>
      s.children.some(c => c.id === childId)
    );

    // Pre-compute child rows from snapshot (stable during drag)
    const childRows = parentSection ? detectChildRows(parentSection.children) : [];

    const startX = e.clientX;
    const startY = e.clientY;
    const sb = { x: comp.x, y: comp.y, width: comp.width, height: comp.height };
    const MIN_SIZE = 10;

    const onMove = (me) => {
      const dx = me.clientX - startX;
      const dy = me.clientY - startY;
      let nx = sb.x, ny = sb.y, nw = sb.width, nh = sb.height;

      if (handle.includes('e')) nw = Math.max(MIN_SIZE, sb.width + dx);
      if (handle.includes('w')) {
        nw = Math.max(MIN_SIZE, sb.width - dx);
        nx = sb.x + (sb.width - nw);
      }
      if (handle.includes('s')) nh = Math.max(MIN_SIZE, sb.height + dy);
      if (handle.includes('n')) {
        nh = Math.max(MIN_SIZE, sb.height - dy);
        ny = sb.y + (sb.height - nh);
      }

      // Clamp resized child to panel bounds
      const proposed = { x: nx, y: ny, width: nw, height: nh };
      const clamped = parentSection
        ? clampChildToPanel(proposed, parentSection.panel, MIN_SIZE)
        : proposed;

      // Compute neighbor adjustments
      const neighborUpdates = parentSection
        ? adjustAdjacentChildren(comp, clamped, parentSection, childRows)
        : [];

      // Build update map: resized child + all affected neighbors
      const updateMap = new Map();
      updateMap.set(childId, clamped);
      for (const u of neighborUpdates) {
        updateMap.set(u.id, { x: u.x, y: u.y, width: u.width, height: u.height });
      }

      // Apply updates to snapshot (not live state — prevents drift)
      const updated = snapshotComponents.map(c => {
        const u = updateMap.get(c.id);
        if (u) return { ...c, ...u };
        return c;
      });

      onUpdateComponents(updated);
    };

    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      if (onCommitResize) onCommitResize();
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [components, sections, onUpdateComponents, onCommitResize]);

  // ── Panel drag-to-swap ─────────────────────────────────────────────────

  const handlePanelMouseDown = useCallback((panelId, e) => {
    e.preventDefault();

    const startX = e.clientX;
    const startY = e.clientY;
    let isDragging = false;

    const snapshotSections = sections.map(s => ({
      ...s, panel: { ...s.panel }, children: s.children.map(c => ({ ...c })),
    }));
    const snapshotComponents = components.map(c => ({ ...c }));
    const draggedSection = snapshotSections.find(s => s.panel.id === panelId);

    if (!draggedSection) return;

    const onMove = (me) => {
      const dx = me.clientX - startX;
      const dy = me.clientY - startY;

      if (!isDragging) {
        if (Math.sqrt(dx * dx + dy * dy) < DRAG_DEADZONE) return;
        isDragging = true;
      }

      setDragGhost({
        label: draggedSection.label || 'Section',
        icon: 'fa-solid fa-square',
        x: me.clientX,
        y: me.clientY,
      });

      const pos = getCanvasPos(me);
      let target = null;
      for (const sec of snapshotSections) {
        if (sec.panel.id === panelId) continue;
        const p = sec.panel;
        if (pos.x >= p.x && pos.x <= p.x + p.width && pos.y >= p.y && pos.y <= p.y + p.height) {
          target = { type: 'panel', id: sec.panel.id };
          break;
        }
      }

      dropTargetRef.current = target;
      setDropTarget(target);
    };

    const onUp = (me) => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);

      if (!isDragging) {
        onSelect(panelId, me.shiftKey);
        return;
      }

      const target = dropTargetRef.current;
      dropTargetRef.current = null;
      setDropTarget(null);
      setDragGhost(null);

      if (target?.type === 'panel') {
        const targetSection = snapshotSections.find(s => s.panel.id === target.id);
        if (targetSection) {
          const updated = swapSectionPositions(draggedSection, targetSection, snapshotComponents);
          onUpdateComponents(updated);
          if (onCommitResize) onCommitResize();
        }
      }
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [sections, components, onSelect, onUpdateComponents, onCommitResize, getCanvasPos]);

  // ── Child drag-to-swap ─────────────────────────────────────────────────

  const handleChildMouseDown = useCallback((childId, sectionPanelId, e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startY = e.clientY;
    let isDragging = false;

    const snapshotSections = sections.map(s => ({
      ...s, panel: { ...s.panel }, children: s.children.map(c => ({ ...c })),
    }));
    const snapshotComponents = components.map(c => ({ ...c }));
    const parentSection = snapshotSections.find(s => s.panel.id === sectionPanelId);
    const draggedChild = parentSection?.children.find(c => c.id === childId);

    if (!draggedChild || !parentSection) return;

    const rows = detectChildRows(parentSection.children);
    const draggedRow = rows.find(r => r.some(c => c.id === childId));

    const onMove = (me) => {
      const dx = me.clientX - startX;
      const dy = me.clientY - startY;

      if (!isDragging) {
        if (Math.sqrt(dx * dx + dy * dy) < DRAG_DEADZONE) return;
        isDragging = true;
      }

      setDragGhost({
        label: draggedChild.label || draggedChild.type,
        icon: TYPE_ICONS[draggedChild.type] || 'fa-solid fa-cube',
        x: me.clientX,
        y: me.clientY,
      });

      const pos = getCanvasPos(me);
      let target = null;

      // Check if cursor is directly over a child in the same section
      for (const child of parentSection.children) {
        if (child.id === childId) continue;
        if (pos.x >= child.x && pos.x <= child.x + child.width &&
            pos.y >= child.y && pos.y <= child.y + child.height) {
          const targetRow = rows.find(r => r.some(c => c.id === child.id));
          if (targetRow && draggedRow && targetRow !== draggedRow) {
            target = {
              type: 'childRow',
              id: child.id,
              sectionPanelId,
              rowChildIds: targetRow.map(c => c.id),
              targetRow,
              draggedRow,
            };
          } else {
            target = { type: 'child', id: child.id, sectionPanelId };
          }
          break;
        }
      }

      // If not directly over a child, check if in a different row's Y range
      if (!target && draggedRow) {
        for (const row of rows) {
          if (row === draggedRow) continue;
          const rowMinY = Math.min(...row.map(c => c.y));
          const rowMaxY = Math.max(...row.map(c => c.y + c.height));
          if (pos.y >= rowMinY && pos.y <= rowMaxY &&
              pos.x >= parentSection.panel.x && pos.x <= parentSection.panel.x + parentSection.panel.width) {
            target = {
              type: 'childRow',
              id: row[0].id,
              sectionPanelId,
              rowChildIds: row.map(c => c.id),
              targetRow: row,
              draggedRow,
            };
            break;
          }
        }
      }

      dropTargetRef.current = target;
      setDropTarget(target);
    };

    const onUp = (me) => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);

      if (!isDragging) {
        onSelect(childId, me.shiftKey);
        return;
      }

      const target = dropTargetRef.current;
      dropTargetRef.current = null;
      setDropTarget(null);
      setDragGhost(null);

      if (!target) return;

      if (target.type === 'child') {
        const targetChild = parentSection.children.find(c => c.id === target.id);
        if (targetChild) {
          const updated = swapChildPositions(draggedChild, targetChild, snapshotComponents);
          onUpdateComponents(updated);
          if (onCommitResize) onCommitResize();
        }
      } else if (target.type === 'childRow') {
        const updated = swapChildRowPositions(target.draggedRow, target.targetRow, snapshotComponents);
        onUpdateComponents(updated);
        if (onCommitResize) onCommitResize();
      }
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [sections, components, onSelect, onUpdateComponents, onCommitResize, getCanvasPos]);

  // ── Canvas click to deselect ────────────────────────────────────────────

  const handleCanvasClick = useCallback((e) => {
    if (e.target === canvasRef.current) {
      onDeselect();
    }
  }, [onDeselect]);

  // ── Drop target highlight helpers ───────────────────────────────────────

  const dropTargetPanelId = dropTarget?.type === 'panel' ? dropTarget.id : null;
  const dropTargetChildIds = useMemo(() => {
    if (!dropTarget) return new Set();
    if (dropTarget.type === 'child') return new Set([dropTarget.id]);
    if (dropTarget.type === 'childRow') return new Set(dropTarget.rowChildIds);
    return new Set();
  }, [dropTarget]);

  return (
    <div
      ref={canvasRef}
      onClick={handleCanvasClick}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: config.width,
        height: config.height,
        background: 'rgba(0, 0, 0, 0.15)',
        zIndex: 50,
        ...(dragGhost && { cursor: 'grabbing' }),
      }}
    >
      {/* Structure mode badge */}
      <div style={{
        position: 'absolute', top: 4, right: 8,
        padding: '2px 8px', borderRadius: 4,
        background: 'rgba(186, 156, 255, 0.2)',
        border: '1px solid rgba(186, 156, 255, 0.25)',
        fontSize: 9, fontWeight: 600, letterSpacing: '0.08em',
        color: 'rgba(186, 156, 255, 0.8)',
        zIndex: 20, pointerEvents: 'none',
      }}>
        STRUCTURE
      </div>

      {/* Alignment snap guides */}
      {snapGuides.map((guide, i) => (
        guide.type === 'vertical' ? (
          <div key={`sg-${i}`} style={{
            position: 'absolute', left: guide.position, top: 0, width: 0,
            height: '100%', borderLeft: '1px dashed rgba(0, 200, 255, 0.7)',
            pointerEvents: 'none', zIndex: 100,
          }} />
        ) : (
          <div key={`sg-${i}`} style={{
            position: 'absolute', top: guide.position, left: 0, height: 0,
            width: '100%', borderTop: '1px dashed rgba(0, 200, 255, 0.7)',
            pointerEvents: 'none', zIndex: 100,
          }} />
        )
      ))}

      {/* Render section outlines */}
      {sections.map(section => (
        <SectionBox
          key={section.panel.id}
          section={section}
          isSelected={selectedIds.includes(section.panel.id)}
          isDropTarget={dropTargetPanelId === section.panel.id}
          onMouseDown={(panelId, e) => handlePanelMouseDown(panelId, e)}
          onResizeStart={handleResizeStart}
        >
          {section.children.map(child => (
            <ChildBox
              key={child.id}
              comp={child}
              isSelected={selectedIds.includes(child.id)}
              isDropTarget={dropTargetChildIds.has(child.id)}
              onMouseDown={(childId, e) => handleChildMouseDown(childId, section.panel.id, e)}
              onResizeStart={handleChildResizeStart}
              panelBounds={section.panel}
            />
          ))}
        </SectionBox>
      ))}

      {/* Render orphans */}
      {orphans.map(comp => (
        <div
          key={comp.id}
          onClick={(e) => { e.stopPropagation(); onSelect(comp.id, e.shiftKey); }}
          style={{
            position: 'absolute',
            left: comp.x, top: comp.y,
            width: comp.width, height: comp.height,
            border: selectedIds.includes(comp.id)
              ? '2px solid rgba(255, 170, 0, 0.8)'
              : '1px dashed rgba(255, 170, 0, 0.4)',
            background: 'rgba(255, 170, 0, 0.05)',
            borderRadius: 2,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            gap: 3, fontSize: 9,
            color: 'rgba(255, 170, 0, 0.6)',
            cursor: 'pointer', zIndex: 2,
          }}
        >
          <i className={TYPE_ICONS[comp.type] || 'fa-solid fa-cube'} style={{ fontSize: 8 }} />
          {comp.label && <span>{comp.label}</span>}
        </div>
      ))}

      {/* Drag ghost */}
      {dragGhost && (
        <div style={{
          position: 'fixed',
          left: dragGhost.x + 14,
          top: dragGhost.y - 10,
          padding: '4px 10px',
          background: 'rgba(186, 156, 255, 0.35)',
          border: '1px solid rgba(186, 156, 255, 0.7)',
          borderRadius: 6,
          fontSize: 10,
          fontWeight: 600,
          color: '#fff',
          pointerEvents: 'none',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          backdropFilter: 'blur(4px)',
          whiteSpace: 'nowrap',
          boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
        }}>
          <i className={dragGhost.icon} style={{ fontSize: 9 }} />
          {dragGhost.label}
          {dropTarget && (
            <span style={{ fontSize: 8, opacity: 0.7, marginLeft: 2 }}>
              → swap
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default StructureView;
