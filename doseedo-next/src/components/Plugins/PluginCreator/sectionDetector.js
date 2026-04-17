/**
 * Section Detector — Shared utility for detecting panel-child relationships
 * and performing section-level operations (reflow, adjacent adjustment).
 *
 * Components are stored as a flat array with absolute positions.
 * Panels "contain" children when the child's center is inside the panel bounds.
 */

// No layout-engine imports needed — we use proportional scaling for reflow

// ── Constants ────────────────────────────────────────────────────────────────

export const MIN_PANEL_WIDTH = 80;
export const MIN_PANEL_HEIGHT = 60;
const ROW_Y_TOLERANCE = 20; // px — panels within this Y distance are "same row"
const COL_X_TOLERANCE = 20; // px — panels within this X distance are "same column"

// ── Build section map from flat component array ─────────────────────────────

/**
 * Build a section map from the flat component array.
 * A component is "inside" a panel if its center falls within the panel bounds.
 * When panels overlap, child is assigned to the smallest containing panel.
 *
 * @param {Array} components — flat component array
 * @returns {{ sections: Array<{panel, children, label}>, orphans: Array }}
 */
export function buildSectionMap(components) {
  // Section panels are at zIndex 0 (created by layout engine).
  // Sub-panels (zIndex >= 1) are children.
  const panels = components.filter(c => c.type === 'panel' && (c.zIndex === 0 || c.zIndex === undefined));
  // If no zIndex-0 panels, fall back to all panels
  const sectionPanels = panels.length > 0
    ? panels
    : components.filter(c => c.type === 'panel');

  // Sort panels by area ascending so we assign to smallest containing panel first
  const sortedPanels = [...sectionPanels].sort((a, b) =>
    (a.width * a.height) - (b.width * b.height)
  );

  const nonPanelComps = components.filter(c =>
    !sectionPanels.some(p => p.id === c.id)
  );

  const assigned = new Set();
  const childMap = new Map(); // panel.id -> children[]

  for (const comp of nonPanelComps) {
    const cx = comp.x + (comp.width || 0) / 2;
    const cy = comp.y + (comp.height || 0) / 2;

    for (const panel of sortedPanels) {
      if (cx >= panel.x && cx <= panel.x + panel.width &&
          cy >= panel.y && cy <= panel.y + panel.height) {
        if (!childMap.has(panel.id)) childMap.set(panel.id, []);
        childMap.get(panel.id).push(comp);
        assigned.add(comp.id);
        break; // Assigned to smallest containing panel
      }
    }
  }

  const sections = sectionPanels.map(panel => ({
    panel,
    children: childMap.get(panel.id) || [],
    label: findSectionLabel(childMap.get(panel.id) || [], panel),
  }));

  const orphans = components.filter(c =>
    !assigned.has(c.id) && !sectionPanels.some(p => p.id === c.id)
  );

  return { sections, orphans };
}

/**
 * Find the section label — typically the first label near the panel's top edge.
 */
function findSectionLabel(children, panel) {
  const labels = children.filter(c => c.type === 'label');
  if (labels.length === 0) return panel.label || 'Section';
  const sorted = [...labels].sort((a, b) => a.y - b.y);
  const topLabel = sorted[0];
  if (topLabel.y < panel.y + 30) return topLabel.label;
  return panel.label || topLabel.label || 'Section';
}

// ── Grid structure detection ────────────────────────────────────────────────

/**
 * Detect the row/column structure from sections.
 * Sections on the same row share approximately the same Y position.
 *
 * @param {Array} sections — from buildSectionMap
 * @returns {{ rows: Array<{ sections: Array, y: number, height: number }> }}
 */
export function detectGridStructure(sections) {
  if (sections.length === 0) return { rows: [] };

  // Sort sections by Y then X
  const sorted = [...sections].sort((a, b) => {
    const dy = a.panel.y - b.panel.y;
    return dy !== 0 ? dy : a.panel.x - b.panel.x;
  });

  const rows = [];
  let currentRow = [sorted[0]];
  let currentRowY = sorted[0].panel.y;

  for (let i = 1; i < sorted.length; i++) {
    const sec = sorted[i];
    if (Math.abs(sec.panel.y - currentRowY) <= ROW_Y_TOLERANCE) {
      currentRow.push(sec);
    } else {
      const rowHeight = Math.max(...currentRow.map(s => s.panel.height));
      rows.push({ sections: currentRow, y: currentRowY, height: rowHeight });
      currentRow = [sec];
      currentRowY = sec.panel.y;
    }
  }
  // Last row
  if (currentRow.length > 0) {
    const rowHeight = Math.max(...currentRow.map(s => s.panel.height));
    rows.push({ sections: currentRow, y: currentRowY, height: rowHeight });
  }

  return { rows };
}

// ── Find adjacent sections ──────────────────────────────────────────────────

/**
 * Find sections adjacent to the target section.
 *
 * @param {Object} target — section object with .panel
 * @param {Array} allSections — all sections
 * @param {Object} grid — from detectGridStructure
 * @returns {{ sameRow: Array, sameColumn: Array, rowIndex: number }}
 */
export function findAdjacentSections(target, allSections, grid) {
  const sameRow = [];
  const sameColumn = [];
  let rowIndex = -1;

  for (let ri = 0; ri < grid.rows.length; ri++) {
    const row = grid.rows[ri];
    const inRow = row.sections.find(s => s.panel.id === target.panel.id);
    if (inRow) {
      rowIndex = ri;
      for (const s of row.sections) {
        if (s.panel.id !== target.panel.id) sameRow.push(s);
      }
    }
  }

  // Same column: sections in other rows at approximately the same X
  const targetCenterX = target.panel.x + target.panel.width / 2;
  for (let ri = 0; ri < grid.rows.length; ri++) {
    if (ri === rowIndex) continue;
    for (const s of grid.rows[ri].sections) {
      const sCenterX = s.panel.x + s.panel.width / 2;
      if (Math.abs(sCenterX - targetCenterX) <= COL_X_TOLERANCE + target.panel.width / 2) {
        sameColumn.push(s);
      }
    }
  }

  return { sameRow, sameColumn, rowIndex };
}

// ── Reflow section children ─────────────────────────────────────────────────

/**
 * Proportionally re-position children within new panel bounds.
 * Preserves component IDs, svgStyle, color, etc — only updates x/y/width/height.
 * Uses simple proportional scaling: children maintain their relative position
 * within the panel as it resizes. This is bulletproof — no layout engine re-run.
 *
 * @param {Object} section — { panel, children, label }
 * @param {{ x, y, width, height }} newBounds — new panel bounds
 * @returns {Array} — updated children with new x/y/width/height
 */
export function reflowSectionChildren(section, newBounds) {
  const { children, panel: oldBounds } = section;
  if (!children || children.length === 0) return [];
  if (!oldBounds || oldBounds.width === 0 || oldBounds.height === 0) return children;

  const scaleX = newBounds.width / oldBounds.width;
  const scaleY = newBounds.height / oldBounds.height;

  return children.map(child => ({
    ...child,
    x: newBounds.x + (child.x - oldBounds.x) * scaleX,
    y: newBounds.y + (child.y - oldBounds.y) * scaleY,
    width: child.width * scaleX,
    height: child.height * scaleY,
  }));
}

// ── Adjust adjacent panels ──────────────────────────────────────────────────

/**
 * Compute updates for adjacent panels when a panel is resized.
 * Same-row neighbors absorb horizontal changes. Canvas width is preserved.
 *
 * @param {Object} resizedSection — the section being resized
 * @param {{ width: number, height: number, x: number, y: number }} newBounds — new panel bounds
 * @param {Array} allSections — all sections
 * @param {Object} grid — from detectGridStructure
 * @returns {Array<{ section, newBounds }>} — updates for adjacent sections
 */
export function adjustAdjacentPanels(resizedSection, newBounds, allSections, grid) {
  const { sameRow, rowIndex } = findAdjacentSections(resizedSection, allSections, grid);
  const updates = [];
  const oldBounds = resizedSection.panel;
  const deltaW = newBounds.width - oldBounds.width;
  const deltaX = newBounds.x - oldBounds.x;
  const deltaH = newBounds.height - oldBounds.height;
  const deltaY = newBounds.y - oldBounds.y;

  // Nothing changed
  if (deltaW === 0 && deltaX === 0 && deltaH === 0 && deltaY === 0) return updates;

  // Sort same-row neighbors by x position
  const sorted = [...sameRow].sort((a, b) => a.panel.x - b.panel.x);
  const rightNeighbors = sorted.filter(s => s.panel.x > oldBounds.x);
  const leftNeighbors = sorted.filter(s => s.panel.x < oldBounds.x);

  // Helper to add or merge an update for a section
  const mergeUpdate = (section, boundsUpdate) => {
    const existing = updates.find(u => u.section.panel.id === section.panel.id);
    if (existing) {
      Object.assign(existing.newBounds, boundsUpdate);
    } else {
      updates.push({ section, newBounds: { ...section.panel, ...boundsUpdate } });
    }
  };

  // ── Horizontal adjustments ──────────────────────────────────────────────
  if (deltaW !== 0) {
    // Right edge moved → right neighbors shift and absorb
    if (deltaX === 0 || Math.abs(deltaX) < Math.abs(deltaW)) {
      const rightShare = rightNeighbors.length > 0 ? -deltaW / rightNeighbors.length : 0;
      for (const neighbor of rightNeighbors) {
        mergeUpdate(neighbor, {
          width: Math.max(MIN_PANEL_WIDTH, neighbor.panel.width + rightShare),
          x: neighbor.panel.x + deltaW,
        });
      }
    }
    // Left edge moved → left neighbors absorb
    if (deltaX !== 0 && leftNeighbors.length > 0) {
      const leftShare = deltaX / leftNeighbors.length;
      for (const neighbor of leftNeighbors) {
        mergeUpdate(neighbor, {
          width: Math.max(MIN_PANEL_WIDTH, neighbor.panel.width + leftShare),
        });
      }
    }
  }

  // ── Vertical adjustments ────────────────────────────────────────────────
  if (deltaH !== 0 || deltaY !== 0) {
    // Same-row panels: match Y and height — but ONLY for truly side-by-side
    // panels (those that vertically overlap with the resized panel).
    // Panels whose bottom is above the resized top, or whose top is below
    // the resized bottom, are stacked — not same-row — even if the grid
    // grouped them together.
    if (rowIndex >= 0) {
      for (const neighbor of sameRow) {
        const nBot = neighbor.panel.y + neighbor.panel.height;
        const isAbove = nBot <= oldBounds.y;
        const isBelow = neighbor.panel.y >= oldBounds.y + oldBounds.height;
        if (!isAbove && !isBelow) {
          mergeUpdate(neighbor, { y: newBounds.y, height: newBounds.height });
        }
      }
    }

    // Panels above: shift by deltaY to maintain gap (mirrors below-panel shift)
    if (deltaY !== 0) {
      for (const sec of allSections) {
        if (sec.panel.id === resizedSection.panel.id) continue;
        if (sec.panel.y + sec.panel.height <= oldBounds.y) {
          mergeUpdate(sec, { y: sec.panel.y + deltaY });
        }
      }
    }

    // Panels below: shift by the change in the row's bottom edge.
    const oldRowBottom = oldBounds.y + oldBounds.height;
    const newRowBottom = newBounds.y + newBounds.height;
    const verticalShift = newRowBottom - oldRowBottom;

    if (verticalShift !== 0) {
      for (const sec of allSections) {
        if (sec.panel.id === resizedSection.panel.id) continue;
        if (sec.panel.y >= oldRowBottom) {
          mergeUpdate(sec, { y: sec.panel.y + verticalShift });
        }
      }
    }
  }

  return updates;
}

// ── Clamp child bounds to parent panel ────────────────────────────────────────

/**
 * Clamp a child's bounds to stay within a panel.
 * @param {{ x, y, width, height }} childBounds
 * @param {{ x, y, width, height }} panelBounds
 * @param {number} minSize — minimum allowed width/height
 * @returns {{ x, y, width, height }}
 */
export function clampChildToPanel(childBounds, panelBounds, minSize = 10) {
  let { x, y, width, height } = childBounds;
  x = Math.max(panelBounds.x, x);
  y = Math.max(panelBounds.y, y);
  width = Math.min(width, panelBounds.x + panelBounds.width - x);
  height = Math.min(height, panelBounds.y + panelBounds.height - y);
  width = Math.max(minSize, width);
  height = Math.max(minSize, height);
  return { x, y, width, height };
}

// ── Adjust adjacent children within a section ─────────────────────────────────

/**
 * Find siblings in the same horizontal band (their vertical extents overlap
 * with the resized child). This catches dropdowns, labels, knobs at different
 * heights — anything that shares vertical space.
 */
function findHorizontalBandSiblings(resizedChild, allChildren) {
  const top = resizedChild.y;
  const bottom = resizedChild.y + resizedChild.height;
  return allChildren.filter(c => {
    if (c.id === resizedChild.id) return false;
    const cTop = c.y;
    const cBot = c.y + c.height;
    return cTop < bottom && cBot > top; // vertical overlap
  });
}

/**
 * Compute updates for sibling children when a child is resized within a section.
 *
 * Horizontal: uses vertical-overlap band detection (not row-based) so dropdowns,
 * labels, and mixed-height components are all treated as neighbors.
 * Push-first, shrink-only-if-overflow: neighbors shift by the edge delta,
 * then only shrink if they'd overflow the panel boundary.
 *
 * Vertical: rows below/above (from detectChildRows) shift by the height delta.
 *
 * @param {Object} resizedChild — original child (from snapshot)
 * @param {{ x, y, width, height }} newChildBounds — proposed new bounds
 * @param {Object} section — { panel, children } from snapshot
 * @param {Array<Array>} childRows — pre-computed from detectChildRows
 * @returns {Array<{ id, x, y, width, height }>} — updates for affected siblings
 */
export function adjustAdjacentChildren(resizedChild, newChildBounds, section, childRows) {
  const MIN_CHILD_SIZE = 10;
  const panelBounds = section.panel;
  const panelRight = panelBounds.x + panelBounds.width;
  const panelLeft = panelBounds.x;
  const oldBounds = { x: resizedChild.x, y: resizedChild.y,
                      width: resizedChild.width, height: resizedChild.height };

  const deltaW = newChildBounds.width - oldBounds.width;
  const deltaX = newChildBounds.x - oldBounds.x;
  const deltaH = newChildBounds.height - oldBounds.height;
  const deltaY = newChildBounds.y - oldBounds.y;

  if (deltaW === 0 && deltaX === 0 && deltaH === 0 && deltaY === 0) return [];

  const updateMap = new Map(); // id -> { x, y, width, height }

  const mergeUpdate = (child, partial) => {
    const existing = updateMap.get(child.id) || {
      x: child.x, y: child.y, width: child.width, height: child.height,
    };
    Object.assign(existing, partial);
    updateMap.set(child.id, existing);
  };

  // ── Horizontal adjustments (band-based, push-first) ──
  if (deltaW !== 0 || deltaX !== 0) {
    // Use vertical-overlap detection instead of row-based
    const bandSiblings = findHorizontalBandSiblings(resizedChild, section.children);
    const rightNeighbors = bandSiblings
      .filter(c => (c.x + c.width / 2) > (oldBounds.x + oldBounds.width / 2))
      .sort((a, b) => a.x - b.x);
    const leftNeighbors = bandSiblings
      .filter(c => (c.x + c.width / 2) < (oldBounds.x + oldBounds.width / 2))
      .sort((a, b) => b.x - a.x); // sort right-to-left

    // Right edge grew → push right neighbors, shrink only if overflow
    const rightEdgeDelta = (newChildBounds.x + newChildBounds.width) - (oldBounds.x + oldBounds.width);
    if (rightEdgeDelta > 0 && rightNeighbors.length > 0) {
      // Step 1: push all right neighbors by the delta (preserve their widths)
      for (const neighbor of rightNeighbors) {
        mergeUpdate(neighbor, { x: neighbor.x + rightEdgeDelta });
      }

      // Step 2: check if any neighbor overflows the panel right edge
      let totalOverflow = 0;
      for (const neighbor of rightNeighbors) {
        const pushed = updateMap.get(neighbor.id);
        const rightEdge = pushed.x + pushed.width;
        if (rightEdge > panelRight) {
          totalOverflow = Math.max(totalOverflow, rightEdge - panelRight);
        }
      }

      // Step 3: if overflow, shrink neighbors (distribute evenly)
      if (totalOverflow > 0) {
        const shrinkPer = totalOverflow / rightNeighbors.length;
        for (const neighbor of rightNeighbors) {
          const existing = updateMap.get(neighbor.id);
          existing.width = Math.max(MIN_CHILD_SIZE, existing.width - shrinkPer);
        }
      }
    } else if (rightEdgeDelta < 0 && rightNeighbors.length > 0) {
      // Right edge shrank → pull right neighbors left (give them back space)
      for (const neighbor of rightNeighbors) {
        mergeUpdate(neighbor, {
          x: neighbor.x + rightEdgeDelta,
          width: neighbor.width - rightEdgeDelta, // grow by the recovered space
        });
      }
      // Only the nearest neighbor grows; the rest just shift
      if (rightNeighbors.length > 1) {
        for (let i = 1; i < rightNeighbors.length; i++) {
          const existing = updateMap.get(rightNeighbors[i].id);
          existing.width = rightNeighbors[i].width; // restore original width, just shift
        }
      }
    }

    // Left edge moved → push/adjust left neighbors
    const leftEdgeDelta = newChildBounds.x - oldBounds.x;
    if (leftEdgeDelta < 0 && leftNeighbors.length > 0) {
      // Left edge grew (moved left) → push left neighbors left, shrink if overflow
      for (const neighbor of leftNeighbors) {
        const pushedRight = neighbor.x + neighbor.width + leftEdgeDelta;
        if (pushedRight > newChildBounds.x) {
          // Neighbor's right edge would overlap — shrink it
          mergeUpdate(neighbor, {
            width: Math.max(MIN_CHILD_SIZE, newChildBounds.x - neighbor.x),
          });
        }
      }
      // Check left overflow
      for (const neighbor of leftNeighbors) {
        const existing = updateMap.get(neighbor.id);
        if (existing && existing.x < panelLeft) {
          existing.width = Math.max(MIN_CHILD_SIZE, existing.width - (panelLeft - existing.x));
          existing.x = panelLeft;
        }
      }
    } else if (leftEdgeDelta > 0 && leftNeighbors.length > 0) {
      // Left edge shrank (moved right) → nearest left neighbor can grow
      const nearest = leftNeighbors[0]; // rightmost of the left neighbors
      mergeUpdate(nearest, {
        width: nearest.width + leftEdgeDelta,
      });
    }
  }

  // ── Vertical adjustments (row-based) ──
  if (deltaH !== 0 || deltaY !== 0) {
    const resizedRowIndex = childRows.findIndex(row => row.some(c => c.id === resizedChild.id));

    if (resizedRowIndex >= 0) {
      const oldBottom = oldBounds.y + oldBounds.height;
      const newBottom = newChildBounds.y + newChildBounds.height;
      const verticalShift = newBottom - oldBottom;

      // Rows below: shift down
      if (verticalShift !== 0) {
        for (let ri = resizedRowIndex + 1; ri < childRows.length; ri++) {
          for (const child of childRows[ri]) {
            mergeUpdate(child, { y: child.y + verticalShift });
          }
        }
      }

      // Rows above: shift by deltaY (if top edge moved)
      if (deltaY !== 0) {
        for (let ri = 0; ri < resizedRowIndex; ri++) {
          for (const child of childRows[ri]) {
            mergeUpdate(child, { y: child.y + deltaY });
          }
        }
      }
    }
  }

  // ── Clamp all updates to panel bounds ──
  const results = [];
  for (const [id, bounds] of updateMap) {
    bounds.x = Math.max(panelBounds.x, bounds.x);
    bounds.y = Math.max(panelBounds.y, bounds.y);
    if (bounds.x + bounds.width > panelBounds.x + panelBounds.width) {
      bounds.width = panelBounds.x + panelBounds.width - bounds.x;
    }
    if (bounds.y + bounds.height > panelBounds.y + panelBounds.height) {
      bounds.height = panelBounds.y + panelBounds.height - bounds.y;
    }
    bounds.width = Math.max(MIN_CHILD_SIZE, bounds.width);
    bounds.height = Math.max(MIN_CHILD_SIZE, bounds.height);
    results.push({ id, ...bounds });
  }

  return results;
}

// ── Apply section updates to component array ────────────────────────────────

/**
 * Apply a set of section updates (panel + reflowed children) to the component array.
 * Returns a new array with all updates applied.
 *
 * @param {Array} components — current flat component array
 * @param {Array} updates — array of { panelId, newPanelBounds, updatedChildren }
 * @returns {Array} — new component array
 */
export function applySectionUpdates(components, updates) {
  const updateMap = new Map(); // componentId -> updates

  for (const { panelId, newPanelBounds, updatedChildren } of updates) {
    // Panel update
    updateMap.set(panelId, {
      x: newPanelBounds.x,
      y: newPanelBounds.y,
      width: newPanelBounds.width,
      height: newPanelBounds.height,
    });

    // Children updates
    for (const child of updatedChildren) {
      updateMap.set(child.id, {
        x: child.x,
        y: child.y,
        width: child.width,
        height: child.height,
      });
    }
  }

  return components.map(c => {
    const update = updateMap.get(c.id);
    if (update) return { ...c, ...update };
    return c;
  });
}

// ── Visual row detection within a section ────────────────────────────────

/**
 * Group a section's children into visual rows based on center-Y proximity.
 */
export function detectChildRows(children, tolerance = 20) {
  if (!children || children.length === 0) return [];
  const sorted = [...children].sort((a, b) => (a.y + a.height / 2) - (b.y + b.height / 2));

  const rows = [];
  let currentRow = [sorted[0]];
  let rowCenterY = sorted[0].y + sorted[0].height / 2;

  for (let i = 1; i < sorted.length; i++) {
    const cy = sorted[i].y + sorted[i].height / 2;
    if (Math.abs(cy - rowCenterY) <= tolerance) {
      currentRow.push(sorted[i]);
    } else {
      rows.push(currentRow);
      currentRow = [sorted[i]];
      rowCenterY = cy;
    }
  }
  if (currentRow.length > 0) rows.push(currentRow);

  return rows;
}

// ── Swap operations ──────────────────────────────────────────────────────

/**
 * Swap two sections' positions. Each panel takes the other's exact bounds.
 * Children are proportionally reflowed into the new bounds.
 */
export function swapSectionPositions(sectionA, sectionB, components) {
  const boundsA = { x: sectionA.panel.x, y: sectionA.panel.y, width: sectionA.panel.width, height: sectionA.panel.height };
  const boundsB = { x: sectionB.panel.x, y: sectionB.panel.y, width: sectionB.panel.width, height: sectionB.panel.height };

  const aChildrenInB = reflowSectionChildren(sectionA, boundsB);
  const bChildrenInA = reflowSectionChildren(sectionB, boundsA);

  return applySectionUpdates(components, [
    { panelId: sectionA.panel.id, newPanelBounds: boundsB, updatedChildren: aChildrenInB },
    { panelId: sectionB.panel.id, newPanelBounds: boundsA, updatedChildren: bChildrenInA },
  ]);
}

/**
 * Swap two child components' center-aligned positions.
 */
export function swapChildPositions(compA, compB, components) {
  const aCx = compA.x + compA.width / 2;
  const aCy = compA.y + compA.height / 2;
  const bCx = compB.x + compB.width / 2;
  const bCy = compB.y + compB.height / 2;

  return components.map(c => {
    if (c.id === compA.id) return { ...c, x: bCx - c.width / 2, y: bCy - c.height / 2 };
    if (c.id === compB.id) return { ...c, x: aCx - c.width / 2, y: aCy - c.height / 2 };
    return c;
  });
}

/**
 * Swap two visual rows of children vertically, preserving the gap between them.
 */
export function swapChildRowPositions(rowA, rowB, components) {
  const rowAMinY = Math.min(...rowA.map(c => c.y));
  const rowBMinY = Math.min(...rowB.map(c => c.y));
  const rowAMaxBot = Math.max(...rowA.map(c => c.y + c.height));
  const rowBMaxBot = Math.max(...rowB.map(c => c.y + c.height));

  let topRow, bottomRow, topMinY, topMaxBot, botMinY, botMaxBot;
  if (rowAMinY <= rowBMinY) {
    topRow = rowA; bottomRow = rowB;
    topMinY = rowAMinY; topMaxBot = rowAMaxBot;
    botMinY = rowBMinY; botMaxBot = rowBMaxBot;
  } else {
    topRow = rowB; bottomRow = rowA;
    topMinY = rowBMinY; topMaxBot = rowBMaxBot;
    botMinY = rowAMinY; botMaxBot = rowAMaxBot;
  }

  const botHeight = botMaxBot - botMinY;
  const gap = botMinY - topMaxBot;

  // After swap: bottom row goes to top position, top row goes below with same gap
  const newBotTop = topMinY;
  const newTopTop = topMinY + botHeight + gap;

  const shiftBot = newBotTop - botMinY;
  const shiftTop = newTopTop - topMinY;

  const topIds = new Set(topRow.map(c => c.id));
  const botIds = new Set(bottomRow.map(c => c.id));

  return components.map(c => {
    if (topIds.has(c.id)) return { ...c, y: c.y + shiftTop };
    if (botIds.has(c.id)) return { ...c, y: c.y + shiftBot };
    return c;
  });
}
