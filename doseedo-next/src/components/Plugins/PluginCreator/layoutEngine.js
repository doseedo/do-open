/**
 * Layout Engine — Deterministic component positioning
 * Takes a Design Brief (sections + components) and computes x/y/width/height
 * for every component. No AI involved.
 */

import { generateKnobSVG, generateSliderSVG, generateButtonSVG } from './svgComponentLibrary';
import { generateTheme } from './themeGenerator';

// ── Default component sizes ──────────────────────────────────────────────────

const TYPE_SIZES = {
  knob:     { w: 60,  h: 60 },
  slider:   { w: 30,  h: 120 },
  button:   { w: 70,  h: 28 },
  label:    { w: 100, h: 20 },
  led:      { w: 12,  h: 12 },
  dropdown: { w: 110, h: 26 },
  image:    { w: 80,  h: 80 },
  panel:    { w: 200, h: 150 },
  meter:    { w: 24,  h: 100 },
  waveform: { w: 160, h: 50 },
  'wavetable_3d': { w: 220, h: 140 },
  'xy-pad': { w: 120, h: 120 },
};

// Medium density — for large canvases with many components (matches pro synth sizing)
const MEDIUM_TYPE_SIZES = {
  knob:     { w: 48,  h: 48 },
  slider:   { w: 24,  h: 80 },
  button:   { w: 56,  h: 22 },
  label:    { w: 70,  h: 14 },
  led:      { w: 10,  h: 10 },
  dropdown: { w: 90,  h: 22 },
  image:    { w: 50,  h: 50 },
  panel:    { w: 180, h: 135 },
  meter:    { w: 20,  h: 80 },
  waveform: { w: 220, h: 100 },
  'wavetable_3d': { w: 240, h: 130 },
  'xy-pad': { w: 90,  h: 90 },
};

const COMPACT_TYPE_SIZES = {
  knob:     { w: 44,  h: 44 },
  slider:   { w: 24,  h: 60 },
  button:   { w: 56,  h: 22 },
  label:    { w: 56,  h: 12 },
  led:      { w: 8,   h: 8 },
  dropdown: { w: 70,  h: 20 },
  image:    { w: 36,  h: 36 },
  panel:    { w: 150, h: 120 },
  meter:    { w: 14,  h: 55 },
  waveform: { w: 180, h: 80 },
  'wavetable_3d': { w: 200, h: 110 },
  'xy-pad': { w: 70,  h: 70 },
};


// ── Post-layout snap pass for consistent alignment ──────────────────────────
function snapLayout(components, snapGrid = 10) {
  if (!components || components.length === 0) return components;
  
  // Find minimum X to detect left edge
  const minX = Math.min(...components.map(c => c.x));
  const quantizedMinX = Math.round(minX / snapGrid) * snapGrid;
  
  return components.map(comp => ({
    ...comp,
    // Snap X relative to left edge
    x: quantizedMinX + Math.round((comp.x - minX) / snapGrid) * snapGrid,
    // Snap Y to grid
    y: Math.round(comp.y / snapGrid) * snapGrid,
    // Snap dimensions for consistency (even numbers)
    width: Math.round(comp.width / 2) * 2,
    height: Math.round(comp.height / 2) * 2,
  }));
}

// ── Post-layout collision resolution ─────────────────────────────────────────
// Detects overlapping interactive components and nudges them apart.
// Runs AFTER snapLayout to fix any overlaps introduced by grid quantization.
function resolveOverlaps(components, maxIterations = 3) {
  if (!components || components.length < 2) return components;

  const skipTypes = new Set(['panel', 'label', 'led']);
  const interactive = components.filter(c => !skipTypes.has(c.type));

  const overlaps = (a, b) => (
    a.x < b.x + b.width &&
    a.x + a.width > b.x &&
    a.y < b.y + b.height &&
    a.y + a.height > b.y
  );

  for (let iter = 0; iter < maxIterations; iter++) {
    let anyFixed = false;

    interactive.sort((a, b) => a.y - b.y || a.x - b.x);

    for (let i = 0; i < interactive.length; i++) {
      for (let j = i + 1; j < interactive.length; j++) {
        const a = interactive[i];
        const b = interactive[j];

        // Skip components in different tabs
        if (a.tabIndex !== undefined && b.tabIndex !== undefined &&
            a.tabIndex !== b.tabIndex) continue;

        if (!overlaps(a, b)) continue;

        const overlapX = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
        const overlapY = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);

        if (overlapX < overlapY && overlapX <= 20) {
          // Small horizontal overlap: nudge b right
          b.x = a.x + a.width + 2;
        } else {
          // Nudge b down below a
          b.y = a.y + a.height + 2;
        }
        anyFixed = true;
      }
    }

    if (!anyFixed) break;
  }

  return components;
}

const PADDING = 16;
const SECTION_GAP = 12;
const TITLE_BAR_HEIGHT = 32;
const SECTION_LABEL_HEIGHT = 22;
const COMPONENT_GAP = 10;

const COMPACT_PADDING = 10;
const COMPACT_SECTION_GAP = 8;
const COMPACT_SECTION_LABEL_HEIGHT = 16;
const COMPACT_COMPONENT_GAP = 8;

const MEDIUM_PADDING = 10;
const MEDIUM_SECTION_GAP = 8;
const MEDIUM_SECTION_LABEL_HEIGHT = 14;
const MEDIUM_COMPONENT_GAP = 8;

// ── Density detection ────────────────────────────────────────────────────────
// Returns 'compact', 'medium', or 'standard'

function assessDensity(sections, canvasW = 600, canvasH = 400) {
  const totalComponents = sections.reduce((sum, s) => sum + (s.components?.length || 0), 0);
  const canvasArea = canvasW * canvasH;
  const isLargeCanvas = canvasArea >= 500000; // ~900x560 or bigger

  if (isLargeCanvas) {
    // Large canvas: use medium for complex layouts, standard for simple
    if (sections.length >= 4 || totalComponents >= 20) {
      return { isDense: false, isMedium: true, totalComponents };
    }
    return { isDense: false, isMedium: false, totalComponents };
  }
  // Small canvas: compact for dense layouts
  const isDense = sections.length >= 4 || totalComponents >= 12;
  return { isDense, isMedium: false, totalComponents };
}

function getSectionSizes(section, isDense, isMedium) {
  if (section.compact === true || isDense) return COMPACT_TYPE_SIZES;
  if (isMedium) return MEDIUM_TYPE_SIZES;
  return TYPE_SIZES;
}

function getSectionGap(isDense, isMedium) {
  if (isDense) return { pad: COMPACT_PADDING, secGap: COMPACT_SECTION_GAP, secLabelH: COMPACT_SECTION_LABEL_HEIGHT, compGap: COMPACT_COMPONENT_GAP };
  if (isMedium) return { pad: MEDIUM_PADDING, secGap: MEDIUM_SECTION_GAP, secLabelH: MEDIUM_SECTION_LABEL_HEIGHT, compGap: MEDIUM_COMPONENT_GAP };
  return { pad: PADDING, secGap: SECTION_GAP, secLabelH: SECTION_LABEL_HEIGHT, compGap: COMPONENT_GAP };
}

// Legacy compat
function isSectionCompact(section, isDense, isMedium = false) {
  if (section.compact === true) return true;
  if (isDense) return true;
  if (isMedium) return 'medium';
  return false;
}

// ── Content-fit section width estimation ──────────────────────────────────────
// Estimates the ideal panel width for a section based on its component types/counts.
// Used by grid and horizontal layouts to size panels to content instead of weight.

function computeIdealSectionWidth(section, sizes, gap, innerPad, sectionLayout) {
  const comps = section.components || [];
  if (comps.length === 0) return 120;

  const displays = comps.filter(c => ['waveform', 'xy-pad', 'image', 'wavetable_3d', 'oscilloscope', 'spectrum-analyzer', 'adsr'].includes(c.type) || (c.type === 'meter' && c.svgStyle === 'vu-needle'));
  const knobs = comps.filter(c => c.type === 'knob');
  const sliders = comps.filter(c => ['slider', 'meter'].includes(c.type) && !(c.type === 'meter' && c.svgStyle === 'vu-needle'));
  const dropdowns = comps.filter(c => c.type === 'dropdown');
  const buttons = comps.filter(c => c.type === 'button');

  const kw = sizes.knob.w;
  const displayMaxW = displays.length > 0
    ? Math.max(...displays.map(d => d.width || (sizes[d.type] || sizes.waveform).w))
    : 0;

  const useDisplayControls = sectionLayout === 'display-controls' ||
    (displays.length === 1 && knobs.length >= 3);

  const nk = knobs.length;

  if (useDisplayControls) {
    // Left: waveform/selectors/verticals.  Right: hero + knob grid.
    // Left column width = max(displayW, 30% of total) — selectors/verticals go below waveform
    const leftW = Math.max(displayMaxW, 80);
    const heroKnobs = knobs.filter(k => k.size === 'large');
    const normalKnobs = knobs.filter(k => k.size !== 'large');
    const nn = normalKnobs.length;
    const idealPerRow = nn <= 3 ? nn : nn <= 6 ? 3 : nn <= 9 ? 3 : 4;
    const heroW = heroKnobs.length > 0 ? Math.round(kw * 1.4) + gap : 0;
    const rightColW = heroW + idealPerRow * (kw + gap);
    return leftW + gap * 2 + rightColW + innerPad;
  }

  // Check for prefix-grouped components (sub-column layout)
  // Include buttons + LEDs in detection — matches positionComponents group logic
  const allGroupable = [...knobs, ...buttons, ...comps.filter(c => c.type === 'led')];
  const prefixMap = {};
  for (const k of allGroupable) {
    const prefix = (k.label || '').split(/\s+/)[0];
    if (prefix && prefix.length >= 2) {
      if (!prefixMap[prefix]) prefixMap[prefix] = [];
      prefixMap[prefix].push(k);
    }
  }
  const groups = Object.entries(prefixMap).filter(([, items]) => items.length >= 2);

  if (groups.length >= 2) {
    // Grouped sub-columns layout
    const numGroups = groups.length;
    const maxPerGroup = Math.max(...groups.map(([, items]) => items.length));
    const knobsPerGroupRow = Math.min(maxPerGroup, 3);
    const groupColW = knobsPerGroupRow * (kw + gap) + 16;
    return numGroups * groupColW + (numGroups - 1) * gap + innerPad * 2;
  }

  // Standard knob grid — aim for compact rows
  const idealKnobsPerRow = nk <= 2 ? nk : nk <= 4 ? nk : nk <= 8 ? 4 : nk <= 12 ? 4 : 5;
  const knobRowW = idealKnobsPerRow > 0 ? idealKnobsPerRow * (kw + gap) : 0;

  const ddMaxW = dropdowns.reduce((s, d) => s + (d.width || (sizes.dropdown || sizes.knob).w) + gap, 0);
  const sliderRowW = sliders.reduce((s, sl) => s + (sl.width || (sizes.slider || sizes.knob).w) + gap, 0);
  const btnRowW = buttons.reduce((s, b) => s + (b.width || (sizes.button || sizes.knob).w) + gap, 0);

  const contentW = Math.max(displayMaxW, knobRowW, ddMaxW, sliderRowW, btnRowW);
  return Math.max(contentW + innerPad * 2, 100);
}

// ── Post-layout: shrink panels to content, close horizontal gaps ────────────
// After layout, panels may be wider than their children. This pass:
// 1. Shrinks each top-level panel to tightly wrap its children
// 2. Groups panels by row and closes any horizontal gaps
// Result: no wasted space in panels, canvas auto-fits to actual content.

function shrinkPanelsToContent(allComps, pad, secGap) {
  // Safety-net pass: close horizontal gaps between panels in each row.
  // Width/height are now set correctly by the layout functions (proportional fill),
  // so we only need to fix any leftover gaps from rounding.
  // Only consider panels in the same tab (or non-tabbed panels)
  const topPanels = allComps.filter(c => c.type === 'panel' && c.zIndex === 0);
  if (topPanels.length === 0) return;

  const findChildren = (panel) => allComps.filter(c => {
    if (c === panel || (c.type === 'panel' && c.zIndex === 0)) return false;
    // Only match children in the same tab
    if (panel.tabIndex !== undefined && c.tabIndex !== undefined && panel.tabIndex !== c.tabIndex) return false;
    const cx = c.x + (c.width || 0) / 2;
    const cy = c.y + (c.height || 0) / 2;
    return cx >= panel.x && cx <= panel.x + panel.width &&
           cy >= panel.y && cy <= panel.y + panel.height;
  });

  // Group panels by row (similar Y coordinate) AND same tab
  const rows = [];
  const sorted = [...topPanels].sort((a, b) => a.y - b.y);
  let curRow = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const sameTab = sorted[i].tabIndex === curRow[0].tabIndex;
    if (sameTab && Math.abs(sorted[i].y - curRow[0].y) < 20) {
      curRow.push(sorted[i]);
    } else {
      rows.push(curRow.sort((a, b) => a.x - b.x));
      curRow = [sorted[i]];
    }
  }
  rows.push(curRow.sort((a, b) => a.x - b.x));

  // Close horizontal gaps within each row
  for (const row of rows) {
    if (row.length <= 1) continue;
    for (let i = 1; i < row.length; i++) {
      const prevPanel = row[i - 1];
      const panel = row[i];
      const expectedX = prevPanel.x + prevPanel.width + secGap;
      const excess = panel.x - expectedX;
      if (excess > 4) {
        const children = findChildren(panel);
        panel.x -= excess;
        for (const child of children) child.x -= excess;
      }
    }
  }
}

// ── Main layout function ─────────────────────────────────────────────────────

/**
 * Compute a full layout from a Design Brief.
 * @param {Object} brief - Design brief from the Design Director
 * @param {number} canvasW - Canvas width (from brief or default)
 * @param {number} canvasH - Canvas height (from brief or default)
 * @returns {{ pluginConfig: Object, components: Array }} - Same shape as pluginlang
 */
// Max canvas dimensions — plugins should fit comfortably on screen
const MAX_CANVAS_W = 1100;
const MAX_CANVAS_H = 800;

export function computeLayout(brief, canvasW = 600, canvasH = 400) {
  // Use brief dimensions when available (callers may omit explicit size)
  canvasW = brief.width || canvasW;
  canvasH = brief.height || canvasH;
  // Clamp canvas width; height is determined by content (auto-fit below)
  canvasW = Math.min(canvasW, MAX_CANVAS_W);

  // ── Handle tabbed layout: tabs + persistentSections ──
  // When the brief uses "tabs" instead of "sections", flatten all tab sections
  // and persistent sections into one array, tracking which tab each belongs to.
  let _tabMeta = null;
  if (brief.tabs && Array.isArray(brief.tabs) && brief.tabs.length > 0) {
    const flatSections = [];
    const tabDefs = [];
    for (const tab of brief.tabs) {
      const startIdx = flatSections.length;
      const tabSections = tab.sections || [];
      flatSections.push(...tabSections);
      tabDefs.push({
        label: tab.label || `Tab ${tabDefs.length + 1}`,
        sectionIndices: tabSections.map((_, i) => startIdx + i),
      });
    }
    const persistentStart = flatSections.length;
    const persistent = brief.persistentSections || [];
    flatSections.push(...persistent);
    _tabMeta = {
      tabs: tabDefs,
      persistentIndices: persistent.map((_, i) => persistentStart + i),
    };
    brief.sections = flatSections;
    brief.layout = 'tabbed';
  }

  // ── Flatten nested sub-sections into top-level sections ──
  // If a section has `subSections`, expand each into its own section
  // with the parent label as prefix. This lets the AI define grouped layouts.
  let sections = brief.sections || [];
  const expanded = [];
  for (const sec of sections) {
    if (sec.subSections && Array.isArray(sec.subSections)) {
      for (const sub of sec.subSections) {
        expanded.push({
          ...sub,
          label: sub.label || `${sec.label} ${expanded.length + 1}`,
          parentGroup: sec.label,
          weight: sub.weight || (sec.weight || 1) / sec.subSections.length,
        });
      }
    } else {
      expanded.push(sec);
    }
  }
  sections = expanded;
  if (sections.length === 0) return { pluginConfig: buildConfig(brief, null), components: [] };

  // Generate theme for color resolution
  const theme = generateTheme(brief.accentColor || '#667eea', brief.aesthetic || 'pro-studio');

  // Debug: save generation params for inspection
  try {
    window.__layoutEngineDebug = {
      brief: JSON.parse(JSON.stringify(brief)),
      theme: JSON.parse(JSON.stringify(theme)),
      canvasW, canvasH,
      timestamp: new Date().toISOString(),
    };
    console.log('[LayoutEngine] Brief:', brief);
    console.log('[LayoutEngine] Theme:', theme);
  } catch (e) { /* noop */ }

  // Decide layout strategy based on section count and canvas shape
  const aspectRatio = canvasW / canvasH;
  const layoutMode = brief.layout || chooseLayoutMode(sections.length, aspectRatio);

  let allComponents = [];

  if (layoutMode === 'tabbed') {
    allComponents = layoutTabbed(sections, canvasW, canvasH, theme, brief, _tabMeta);
  } else if (layoutMode === 'horizontal') {
    allComponents = layoutHorizontal(sections, canvasW, canvasH, theme, brief);
  } else if (layoutMode === 'grid') {
    allComponents = layoutGrid(sections, canvasW, canvasH, theme, brief);
  } else {
    allComponents = layoutVertical(sections, canvasW, canvasH, theme, brief);
  }

  // ── Shrink panels to actual content + close gaps ──
  const { isDense: _d, isMedium: _m } = assessDensity(sections, canvasW, canvasH);
  const fitPad = _d ? COMPACT_PADDING : _m ? MEDIUM_PADDING : PADDING;
  const fitSecGap = _d ? COMPACT_SECTION_GAP : _m ? MEDIUM_SECTION_GAP : SECTION_GAP;
  shrinkPanelsToContent(allComponents, fitPad, fitSecGap);

  // ── Auto-fit canvas to actual content ──
  const isTabbed = allComponents._tabMeta && allComponents._tabMeta.tabCount > 0;
  let maxContentY = TITLE_BAR_HEIGHT;
  let maxContentX = 0;
  if (isTabbed) {
    // Measure ALL tabs — canvas must fit the tallest tab's content
    const tabCount = allComponents._tabMeta.tabCount;
    for (let t = 0; t < tabCount; t++) {
      for (const comp of allComponents) {
        if (comp.tabIndex === t || comp.tabIndex === -1 || comp.isTabButton) {
          maxContentY = Math.max(maxContentY, (comp.y || 0) + (comp.height || 0));
          maxContentX = Math.max(maxContentX, (comp.x || 0) + (comp.width || 0));
        }
      }
    }
  } else {
    for (const comp of allComponents) {
      maxContentY = Math.max(maxContentY, (comp.y || 0) + (comp.height || 0));
      maxContentX = Math.max(maxContentX, (comp.x || 0) + (comp.width || 0));
    }
  }
  if (isTabbed) {
    // Preserve brief dimensions — panels fill the tab area, canvas matches brief
    canvasW = brief.width || canvasW;
    // Use recommended height from layout engine — shrinks canvas to fit content
    canvasH = allComponents._tabMeta?.recommendedHeight || brief.height || canvasH;
    // Only expand if content actually overflows (safety net)
    canvasW = Math.max(canvasW, maxContentX + fitPad + 4);
    canvasH = Math.max(canvasH, maxContentY + fitPad + 4);
    canvasH = Math.min(canvasH, MAX_CANVAS_H);
  } else {
    canvasH = maxContentY + fitPad + 4;
    // EXPAND canvas to fit content — never clip panels
    canvasW = Math.max(400, maxContentX + fitPad + 4);
  }

  // Apply snap pass for consistent alignment
  // Snap grid is always small (2-4px) for precise positioning — independent of component gap
  const snapGridSize = _d ? COMPACT_COMPONENT_GAP : _m ? 2 : COMPONENT_GAP;
  const snappedComponents = snapLayout(allComponents, snapGridSize);

  // Post-layout collision resolution — fix any overlaps from snap quantization
  resolveOverlaps(snappedComponents);

  // Row alignment pass — snap components within a small Y tolerance to the same Y.
  // This fixes misalignment caused by overlap resolution nudging or quantization drift.
  // Only aligns interactive components (knobs, sliders, buttons, dropdowns) within same tab.
  const alignSkipTypes = new Set(['panel', 'label', 'led', 'waveform', 'wavetable_3d', 'oscilloscope', 'spectrum-analyzer', 'adsr', 'image']);
  const alignable = snappedComponents.filter(c => !alignSkipTypes.has(c.type));
  const ROW_SNAP_TOLERANCE = 10; // snap within 10px
  for (let i = 0; i < alignable.length; i++) {
    const a = alignable[i];
    // Find nearby components on approximately the same row
    const rowPeers = [a];
    for (let j = i + 1; j < alignable.length; j++) {
      const b = alignable[j];
      if (a.tabIndex !== undefined && b.tabIndex !== undefined && a.tabIndex !== b.tabIndex) continue;
      if (Math.abs(a.y - b.y) <= ROW_SNAP_TOLERANCE && a.height === b.height) {
        rowPeers.push(b);
      }
    }
    if (rowPeers.length >= 2) {
      // Snap all to the most common Y (mode)
      const yCounts = {};
      for (const c of rowPeers) {
        yCounts[c.y] = (yCounts[c.y] || 0) + 1;
      }
      const modeY = Number(Object.entries(yCounts).sort((a, b) => b[1] - a[1])[0][0]);
      for (const c of rowPeers) c.y = modeY;
    }
  }

  // ── Final clamp: ensure no content extends past panel bounds ──
  // Post-processing (snap, overlap resolution, row alignment) can push
  // components outside their parent panels. Clamp them back.
  const panels = snappedComponents.filter(c => c.type === 'panel');
  const nonPanels = snappedComponents.filter(c => c.type !== 'panel' && c.type !== 'label' && !c.isTabButton);
  for (const comp of nonPanels) {
    // Find the parent panel (smallest panel that contains this component's origin)
    let parentPanel = null;
    let parentArea = Infinity;
    for (const p of panels) {
      if (p.tabIndex !== undefined && comp.tabIndex !== undefined && p.tabIndex !== comp.tabIndex) continue;
      if (comp.x >= p.x && comp.x < p.x + p.width && comp.y >= p.y && comp.y < p.y + p.height) {
        const area = p.width * p.height;
        if (area < parentArea) { parentPanel = p; parentArea = area; }
      }
    }
    if (!parentPanel) continue;
    const pBottom = parentPanel.y + parentPanel.height - 2;
    const pRight = parentPanel.x + parentPanel.width - 2;
    if (comp.y + comp.height > pBottom) {
      if (comp.y < pBottom - 8) {
        comp.height = Math.max(8, pBottom - comp.y);
      } else {
        comp.y = Math.max(parentPanel.y + 4, pBottom - comp.height);
      }
    }
    if (comp.x + comp.width > pRight) {
      comp.width = Math.max(8, pRight - comp.x);
    }
  }

  const result = {
    pluginConfig: buildConfig(brief, theme),
    components: snappedComponents,
    mode: 'replace',
  };
  // Override dimensions with auto-fitted values
  result.pluginConfig.height = canvasH;
  result.pluginConfig.width = canvasW;

  // Attach tab metadata to pluginConfig for downstream consumers (JUCE code generator)
  if (allComponents._tabMeta) {
    result.pluginConfig.tabCount = allComponents._tabMeta.tabCount;
    result.pluginConfig.tabLabels = allComponents._tabMeta.tabLabels;
  }

  // Debug: save final layout output
  try {
    window.__layoutEngineDebug.output = {
      pluginConfig: result.pluginConfig,
      componentCount: allComponents.length,
      componentSummary: allComponents.map(c => ({
        type: c.type, label: c.label, svgStyle: c.svgStyle || '',
        hasSvg: !!c.svg, bodyColor: c.bodyColor, indicatorColor: c.indicatorColor,
        w: c.width, h: c.height,
      })),
    };
    console.log('[LayoutEngine] Output:', result.pluginConfig, `${allComponents.length} components`);
    console.log('[LayoutEngine] Auto-fit: brief', brief.width + 'x' + brief.height, '→ fitted', canvasW + 'x' + canvasH, '| contentMax:', maxContentX + 'x' + maxContentY);
    console.log('[LayoutEngine] SVG components:', allComponents.filter(c => c.svg).length, 'have SVGs');
  } catch (e) { /* noop */ }

  return result;
}

function buildConfig(brief, theme) {
  const bgPrompt = (theme && theme.dallePrompt) || brief.backgroundPrompt || '';
  return {
    name: brief.pluginName || 'My Plugin',
    width: Math.min(brief.width || 600, MAX_CANVAS_W),
    height: Math.min(brief.height || 400, MAX_CANVAS_H),
    bgColor: brief.bgColor || '#1a1a2e',
    titleBarColor: brief.titleBarColor || '#2d2d4e',
    bgImage: bgPrompt ? { generate: sanitizeBackgroundPrompt(bgPrompt) } : '',
  };
}

function chooseLayoutMode(sectionCount, aspectRatio) {
  if (sectionCount <= 3 && aspectRatio >= 1.2) return 'horizontal';
  if (sectionCount >= 4) return 'grid';
  return 'horizontal';
}

// ── Tabbed layout — sections share space with navigation tabs ────────────────
function layoutTabbed(sections, canvasW, canvasH, theme, brief, tabMeta) {
  // Tabbed layouts are ALWAYS dense — Serum-class UIs pack everything tight
  const pad = 6;
  const secGap = 4;
  const secLabelH = 12;
  const compGap = 6;
  const panelOverhead = secLabelH + 10;
  const TAB_BAR_HEIGHT = 26;
  const tabBarY = TITLE_BAR_HEIGHT + 4;
  const contentY = tabBarY + TAB_BAR_HEIGHT + 4;

  // If no tabMeta, fall back to old behavior (each section = one tab)
  if (!tabMeta) {
    tabMeta = {
      tabs: sections.map((s, i) => ({ label: s.label || `Tab ${i + 1}`, sectionIndices: [i] })),
      persistentIndices: [],
    };
  }

  const tabCount = tabMeta.tabs.length;
  const usableW = canvasW - pad * 2;
  const tabW = Math.min(100, Math.floor((usableW - (tabCount - 1) * 2) / tabCount));

  // Persistent section: compact bottom strip (Serum-style macro bar)
  const persistentSections = tabMeta.persistentIndices.map(i => sections[i]).filter(Boolean);
  const hasPersistent = persistentSections.length > 0;
  const persistentCompCount = persistentSections.reduce((s, sec) => s + (sec.components?.length || 0), 0);
  // Compact height — one row of knobs + label + breathing room
  const PERSISTENT_HEIGHT = hasPersistent ? 56 : 0;
  const tabContentH = canvasH - contentY - PERSISTENT_HEIGHT - (hasPersistent ? 4 : 0) - pad;

  const result = [];

  // Tab buttons — compact strip at top
  for (let i = 0; i < tabCount; i++) {
    result.push({
      type: 'button',
      label: tabMeta.tabs[i].label.toUpperCase(),
      x: pad + i * (tabW + 2),
      y: tabBarY,
      width: tabW,
      height: TAB_BAR_HEIGHT,
      color: i === 0 ? theme.accentColor : 'rgba(255,255,255,0.15)',
      fontSize: 10,
      borderRadius: 3,
      opacity: 1,
      zIndex: 5,
      tabIndex: -1,
      isTabButton: true,
      tabTargetIndex: i,
    });
  }

  // ── Pre-pass: Measure ALL tabs to determine panel heights ──
  // Track horizontal (single-row) and grid (multi-row) tab heights separately.
  // Horizontal tabs share tallestHorizPanelH so all have consistent panel height.
  // Grid tabs use their natural row heights (not inflated by horizontal tabs).
  // The persistent strip position uses the overall max (tallestTabAreaH).
  // IMPORTANT: Deep-clone components so pre-pass doesn't mutate originals.
  let tallestHorizPanelH = 0;  // Max single-section height across horizontal tabs
  let tallestTabAreaH = 0;     // Max total tab area height (for persistent strip)
  const tabGridRowHeights = {}; // Per-tab grid row heights (for grid tabs only)
  for (let ti = 0; ti < tabCount; ti++) {
    const preSections = tabMeta.tabs[ti].sectionIndices.map(i => sections[i]).filter(Boolean);
    if (preSections.length === 0) continue;
    const { isDense: preDense } = assessDensity(preSections, canvasW, canvasH);
    const preIsMedium = !preDense;
    const preInnerPad = preDense ? COMPACT_PADDING : MEDIUM_PADDING;
    const prePanelOH = secLabelH + (preDense ? 14 : 4);
    const preIdealWidths = preSections.map(sec => {
      const compact = isSectionCompact(sec, preDense, preIsMedium);
      const sectionSizes = getSectionSizes(sec, preDense, preIsMedium);
      const sGap = compact === 'medium' ? MEDIUM_COMPONENT_GAP : compact ? COMPACT_COMPONENT_GAP : COMPONENT_GAP;
      return computeIdealSectionWidth(sec, sectionSizes, sGap, preInnerPad, sec.layout || null);
    });
    if (preSections.length <= 3) {
      const totalIdealW = preIdealWidths.reduce((s, w) => s + w, 0);
      const availForContent = usableW - (preSections.length - 1) * secGap;
      let colWidths;
      if (availForContent <= totalIdealW) {
        colWidths = preIdealWidths.map(w => Math.floor(w * availForContent / totalIdealW));
      } else {
        const extraSpace = availForContent - totalIdealW;
        const counts = preSections.map(sec => (sec.components || []).length);
        const totalCounts = counts.reduce((s, c) => s + c, 0);
        colWidths = preIdealWidths.map((w, i) => {
          const share = totalCounts > 0 ? counts[i] / totalCounts : 1 / preSections.length;
          return Math.floor(w + extraSpace * share);
        });
      }
      let maxH = 50;
      for (let i = 0; i < preSections.length; i++) {
        const section = preSections[i];
        const compact = isSectionCompact(section, preDense, preIsMedium);
        const innerW = colWidths[i] - preInnerPad;
        const clonedComps = (section.components || []).map(c => ({...c}));
        const { contentHeight } = layoutSectionContent(
          clonedComps, innerW, theme, brief, compact, section.layout || null, section
        );
        maxH = Math.max(maxH, contentHeight + prePanelOH);
      }
      tallestHorizPanelH = Math.max(tallestHorizPanelH, maxH);
      tallestTabAreaH = Math.max(tallestTabAreaH, maxH);
    } else {
      const cols = preSections.length >= 8 ? 3 : 2;
      const numRows = Math.ceil(preSections.length / cols);
      let totalGridH = 0;
      const rowHeightsArr = [];
      for (let r = 0; r < numRows; r++) {
        const rowStart = r * cols;
        const rowSecs = [];
        const rowIdealW = [];
        for (let c = 0; c < cols; c++) {
          const idx = rowStart + c;
          if (idx < preSections.length) {
            rowSecs.push(preSections[idx]);
            rowIdealW.push(preIdealWidths[idx]);
          }
        }
        const totalIdealW = rowIdealW.reduce((s, w) => s + w, 0);
        const availForContent = usableW - (rowSecs.length - 1) * secGap;
        let colWidths;
        if (availForContent <= totalIdealW) {
          colWidths = rowIdealW.map(w => Math.floor(w * availForContent / totalIdealW));
        } else {
          const extraSpace = availForContent - totalIdealW;
          const counts = rowSecs.map(sec => (sec.components || []).length);
          const totalCounts = counts.reduce((s, c) => s + c, 0);
          colWidths = rowIdealW.map((w, i) => {
            const share = totalCounts > 0 ? counts[i] / totalCounts : 1 / rowSecs.length;
            return Math.floor(w + extraSpace * share);
          });
        }
        let rowMaxH = 50;
        for (let i = 0; i < rowSecs.length; i++) {
          const section = rowSecs[i];
          const compact = isSectionCompact(section, preDense, preIsMedium);
          const innerW = colWidths[i] - preInnerPad;
          const clonedComps = (section.components || []).map(c => ({...c}));
          const { contentHeight } = layoutSectionContent(
            clonedComps, innerW, theme, brief, compact, section.layout || null, section
          );
          rowMaxH = Math.max(rowMaxH, contentHeight + prePanelOH);
        }
        rowHeightsArr.push(rowMaxH);
        totalGridH += rowMaxH;
      }
      totalGridH += (numRows - 1) * secGap;
      tabGridRowHeights[ti] = rowHeightsArr;
      tallestTabAreaH = Math.max(tallestTabAreaH, totalGridH);
    }
  }
  // Small headroom
  tallestHorizPanelH += Math.min(16, Math.floor(tallestHorizPanelH * 0.03));
  tallestTabAreaH += Math.min(16, Math.floor(tallestTabAreaH * 0.03));
  // Horizontal panels should be at least as tall as any grid ROW (not total)
  // to avoid horizontal tabs being shorter than individual grid rows
  for (const rowHArr of Object.values(tabGridRowHeights)) {
    for (const rh of rowHArr) {
      tallestHorizPanelH = Math.max(tallestHorizPanelH, rh);
    }
  }
  // The tallest tab area must accommodate horizontal panel height too
  tallestTabAreaH = Math.max(tallestTabAreaH, tallestHorizPanelH);

  // ── Main pass: Layout each tab using tallestPanelH ──
  for (let ti = 0; ti < tabCount; ti++) {
    const tabSectionIndices = tabMeta.tabs[ti].sectionIndices;
    const tabSections = tabSectionIndices.map(i => sections[i]).filter(Boolean);
    if (tabSections.length === 0) continue;

    // Force at least medium density for tab content (Serum-class UIs)
    const { isDense: tabDense } = assessDensity(tabSections, canvasW, canvasH);
    const isDense = tabDense;
    const isMedium = !tabDense;
    const innerPad = isDense ? COMPACT_PADDING : MEDIUM_PADDING;
    const panelOH = secLabelH + (isDense ? 14 : 4);

    // Compute content-fit column widths for this tab's sections
    const idealWidths = tabSections.map(sec => {
      const compact = isSectionCompact(sec, isDense, isMedium);
      const sectionSizes = getSectionSizes(sec, isDense, isMedium);
      const sGap = compact === 'medium' ? MEDIUM_COMPONENT_GAP : compact ? COMPACT_COMPONENT_GAP : COMPONENT_GAP;
      return computeIdealSectionWidth(sec, sectionSizes, sGap, innerPad, sec.layout || null);
    });

    if (tabSections.length <= 3) {
      // ── Horizontal strip — sections side by side ──
      const totalIdealW = idealWidths.reduce((s, w) => s + w, 0);
      const availForContent = usableW - (tabSections.length - 1) * secGap;
      let colWidths;
      if (availForContent <= totalIdealW) {
        const scale = availForContent / totalIdealW;
        colWidths = idealWidths.map(w => Math.floor(w * scale));
      } else {
        const extraSpace = availForContent - totalIdealW;
        const counts = tabSections.map(sec => (sec.components || []).length);
        const totalCounts = counts.reduce((s, c) => s + c, 0);
        colWidths = idealWidths.map((w, i) => {
          const share = totalCounts > 0 ? counts[i] / totalCounts : 1 / tabSections.length;
          return Math.floor(w + extraSpace * share);
        });
      }

      // Pass 1: Position at origin, measure heights
      const sectionData = [];
      let maxH = 50;
      for (let i = 0; i < tabSections.length; i++) {
        const section = tabSections[i];
        const compact = isSectionCompact(section, isDense, isMedium);
        const innerW = colWidths[i] - innerPad;
        const { comps, contentHeight } = layoutSectionContent(
          section.components || [], innerW, theme, brief, compact, section.layout || null, section
        );
        sectionData.push({ comps, contentHeight, section });
        maxH = Math.max(maxH, contentHeight + panelOH);
      }
      // Horizontal tabs use tallestTabAreaH — matches the total height of the tallest
      // tab (including grid tabs). This ensures consistent gap to the persistent strip.
      // The redistribution in finalizeSection handles the extra space (stretches displays,
      // spreads knob rows evenly).
      const panelH = tallestTabAreaH;

      // Pass 2: Finalize at contentY directly (no TITLE_BAR_HEIGHT offset)
      // Top-align panels — content-fit means no wasted vertical space
      const panelStartY = contentY;
      const totalRowW = colWidths.reduce((s, w) => s + w, 0) + (tabSections.length - 1) * secGap;
      let xOff = totalRowW < usableW
        ? pad + Math.floor((usableW - totalRowW) / 2)
        : pad;
      for (let i = 0; i < tabSections.length; i++) {
        const { comps: sComps, contentHeight, section } = sectionData[i];
        const sectionW = colWidths[i];
        const sectionX = xOff;
        const sectionY = panelStartY;

        const finalized = finalizeSection(
          sComps, contentHeight, section,
          sectionX, sectionY, sectionW, panelH,
          panelOH, secLabelH, isDense, isMedium, theme
        );
        // Stretch display components to fill gap — dynamic cap based on available space
        const stretchTypes = ['waveform', 'wavetable_3d', 'oscilloscope', 'spectrum-analyzer', 'adsr'];
        // Compute how much space non-display content uses
        const ndComps = finalized.filter(c => !stretchTypes.includes(c.type) && c.type !== 'panel' && c.type !== 'label');
        const ndUsedH = ndComps.length > 0
          ? Math.max(...ndComps.map(c => c.y + c.height)) - Math.min(...ndComps.map(c => c.y))
          : 0;
        // Check if display is side-by-side with controls (display-controls layout)
        const displayComps = finalized.filter(c => stretchTypes.includes(c.type));
        const isSideBySide = displayComps.length > 0 && ndComps.length > 0 &&
          ndComps.some(c => c.y < (displayComps[0].y + displayComps[0].height) &&
            c.x >= displayComps[0].x + displayComps[0].width - 10);
        // Side-by-side: display can fill 70% (doesn't compete with controls vertically)
        // Stacked: display limited to 50% (controls are below)
        const capFraction = isSideBySide ? 0.70 : 0.50;
        const dynMaxH = Math.max(Math.floor(panelH * capFraction), panelH - ndUsedH - 20);
        const maxDisplayH = Math.min(dynMaxH, 200);
        for (const comp of finalized) {
          if (stretchTypes.includes(comp.type)) {
            if (comp.height > maxDisplayH) comp.height = maxDisplayH;
            const panelBottom = sectionY + panelH - 6;
            let stretchBottom = Math.min(panelBottom, comp.y + maxDisplayH);
            for (const other of finalized) {
              if (other === comp || other.type === 'panel' || other.type === 'label') continue;
              if (stretchTypes.includes(other.type)) continue;
              if (other.y > comp.y + comp.height &&
                  other.x < comp.x + comp.width && other.x + other.width > comp.x) {
                stretchBottom = Math.min(stretchBottom, other.y - 4);
              }
            }
            const newH = stretchBottom - comp.y;
            if (newH > comp.height) comp.height = newH;
            if (comp.height > maxDisplayH) comp.height = maxDisplayH;
          }
        }
        for (const comp of finalized) {
          comp.tabIndex = ti;
          if (ti > 0) comp.opacity = 0;
        }
        result.push(...finalized);
        xOff += sectionW + secGap;
      }
    } else {
      // ── Grid layout for 4+ sections — fill tab area proportionally ──
      const cols = tabSections.length >= 8 ? 3 : 2;
      const numRows = Math.ceil(tabSections.length / cols);

      // Phase 1: Measure all rows' content heights and column widths
      const rowMeasurements = [];
      for (let r = 0; r < numRows; r++) {
        const rowStart = r * cols;
        const rowSecs = [];
        const rowIdealW = [];
        for (let c = 0; c < cols; c++) {
          const idx = rowStart + c;
          if (idx < tabSections.length) {
            rowSecs.push(tabSections[idx]);
            rowIdealW.push(idealWidths[idx]);
          }
        }
        const totalIdealW = rowIdealW.reduce((s, w) => s + w, 0);
        const availForContent = usableW - (rowSecs.length - 1) * secGap;
        let colWidths;
        if (availForContent <= totalIdealW) {
          const scale = availForContent / totalIdealW;
          colWidths = rowIdealW.map(w => Math.floor(w * scale));
        } else {
          const extraSpace = availForContent - totalIdealW;
          const counts = rowSecs.map(sec => (sec.components || []).length);
          const totalCounts = counts.reduce((s, c) => s + c, 0);
          colWidths = rowIdealW.map((w, i) => {
            const share = totalCounts > 0 ? counts[i] / totalCounts : 1 / rowSecs.length;
            return Math.floor(w + extraSpace * share);
          });
        }
        const rowData = [];
        let rowMaxH = 50;
        for (let i = 0; i < rowSecs.length; i++) {
          const section = rowSecs[i];
          const compact = isSectionCompact(section, isDense, isMedium);
          const innerW = colWidths[i] - innerPad;
          const { comps, contentHeight } = layoutSectionContent(
            section.components || [], innerW, theme, brief, compact, section.layout || null, section
          );
          rowData.push({ comps, contentHeight, section });
          rowMaxH = Math.max(rowMaxH, contentHeight + panelOH);
        }
        rowMeasurements.push({ rowSecs, colWidths, rowData, rowContentH: rowMaxH });
      }

      // Phase 2: Use natural row heights — take the MAX of pre-pass and main-pass
      // measurements for safety (ensures panels are big enough for content).
      // Add small headroom so content doesn't sit pixel-tight against panel edge.
      const preRowH = tabGridRowHeights[ti];
      const rowHeights = rowMeasurements.map((m, r) => {
        const preH = preRowH ? preRowH[r] : 0;
        const h = Math.max(m.rowContentH, preH);
        return Math.max(50, h + 4); // 4px headroom
      });

      // Phase 3: Position all rows with proportional heights + display stretch
      let yOff = contentY;
      for (let r = 0; r < numRows; r++) {
        const { rowSecs: rSections, colWidths, rowData } = rowMeasurements[r];
        const rowH = rowHeights[r];
        const totalRowW = colWidths.reduce((s, w) => s + w, 0) + (rSections.length - 1) * secGap;
        let xOff = totalRowW < usableW
          ? pad + Math.floor((usableW - totalRowW) / 2)
          : pad;
        for (let i = 0; i < rSections.length; i++) {
          const { comps: sComps, contentHeight, section } = rowData[i];
          const sectionW = colWidths[i];
          const sectionX = xOff;
          const sectionY = yOff;
          const finalized = finalizeSection(
            sComps, contentHeight, section,
            sectionX, sectionY, sectionW, rowH,
            panelOH, secLabelH, isDense, isMedium, theme
          );
          // Stretch display components to fill gap — dynamic cap based on available space
          const stretchTypes = ['waveform', 'wavetable_3d', 'oscilloscope', 'spectrum-analyzer', 'adsr'];
          const ndGridComps = finalized.filter(c => !stretchTypes.includes(c.type) && c.type !== 'panel' && c.type !== 'label');
          const ndGridUsedH = ndGridComps.length > 0
            ? Math.max(...ndGridComps.map(c => c.y + c.height)) - Math.min(...ndGridComps.map(c => c.y))
            : 0;
          // Check if display is side-by-side with controls
          const gridDisplayComps = finalized.filter(c => stretchTypes.includes(c.type));
          const gridSideBySide = gridDisplayComps.length > 0 && ndGridComps.length > 0 &&
            ndGridComps.some(c => c.y < (gridDisplayComps[0].y + gridDisplayComps[0].height) &&
              c.x >= gridDisplayComps[0].x + gridDisplayComps[0].width - 10);
          const gridCapFraction = gridSideBySide ? 0.70 : 0.50;
          const dynGridMaxH = Math.max(Math.floor(rowH * gridCapFraction), rowH - ndGridUsedH - 20);
          const maxDisplayH = Math.min(dynGridMaxH, 200);
          for (const comp of finalized) {
            if (stretchTypes.includes(comp.type)) {
              if (comp.height > maxDisplayH) comp.height = maxDisplayH;
              const panelBottom = sectionY + rowH - 6;
              let stretchBottom = Math.min(panelBottom, comp.y + maxDisplayH);
              for (const other of finalized) {
                if (other === comp || other.type === 'panel' || other.type === 'label') continue;
                if (stretchTypes.includes(other.type)) continue;
                if (other.y > comp.y + comp.height &&
                    other.x < comp.x + comp.width && other.x + other.width > comp.x) {
                  stretchBottom = Math.min(stretchBottom, other.y - 4);
                }
              }
              const newH = stretchBottom - comp.y;
              if (newH > comp.height) comp.height = newH;
              if (comp.height > maxDisplayH) comp.height = maxDisplayH;
            }
          }
          for (const comp of finalized) {
            comp.tabIndex = ti;
            if (ti > 0) comp.opacity = 0;
          }
          result.push(...finalized);
          xOff += sectionW + secGap;
        }
        yOff += rowH + secGap;
      }
    }
  }

  // ── Compute optimal canvas height from tallest tab area ──
  // Canvas shrinks to fit content — no empty space below panels.
  const optimalCanvasH = contentY + tallestTabAreaH + (hasPersistent ? 4 + PERSISTENT_HEIGHT : 0) + pad;

  // Persistent section — compact bottom bar (like Serum's macro/master strip)
  if (hasPersistent) {
    // Flatten all persistent components into a single horizontal row
    const allPersistComps = [];
    for (const sec of persistentSections) {
      allPersistComps.push(...(sec.components || []));
    }

    if (allPersistComps.length > 0) {
      // Position at optimal canvas bottom — tight fit below panels
      const persistY = optimalCanvasH - PERSISTENT_HEIGHT - pad;

      // Background panel for persistent strip
      result.push({
        type: 'panel', label: '',
        x: pad, y: persistY,
        width: canvasW - pad * 2, height: PERSISTENT_HEIGHT,
        bgColor: theme.panelColor, borderColor: theme.panelBorder,
        borderRadius: 4, zIndex: 0, opacity: 1, tabIndex: -1,
      });

      // Section label
      const persLabel = persistentSections[0]?.label || 'MASTER';
      result.push({
        type: 'label', label: persLabel.toUpperCase(),
        x: pad + 4, y: persistY + 2,
        width: 60, height: 10,
        color: theme.textColor, fontSize: 10, zIndex: 10, opacity: 1,
        fontWeight: 600, letterSpacing: '0.08em', tabIndex: -1,
      });

      // Lay out persistent components in a single horizontal row
      const sizes = MEDIUM_TYPE_SIZES;
      const innerY = persistY + 14;
      const availH = PERSISTENT_HEIGHT - 16;
      let cx = pad + 6;
      for (const comp of allPersistComps) {
        const size = sizes[comp.type] || sizes.knob;
        const scale = comp.size === 'large' ? 1.2 : comp.size === 'small' ? 0.8 : 1;
        const w = Math.round(size.w * scale);
        const h = Math.min(Math.round(size.h * scale), availH);

        const built = {
          type: comp.type, label: comp.label || '',
          x: cx, y: innerY,
          width: w, height: h,
          color: theme.accentColor,
          svgStyle: comp.svgStyle || theme[comp.type === 'knob' ? 'knobSvgStyle' : comp.type === 'slider' ? 'sliderSvgStyle' : 'buttonSvgStyle'] || '',
          bodyColor: '', indicatorColor: '', accentColor: '',
          opacity: 1, rotation: 0, borderRadius: 0,
          fontSize: 9, zIndex: 2, tabIndex: -1,
        };

        // Generate SVGs for interactive components
        if (['knob', 'slider', 'button'].includes(comp.type)) {
          const actualBgColor = brief.bgColor || theme.bgColor;
          built.bodyColor = adjustColorForBody(actualBgColor);
          built.indicatorColor = theme.accentColor;
          built.accentColor = adjustColorForAccent(theme.accentColor);
          if (!built.svgStyle) {
            if (comp.type === 'knob') built.svgStyle = theme.knobSvgStyle;
            else if (comp.type === 'slider') built.svgStyle = theme.sliderSvgStyle;
            else if (comp.type === 'button') built.svgStyle = theme.buttonSvgStyle;
          }
          const uid = Math.random().toString(36).slice(2, 10);
          const params = {
            width: w, height: h,
            bodyColor: built.bodyColor, indicatorColor: built.indicatorColor,
            accentColor: built.accentColor, uid, label: built.label,
          };
          if (comp.type === 'knob') {
            let svg = generateKnobSVG(built.svgStyle, params);
            if (svg) {
              const bgDark = darkenHex(actualBgColor, 0.4);
              const bgMid = darkenHex(actualBgColor, 0.25);
              svg = svg.replace(
                /(<svg[^>]*>)/,
                `$1\n  <circle cx="${w / 2}" cy="${h / 2}" r="${Math.min(w, h) / 2 - 1}" fill="${bgDark}" stroke="${bgMid}" stroke-width="1"/>`
              );
            }
            built.svg = svg;
          } else if (comp.type === 'slider') built.svg = generateSliderSVG(built.svgStyle, params);
          else if (comp.type === 'button') built.svg = generateButtonSVG(built.svgStyle, params);
        }
        if (comp.type === 'label') { built.color = theme.textColor; built.fontSize = 9; }
        if (comp.type === 'led') { built.color = theme.accentColor; }

        result.push(built);
        cx += w + compGap + 4;
      }
    }
  }

  // Attach metadata to the result array for upstream consumption
  result._tabMeta = {
    tabCount,
    tabLabels: tabMeta.tabs.map(t => t.label),
    recommendedHeight: optimalCanvasH,
  };

  return result;
}

// ── Two-pass layout helper ────────────────────────────────────────────────────
// Positions section components at origin (0,0) with unlimited height.
// Returns { comps, contentHeight } — the ACTUAL bounding box, not an estimate.
// This is the single source of truth for content height.

function layoutSectionContent(components, areaW, theme, brief, compact, sectionLayout, section = null) {
  if (!components || components.length === 0) return { comps: [], contentHeight: 0 };
  // Position at origin with unlimited height, no centering
  const comps = positionComponents(
    components, 0, 0, areaW, 99999, theme, brief, compact, sectionLayout, true, section
  );
  // Measure actual bounding box
  let contentHeight = 0;
  for (const c of comps) {
    contentHeight = Math.max(contentHeight, c.y + c.height);
  }
  // Add label overhang — knob/slider labels render below the component bounds.
  // Without this, the bottom row's labels overflow the panel.
  const hasLabeledItems = comps.some(c => c.type === 'knob' || c.type === 'slider' || c.type === 'meter');
  if (hasLabeledItems) {
    const labelH = compact === 'medium' ? 6 : compact ? 12 : 16;
    contentHeight += labelH;
  }
  return { comps, contentHeight };
}

// ── Finalize section: add panel, label, offset components ────────────────────

function finalizeSection(sectionComps, contentHeight, section, sectionX, sectionY, cellW, cellH, panelOverhead, secLabelH, isDense, isMedium, theme) {
  const result = [];

  // Panel
  result.push({
    type: 'panel', label: '',
    x: sectionX, y: sectionY, width: cellW, height: cellH,
    bgColor: theme.panelColor, borderColor: theme.panelBorder,
    borderRadius: isDense ? 4 : 6, zIndex: 0, opacity: 1,
  });

  // Section label — tight-fit, visible, bold (matches gold standard)
  const sectionLabelText = (section.label || '').toUpperCase();
  const sectionLabelW = Math.min(cellW - (isDense ? 8 : 16), sectionLabelText.length * 8 + 20);
  result.push({
    type: 'label', label: sectionLabelText,
    x: sectionX + (isDense ? 4 : 8), y: sectionY + (isDense ? 2 : 4),
    width: sectionLabelW, height: isDense ? secLabelH : secLabelH + 2,
    color: theme.textColor, fontSize: isDense ? 10 : isMedium ? 11 : 12, zIndex: 10, opacity: 1,
    fontWeight: 600, letterSpacing: '0.08em',
  });

  // Compute inner area and vertical centering
  const innerPad = isDense ? COMPACT_PADDING : isMedium ? MEDIUM_PADDING : PADDING;
  const innerX = sectionX + innerPad / 2;
  const innerY = sectionY + secLabelH + (isDense ? 8 : isMedium ? 2 : 12);
  const innerH = cellH - panelOverhead;
  const yCenter = 0; // Top-align — no floating content in oversized panels

  // Offset all components from origin to final position
  for (const c of sectionComps) {
    c.x += innerX;
    c.y += innerY + yCenter;
  }

  // Move LEDs next to section title
  const ledTargetX = sectionX + (isDense ? 4 : 8) + sectionLabelW + 4;
  const ledTargetY = sectionY + (isDense ? 2 : 4) + Math.floor((isDense ? secLabelH : secLabelH + 2) / 2) - 5;
  for (const c of sectionComps) {
    if (c.type === 'led') {
      c.x = ledTargetX;
      c.y = ledTargetY;
    }
  }

  // Move "On"/"Bypass"/"Enable" buttons to top-right of section header
  // Stack multiple buttons right-to-left so they don't overlap
  const onBtnPattern = /\b(on|enable|bypass|power)\b/i;
  const onButtons = sectionComps.filter(c => c.type === 'button' && onBtnPattern.test(c.label));
  const onBtnGap = isDense ? 2 : 4;
  let onBtnX = sectionX + cellW - (isDense ? 4 : 6);
  for (const c of onButtons) {
    c.height = Math.min(c.height, isDense ? secLabelH : secLabelH + 2);
    onBtnX -= c.width;
    c.x = onBtnX;
    c.y = sectionY + (isDense ? 2 : 4);
    c.zIndex = 10;
    onBtnX -= onBtnGap;
  }

  // ── Distribute content rows to fill panel height (no empty space at bottom) ──
  const innerBottom = sectionY + cellH - (isDense ? 6 : isMedium ? 6 : 8);
  const fillComps = sectionComps.filter(c =>
    c.type !== 'panel' && c.type !== 'led' &&
    !(c.type === 'label' && c.y < innerY + 4) &&
    !(c.type === 'button' && onBtnPattern.test(c.label))
  );
  if (fillComps.length >= 1) {
    const rowTolerance = 6;
    const sorted = [...fillComps].sort((a, b) => a.y - b.y);
    const rows = [];
    let rowStart = 0;
    for (let i = 1; i <= sorted.length; i++) {
      if (i === sorted.length || sorted[i].y - sorted[rowStart].y > rowTolerance) {
        rows.push(sorted.slice(rowStart, i));
        rowStart = i;
      }
    }

    // Check if section has display components (they'll be stretched later by layoutTabbed)
    const displayTypes = new Set(['waveform', 'wavetable_3d', 'oscilloscope', 'spectrum-analyzer', 'adsr']);
    const hasDisplayRow = rows.some(row => row.some(c => displayTypes.has(c.type)));

    const rowInfo = rows.map(row => {
      const top = Math.min(...row.map(c => c.y));
      const bottom = Math.max(...row.map(c => c.y + c.height));
      const isDisplay = row.some(c => displayTypes.has(c.type));
      return { top, height: bottom - top, comps: row, isDisplay };
    });

    if (rows.length === 1) {
      // Single row: center vertically in panel
      const contentTop = rowInfo[0].top;
      const availH = innerBottom - innerY;
      const centerY = innerY + Math.floor((availH - rowInfo[0].height) / 2);
      const shift = Math.round(centerY - contentTop);
      if (shift > 4) {
        for (const c of rowInfo[0].comps) c.y += shift;
      }
    } else if (rows.length >= 2 && hasDisplayRow) {
      // Display + controls: keep display at top, push non-display rows to bottom.
      // This leaves a gap below the display for stretch to fill.
      // IMPORTANT: Detect side-by-side (display-controls) layout and skip redistribution.
      // In display-controls, non-display components sit BESIDE the display (same Y region),
      // not below it. Redistribution would crush them into the below-display space.
      const contentTop = rowInfo[0].top;
      const nonDisplayRows = rowInfo.filter(r => !r.isDisplay);
      const displayRows = rowInfo.filter(r => r.isDisplay);
      if (nonDisplayRows.length > 0 && displayRows.length > 0) {
        // Display rows stay at top
        let displayBottomY = contentTop;
        for (const dr of displayRows) {
          displayBottomY = Math.max(displayBottomY, dr.top + dr.height);
        }
        // Check if this is a side-by-side layout: non-display content overlaps display Y range
        // (i.e., controls are beside the display, not below it)
        const firstNonDisplayTop = Math.min(...nonDisplayRows.map(r => r.top));
        const isSideBySide = firstNonDisplayTop < displayBottomY;

        // Also check if there's enough space — never use negative gaps
        const totalNonDisplayH = nonDisplayRows.reduce((s, r) => s + r.height, 0);
        const availBelowDisplay = innerBottom - displayBottomY;
        const minGap = isDense ? 4 : isMedium ? 6 : 8;
        const minNeeded = totalNonDisplayH + minGap * Math.max(0, nonDisplayRows.length - 1);

        if (isSideBySide || availBelowDisplay < minNeeded) {
          // Side-by-side (display-controls) layout: display is on the left, controls on right.
          // Spread the NON-display rows to fill the panel height, keeping display at top.
          const allContentBottom = Math.max(...rowInfo.map(r => r.top + r.height));
          const bottomGap = innerBottom - allContentBottom;
          if (bottomGap > 12) {
            if (nonDisplayRows.length >= 2) {
              // Spread non-display rows evenly within the available space
              const ndTotalH = nonDisplayRows.reduce((s, r) => s + r.height, 0);
              const ndStartY = nonDisplayRows[0].top;
              const ndAvailH = innerBottom - ndStartY;
              const ndGapSpace = ndAvailH - ndTotalH;
              const gapPer = ndGapSpace / (nonDisplayRows.length);  // distribute evenly
              let targetY = ndStartY + gapPer / 2; // small top offset for visual balance
              for (const ri of nonDisplayRows) {
                const shift = Math.round(targetY - ri.top);
                if (Math.abs(shift) > 1) {
                  for (const c of ri.comps) c.y += shift;
                }
                targetY += ri.height + gapPer;
              }
            } else if (nonDisplayRows.length === 1) {
              // Single non-display row: push toward bottom of panel
              const ndRow = nonDisplayRows[0];
              const targetY = innerBottom - ndRow.height - 4;
              const shift = Math.round(targetY - ndRow.top);
              if (shift > 4) {
                for (const c of ndRow.comps) c.y += shift;
              }
            }
            // Stretch display components vertically — capped at 200px to prevent oversized visuals
            for (const dr of displayRows) {
              for (const c of dr.comps) {
                if (displayTypes.has(c.type)) {
                  const maxStretch = Math.min(innerBottom - c.y - 6, 200);
                  if (maxStretch > c.height) c.height = maxStretch;
                }
              }
            }
          }
        } else {
          // Standard full-width display layout: push controls toward bottom
          const nonDisplayGap = nonDisplayRows.length > 1
            ? Math.min(8, Math.max(minGap, (availBelowDisplay - totalNonDisplayH) / (nonDisplayRows.length)))
            : 0;
          let targetY = innerBottom - totalNonDisplayH - nonDisplayGap * Math.max(0, nonDisplayRows.length - 1);
          // Don't push controls above display bottom
          targetY = Math.max(targetY, displayBottomY + 4);
          for (const ri of nonDisplayRows) {
            const shift = Math.round(targetY - ri.top);
            for (const c of ri.comps) c.y += shift;
            targetY += ri.height + nonDisplayGap;
          }
          // Stretch display down to fill gap between display bottom and first control row
          const firstControlY = innerBottom - totalNonDisplayH - nonDisplayGap * Math.max(0, nonDisplayRows.length - 1);
          const stretchTarget = Math.max(firstControlY, displayBottomY + 4) - 6;
          for (const dr of displayRows) {
            for (const c of dr.comps) {
              if (displayTypes.has(c.type) && stretchTarget > c.y + c.height) {
                c.height = stretchTarget - c.y;
              }
            }
          }
        }
      }
    } else if (rows.length >= 2) {
      // No displays: distribute rows evenly within panel, vertically centered
      const contentTop = rowInfo[0].top;
      const totalRowH = rowInfo.reduce((s, r) => s + r.height, 0);
      const availH = innerBottom - innerY;
      if (availH > totalRowH + 8) {
        const gapSpace = availH - totalRowH;
        const gapPer = gapSpace / (rows.length + 1); // +1 for top/bottom margins
        let targetY = innerY + gapPer;
        for (let i = 0; i < rowInfo.length; i++) {
          const shift = Math.round(targetY - rowInfo[i].top);
          for (const c of rowInfo[i].comps) c.y += shift;
          targetY += rowInfo[i].height + gapPer;
        }
      }
    }
  }

  // Expand sub-panels to fill available parent panel height
  const panelBottom = sectionY + cellH - (isDense ? 2 : isMedium ? 2 : 4);
  for (const c of sectionComps) {
    if (c.type === 'panel' && c.zIndex >= 1) {
      c.height = Math.max(c.height, panelBottom - c.y);
    }
  }

  // Clamp components within panel bounds
  const clampLeft = sectionX + (isDense ? 2 : isMedium ? 2 : 4);
  const clampRight = sectionX + cellW - (isDense ? 2 : isMedium ? 2 : 4);
  const clampBottom = sectionY + cellH - (isDense ? 2 : isMedium ? 2 : 4);
  const clampTop = sectionY + secLabelH + 4;

  // Filter to clampable components (skip decorative/repositioned elements)
  const clampable = sectionComps.filter(c => {
    if (c.type === 'led' || c.type === 'label' || (c.type === 'panel' && c.zIndex >= 1)) return false;
    if (c.type === 'button' && onBtnPattern.test(c.label)) return false;
    return true;
  });

  // Step 1: If content overflows bottom, compress GAPS between components (preserve sizes)
  const maxBottom = clampable.length > 0 ? Math.max(...clampable.map(c => c.y + c.height)) : 0;
  if (maxBottom > clampBottom && clampable.length > 0) {
    const contentTop = Math.min(...clampable.map(c => c.y));
    const overflow = maxBottom - clampBottom;
    // Sort by Y to process top-to-bottom
    const sorted = [...clampable].sort((a, b) => a.y - b.y);
    // Calculate total gap space between components
    let totalGap = 0;
    for (let i = 1; i < sorted.length; i++) {
      const gapBetween = sorted[i].y - (sorted[i-1].y + sorted[i-1].height);
      if (gapBetween > 0) totalGap += gapBetween;
    }
    if (totalGap > 0 && overflow <= totalGap) {
      // Can fix by only shrinking gaps — keep component sizes intact
      // Enforce minimum 2px gap between components
      const minGapPx = 2;
      const gapScale = Math.max(0, (totalGap - overflow) / totalGap);
      let shift = 0;
      for (let i = 1; i < sorted.length; i++) {
        const gapBetween = sorted[i].y - (sorted[i-1].y + sorted[i-1].height);
        if (gapBetween > 0) {
          const newGap = Math.max(minGapPx, Math.round(gapBetween * gapScale));
          shift += gapBetween - newGap;
        }
        sorted[i].y -= shift;
      }
    } else {
      // Severe overflow: proportional compression as fallback (scale ≥ 0.65)
      const contentSpan = maxBottom - contentTop;
      const availableSpan = clampBottom - contentTop;
      if (contentSpan > 0 && availableSpan > 0) {
        const scale = availableSpan / contentSpan;
        if (scale >= 0.65) {
          for (const c of clampable) {
            c.y = contentTop + Math.round((c.y - contentTop) * scale);
            c.height = Math.round(c.height * scale);
          }
        }
      }
    }
  }

  // Step 2: Hard clamp anything still outside bounds (last resort)
  for (const c of clampable) {
    if (c.x < clampLeft) c.x = clampLeft;
    if (c.x + c.width > clampRight) c.x = Math.max(clampLeft, clampRight - c.width);
    if (c.y + c.height > clampBottom) {
      if (c.y < clampBottom) {
        c.height = Math.max(8, clampBottom - c.y);
      } else {
        c.y = Math.max(clampTop, clampBottom - c.height);
      }
    }
    if (c.y < clampTop) c.y = clampTop;
  }

  // Step 3: Post-clamp overlap check within this section
  const interactiveTypes = new Set(['knob', 'slider', 'button', 'dropdown', 'meter']);
  const sectionInteractives = clampable.filter(c => interactiveTypes.has(c.type));
  sectionInteractives.sort((a, b) => a.y - b.y || a.x - b.x);
  for (let i = 0; i < sectionInteractives.length; i++) {
    for (let j = i + 1; j < sectionInteractives.length; j++) {
      const a = sectionInteractives[i], b = sectionInteractives[j];
      if (a.x < b.x + b.width && a.x + a.width > b.x &&
          a.y < b.y + b.height && a.y + a.height > b.y) {
        b.y = a.y + a.height + 1;
      }
    }
  }

  result.push(...sectionComps);

  return result;
}

// ── Horizontal strip layout (sections side by side) ──────────────────────────

function layoutHorizontal(sections, canvasW, canvasH, theme, brief) {
  const { isDense, isMedium } = assessDensity(sections, canvasW, canvasH);
  const { pad, secGap, secLabelH } = getSectionGap(isDense, isMedium);
  const panelOverhead = secLabelH + (isDense ? 14 : isMedium ? 4 : 20);
  const innerPad = pad;
  const MIN_PANEL_H = 50;

  const usableW = canvasW - pad * 2;
  const availableH = canvasH - TITLE_BAR_HEIGHT - pad * 2;

  // Content-fit column widths based on component counts
  const idealWidths = sections.map(sec => {
    const compact = isSectionCompact(sec, isDense, isMedium);
    const sectionSizes = getSectionSizes(sec, isDense, isMedium);
    const sectionGap = compact === 'medium' ? MEDIUM_COMPONENT_GAP : compact ? COMPACT_COMPONENT_GAP : COMPONENT_GAP;
    return computeIdealSectionWidth(sec, sectionSizes, sectionGap, innerPad, sec.layout || null);
  });
  // Fill available width — ideal width as floor + extra distributed by component count.
  // Dense panels (more components) absorb more extra space; sparse panels stay tight.
  const totalIdealContent = idealWidths.reduce((s, w) => s + w, 0);
  const availableForContent = usableW - (sections.length - 1) * secGap;
  let colWidths;
  if (availableForContent <= totalIdealContent) {
    const scale = availableForContent / totalIdealContent;
    colWidths = idealWidths.map(w => Math.floor(w * scale));
  } else {
    const extraSpace = availableForContent - totalIdealContent;
    const counts = sections.map(sec => (sec.components || []).length);
    const totalCounts = counts.reduce((s, c) => s + c, 0);
    colWidths = idealWidths.map((w, i) => {
      const share = totalCounts > 0 ? counts[i] / totalCounts : 1 / sections.length;
      return Math.floor(w + extraSpace * share);
    });
  }

  // ── Pass 1: Position components at origin, measure actual heights ──
  const sectionData = [];
  let panelH = MIN_PANEL_H;
  for (let i = 0; i < sections.length; i++) {
    const section = sections[i];
    const compact = isSectionCompact(section, isDense, isMedium);
    const innerW = colWidths[i] - innerPad;
    const { comps, contentHeight } = layoutSectionContent(
      section.components || [], innerW, theme, brief, compact, section.layout || null, section
    );
    sectionData.push({ comps, contentHeight, section, compact });
    panelH = Math.max(panelH, contentHeight + panelOverhead);
  }
  // No height cap — canvas auto-fits to content in computeLayout

  // ── Pass 2: Build final layout with correctly sized panels ──
  const result = [];
  const totalRowW = colWidths.reduce((s, w) => s + w, 0) + (sections.length - 1) * secGap;
  let xOffset = totalRowW < usableW
    ? pad + Math.floor((usableW - totalRowW) / 2)
    : pad;
  for (let i = 0; i < sections.length; i++) {
    const { comps: sComps, contentHeight, section } = sectionData[i];
    const sectionW = colWidths[i];
    const sectionX = xOffset;
    const sectionY = TITLE_BAR_HEIGHT + pad;

    result.push(...finalizeSection(
      sComps, contentHeight, section,
      sectionX, sectionY, sectionW, panelH,
      panelOverhead, secLabelH, isDense, isMedium, theme
    ));
    xOffset += sectionW + secGap;
  }

  return result;
}

// ── Vertical strip layout (sections stacked) ─────────────────────────────────

function layoutVertical(sections, canvasW, canvasH, theme, brief) {
  const { isDense, isMedium } = assessDensity(sections, canvasW, canvasH);
  const { pad, secGap, secLabelH } = getSectionGap(isDense, isMedium);
  const panelOverhead = secLabelH + (isDense ? 14 : isMedium ? 4 : 20);

  const totalWeight = sections.reduce((sum, s) => sum + (s.weight || 1), 0);
  const usableW = canvasW - pad * 2;
  const usableH = canvasH - TITLE_BAR_HEIGHT - pad * 2 - secGap * (sections.length - 1);

  // ── Pass 1: Position components at origin, measure actual heights ──
  const sectionData = [];
  for (const section of sections) {
    const compact = isSectionCompact(section, isDense, isMedium);
    const innerW = usableW - pad;
    const { comps, contentHeight } = layoutSectionContent(
      section.components || [], innerW, theme, brief, compact, section.layout || null, section
    );
    const weight = section.weight || 1;
    const weightH = Math.floor(usableH * weight / totalWeight);
    // Use content height + modest padding, capped by weight allocation
    // This prevents sections from being bloated by weight when content doesn't need it
    const contentNeeded = contentHeight + panelOverhead;
    const sectionH = Math.max(contentNeeded, Math.min(weightH, contentNeeded + 30));
    sectionData.push({ comps, contentHeight, section, compact, sectionH });
  }

  // ── Pass 2: Build final layout ──
  const result = [];
  let yOffset = TITLE_BAR_HEIGHT + pad;
  for (const { comps: sComps, contentHeight, section, sectionH } of sectionData) {
    const sectionX = pad;
    const sectionY = yOffset;

    result.push(...finalizeSection(
      sComps, contentHeight, section,
      sectionX, sectionY, usableW, sectionH,
      panelOverhead, secLabelH, isDense, isMedium, theme
    ));
    yOffset += sectionH + secGap;
  }

  return result;
}

// ── Grid layout (content-fit panels, proportional widths, centered) ───────────
// Supports "position": "right" on a section to span full height as a sidebar

function layoutGrid(sections, canvasW, canvasH, theme, brief) {
  const { isDense, isMedium } = assessDensity(sections, canvasW, canvasH);
  const { pad, secGap, secLabelH } = getSectionGap(isDense, isMedium);
  const panelOverhead = secLabelH + (isDense ? 10 : isMedium ? 4 : 16);
  const innerPadding = pad;
  const MIN_PANEL_H = 50;

  // ── Detect sidebar sections ──
  const sidebarRight = sections.filter(s => s.position === 'right');
  const gridSections = sections.filter(s => s.position !== 'right');

  // If we have a sidebar, compute its content-fit width
  let sidebarW = 0;
  if (sidebarRight.length > 0) {
    // Use content-fit for sidebar too — don't waste space
    const sidebarIdealWidths = sidebarRight.map(sec => {
      const compact = isSectionCompact(sec, isDense, isMedium);
      const sectionSizes = getSectionSizes(sec, isDense, isMedium);
      const sectionGap = compact === 'medium' ? MEDIUM_COMPONENT_GAP : compact ? COMPACT_COMPONENT_GAP : COMPONENT_GAP;
      return computeIdealSectionWidth(sec, sectionSizes, sectionGap, innerPadding, sec.layout || null);
    });
    sidebarW = Math.max(...sidebarIdealWidths);
    // Cap sidebar to 40% of canvas
    sidebarW = Math.min(sidebarW, Math.floor(canvasW * 0.4));
  }
  const gridCanvasW = sidebarRight.length > 0
    ? canvasW - sidebarW - secGap
    : canvasW;

  const cols = gridSections.length >= 12 ? 4 : gridSections.length >= 8 ? 3 : gridSections.length >= 4 ? 2 : Math.min(gridSections.length, 2);
  const numRows = Math.ceil(gridSections.length / cols);

  // ── Step 1: Compute per-row CONTENT-FIT column widths ──
  // Instead of weight-based (which expands to fill canvas), estimate the ideal
  // width each section needs based on its components, producing tighter panels.
  const rowMeta = [];
  const rowUsableW = gridCanvasW - pad * 2;
  for (let r = 0; r < numRows; r++) {
    const rowSections = [];
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      if (idx < gridSections.length) rowSections.push(gridSections[idx]);
    }
    const rowColCount = rowSections.length;

    // Compute ideal (content-fit) width for each section
    const idealWidths = rowSections.map(sec => {
      const compact = isSectionCompact(sec, isDense, isMedium);
      const sectionSizes = getSectionSizes(sec, isDense, isMedium);
      const sectionGap = compact === 'medium' ? MEDIUM_COMPONENT_GAP : compact ? COMPACT_COMPONENT_GAP : COMPONENT_GAP;
      return computeIdealSectionWidth(sec, sectionSizes, sectionGap, innerPadding, sec.layout || null);
    });

    // Fill available width — ideal width as floor + extra distributed by component count.
    // Dense panels (more components) absorb more extra space; sparse panels stay tight.
    const totalIdealContent = idealWidths.reduce((s, w) => s + w, 0);
    const availableForContent = rowUsableW - (rowColCount - 1) * secGap;
    let colWidths;
    if (availableForContent <= totalIdealContent) {
      const scale = availableForContent / totalIdealContent;
      colWidths = idealWidths.map(w => Math.floor(w * scale));
    } else {
      const extraSpace = availableForContent - totalIdealContent;
      const counts = rowSections.map(sec => (sec.components || []).length);
      const totalCounts = counts.reduce((s, c) => s + c, 0);
      colWidths = idealWidths.map((w, i) => {
        const share = totalCounts > 0 ? counts[i] / totalCounts : 1 / rowColCount;
        return Math.floor(w + extraSpace * share);
      });
    }
    rowMeta.push({ rowSections, colWidths, rowColCount });
  }

  // Reduce title bar for medium/compact density (gold standard starts panels at y=6)
  const effectiveTitleH = isDense ? 4 : isMedium ? 0 : TITLE_BAR_HEIGHT;

  // ── Step 2: Position components at origin, measure ACTUAL content height ──
  const sectionContents = [];
  for (let r = 0; r < numRows; r++) {
    const { rowSections, colWidths } = rowMeta[r];
    const rowContents = [];
    for (let ci = 0; ci < rowSections.length; ci++) {
      const section = rowSections[ci];
      const innerW = colWidths[ci] - innerPadding;
      const compact = isSectionCompact(section, isDense, isMedium);
      const { comps, contentHeight } = layoutSectionContent(
        section.components || [], innerW, theme, brief, compact, section.layout || null, section
      );
      rowContents.push({ comps, contentHeight, section, compact });
    }
    sectionContents.push(rowContents);
  }

  // ── Step 3: Compute row heights from ACTUAL content (not estimates) ──
  const rowHeights = sectionContents.map(rowContents => {
    let maxH = MIN_PANEL_H;
    for (const { contentHeight } of rowContents) {
      maxH = Math.max(maxH, contentHeight + panelOverhead);
    }
    return maxH;
  });

  // No scale-down — panels are sized to fit their actual content.
  // Canvas height auto-fits in computeLayout.

  // Compute Y offsets
  const rowYOffsets = [0];
  for (let r = 0; r < numRows - 1; r++) {
    rowYOffsets.push(rowYOffsets[r] + rowHeights[r] + secGap);
  }

  // Total grid height for sidebar
  const totalGridH = rowHeights.reduce((sum, h) => sum + h, 0) + Math.max(0, numRows - 1) * secGap;

  // ── Step 5: Build final layout — panels sized to fit actual content ──
  const comps = [];

  for (let r = 0; r < numRows; r++) {
    const { colWidths, rowColCount } = rowMeta[r];
    const rowContents = sectionContents[r];
    const totalRowW = colWidths.reduce((s, w) => s + w, 0) + (rowColCount - 1) * secGap;
    // Always center rows (content-fit widths may be narrower than canvas)
    const rowStartX = totalRowW < rowUsableW
      ? pad + Math.floor((rowUsableW - totalRowW) / 2)
      : pad;

    let xOffset = rowStartX;
    for (let ci = 0; ci < rowContents.length; ci++) {
      const { comps: sectionComps, contentHeight, section } = rowContents[ci];
      const cellW = colWidths[ci];
      const cellH = rowHeights[r];
      const sectionX = xOffset;
      const sectionY = effectiveTitleH + pad + rowYOffsets[r];

      comps.push(...finalizeSection(
        sectionComps, contentHeight, section,
        sectionX, sectionY, cellW, cellH,
        panelOverhead, secLabelH, isDense, isMedium, theme
      ));
      xOffset += cellW + secGap;
    }
  }

  // ── Sidebar sections: span full height on the right ──
  if (sidebarRight.length > 0) {
    const sidebarX = canvasW - pad - sidebarW;
    const sidebarStartY = effectiveTitleH + pad;
    const sidebarFullH = Math.max(totalGridH, MIN_PANEL_H);

    // If multiple sidebar sections, stack them vertically in the sidebar column
    const sidebarWeightTotal = sidebarRight.reduce((s, sec) => s + (sec.weight || 1), 0);
    let sidebarYOffset = 0;

    for (const section of sidebarRight) {
      const sectionWeight = section.weight || 1;
      const sectionH = sidebarRight.length === 1
        ? sidebarFullH
        : Math.floor(sidebarFullH * sectionWeight / sidebarWeightTotal);

      const compact = isSectionCompact(section, isDense, isMedium);
      const innerW = sidebarW - innerPadding;
      const { comps: sectionComps, contentHeight } = layoutSectionContent(
        section.components || [], innerW, theme, brief, compact, section.layout || null, section
      );

      const cellH = Math.max(sectionH, contentHeight + panelOverhead);

      comps.push(...finalizeSection(
        sectionComps, contentHeight, section,
        sidebarX, sidebarStartY + sidebarYOffset, sidebarW, cellH,
        panelOverhead, secLabelH, isDense, isMedium, theme
      ));

      sidebarYOffset += cellH + secGap;
    }
  }

  return comps;
}

// ── Position components within a section area ────────────────────────────────
// Groups by type → balanced centered rows → symmetric grid layout
// Supports "display-controls" layout: waveform on left, controls on right

function positionComponents(components, startX, startY, areaW, areaH, theme, brief, compact = false, sectionLayout = null, noCentering = false, section = null) {
  if (components.length === 0) return [];

  const actualBgColor = (brief && brief.bgColor) || theme.bgColor;
  const sizes = compact === 'medium' ? MEDIUM_TYPE_SIZES : compact ? COMPACT_TYPE_SIZES : TYPE_SIZES;
  const gap = compact === 'medium' ? MEDIUM_COMPONENT_GAP : compact ? COMPACT_COMPONENT_GAP : COMPONENT_GAP;
  const labelExtra = compact === 'medium' ? 4 : compact ? 10 : 14;

  // ── Group components by visual role (order within groups preserved) ──
  const displays = [], selectors = [], knobs = [], verticals = [], toggles = [], others = [];
  for (const comp of components) {
    // VU needle meters are large display elements, not thin vertical bars
    if (comp.type === 'meter' && comp.svgStyle === 'vu-needle') displays.push(comp);
    else if (['waveform', 'xy-pad', 'image', 'wavetable_3d', 'oscilloscope', 'spectrum-analyzer', 'adsr'].includes(comp.type)) displays.push(comp);
    else if (comp.type === 'dropdown') selectors.push(comp);
    else if (comp.type === 'knob') knobs.push(comp);
    else if (['slider', 'meter'].includes(comp.type)) verticals.push(comp);
    else if (['button', 'led'].includes(comp.type)) toggles.push(comp);
    else others.push(comp);
  }

  // ── Auto-detect display-controls layout ──
  // Side-by-side: display left, controls right — but NEVER for large displays (wavetable_3d, xy-pad)
  // Large displays should span full width on top (like Serum's oscillator panels)
  const hasLargeDisplay = displays.some(d => d.type === 'wavetable_3d' || d.type === 'xy-pad');
  const useDisplayControls = displays.length > 0 && !hasLargeDisplay && (
    sectionLayout === 'display-controls' ||
    (displays.length === 1 && knobs.length >= 3 && areaW >= (compact ? 200 : 280)));

  const result = [];
  let cy = 0;

  // ── Build a positioned component object ──
  const buildAt = (comp, x, y) => {
    const size = sizes[comp.type] || sizes.knob;
    // Support "size" property for visual hierarchy: "large" (1.4x), "small" (0.7x)
    const sizeScale = comp.size === 'large' ? 1.4 : comp.size === 'small' ? 0.7 : 1;
    const w = comp.width || Math.round(size.w * sizeScale);
    const h = comp.height || Math.round(size.h * sizeScale);

    const built = {
      type: comp.type, label: comp.label || '',
      x, y, width: w, height: h,
      color: theme.accentColor,
      svgStyle: comp.svgStyle || '', bodyColor: comp.bodyColor || '',
      indicatorColor: comp.indicatorColor || '', accentColor: comp.accentColor || '',
      fluxPrompt: comp.fluxPrompt || '',
      opacity: 1, rotation: 0, borderRadius: comp.borderRadius ?? 0,
      fontSize: comp.fontSize ?? 13, zIndex: 2,
    };
    if (comp.fluxPrompt) built.fluxPrompt = comp.fluxPrompt;

    // Waveform visual style + unique seed for procedural variation
    if (comp.type === 'waveform') {
      const WAVEFORM_STYLE_POOL = ['3d-wavetable', 'neon-glow', 'gradient-fill', 'retro-crt', 'glass-panel', 'holographic', 'led-matrix', 'minimal-line'];
      built.waveformStyle = comp.waveformStyle || WAVEFORM_STYLE_POOL[Math.floor(Math.random() * WAVEFORM_STYLE_POOL.length)];
      // 8 random floats [0,1] — drives unique visual variations per instance
      built.waveformSeed = comp.waveformSeed || Array.from({ length: 8 }, () => Math.random());
    }

    if (['knob', 'slider', 'button'].includes(comp.type)) {
      built.color = theme.accentColor;
      if (!built.svgStyle) {
        if (comp.type === 'knob') built.svgStyle = theme.knobSvgStyle;
        else if (comp.type === 'slider') built.svgStyle = theme.sliderSvgStyle;
        else if (comp.type === 'button') built.svgStyle = theme.buttonSvgStyle;
      }
      if (!built.bodyColor) built.bodyColor = adjustColorForBody(actualBgColor);
      if (!built.indicatorColor) built.indicatorColor = theme.accentColor;
      if (!built.accentColor) built.accentColor = adjustColorForAccent(theme.accentColor);
    }
    if (comp.type === 'label') { built.color = theme.textColor; built.fontSize = comp.fontSize || 11; }
    if (comp.type === 'led') { built.color = theme.accentColor; }

    if (built.svgStyle && ['knob', 'slider', 'button'].includes(built.type)) {
      const uid = Math.random().toString(36).slice(2, 10);
      const params = {
        width: w, height: h,
        bodyColor: built.bodyColor, indicatorColor: built.indicatorColor,
        accentColor: built.accentColor, uid, label: built.label,
      };
      let rawSvg = '';
      if (built.type === 'knob') rawSvg = generateKnobSVG(built.svgStyle, params);
      else if (built.type === 'slider') rawSvg = generateSliderSVG(built.svgStyle, params);
      else if (built.type === 'button') rawSvg = generateButtonSVG(built.svgStyle, params);

      if (rawSvg && built.type === 'knob') {
        const bgDark = darkenHex(actualBgColor, 0.4);
        const bgMid = darkenHex(actualBgColor, 0.25);
        rawSvg = rawSvg.replace(
          /(<svg[^>]*>)/,
          `$1\n  <circle cx="${w / 2}" cy="${h / 2}" r="${Math.min(w, h) / 2 - 1}" fill="${bgDark}" stroke="${bgMid}" stroke-width="1"/>`
        );
      }
      built.svg = rawSvg;
    }
    return { built, w, h };
  };

  // ── Layout uniform items in balanced, centered rows ──
  // Separates large/normal/small items into tiers to prevent overlap
  const layoutUniformGrid = (items, withLabels) => {
    if (items.length === 0) return;
    const s = sizes[items[0].type] || sizes.knob;
    const baseW = s.w;
    const baseH = s.h;

    // Separate by size tier — large items get their own row(s) for clean hierarchy
    const largeItems = items.filter(item => item.size === 'large');
    const normalItems = items.filter(item => !item.size || (item.size !== 'large' && item.size !== 'small'));
    const smallItems = items.filter(item => item.size === 'small');

    const layoutTier = (tierItems, slotW, slotH) => {
      if (tierItems.length === 0) return;
      const maxFit = Math.max(1, Math.floor((areaW + gap) / (slotW + gap)));
      const perRow = Math.min(maxFit, tierItems.length);
      const numRows = Math.ceil(tierItems.length / perRow);
      const fullRowW = perRow * slotW + (perRow - 1) * gap;
      const offX = Math.max(0, Math.floor((areaW - fullRowW) / 2));
      const quantizedOffX = Math.round(offX / gap) * gap;

      let idx = 0;
      for (let r = 0; r < numRows; r++) {
        const remaining = tierItems.length - idx;
        const rowsLeft = numRows - r;
        const count = Math.ceil(remaining / rowsLeft);
        for (let i = 0; i < count && idx < tierItems.length; i++, idx++) {
          const rawX = quantizedOffX + i * (slotW + gap);
          const quantX = Math.round(rawX / gap) * gap;
          const { built } = buildAt(tierItems[idx], startX + quantX, startY + cy);
          result.push(built);
        }
        cy += slotH + (withLabels ? labelExtra : 0) + gap;
      }
    };

    // When ≤2 large knobs, merge into normal row (avoids lonely hero knob on its own row).
    // layoutMixedRow handles per-item sizes so the large knob renders bigger inline.
    if (largeItems.length > 0 && largeItems.length <= 2 && normalItems.length > 0) {
      layoutMixedRow([...largeItems, ...normalItems], withLabels);
    } else {
      // Large knobs: 1.4x slots (only when 3+ large knobs warrant their own tier)
      layoutTier(largeItems, Math.round(baseW * 1.4), Math.round(baseH * 1.4));
      if (largeItems.length > 0 && (normalItems.length > 0 || smallItems.length > 0)) cy += gap;
      // Normal knobs: base-size slots
      layoutTier(normalItems, baseW, baseH);
    }
    if ((largeItems.length > 0 || normalItems.length > 0) && smallItems.length > 0) cy += gap;
    // Small knobs: 0.7x slots
    layoutTier(smallItems, Math.round(baseW * 0.7), Math.round(baseH * 0.7));
  };


    const layoutMixedRow = (items, withLabels) => {
    if (items.length === 0) return;
    const dims = items.map(c => {
      const s = sizes[c.type] || sizes.knob;
      const scale = c.size === "large" ? 1.4 : c.size === "small" ? 0.7 : 1;
      return { 
        w: c.width || Math.round(s.w * scale), 
        h: c.height || Math.round(s.h * scale) 
      };
    });

    // Split into rows by flowing with wrapping
    const rowGroups = [];
    let row = [], rw = 0;
    for (let i = 0; i < items.length; i++) {
      const itemW = dims[i].w + gap;
      if (rw + dims[i].w > areaW && row.length > 0) {
        rowGroups.push({ idxs: row, totalW: rw - gap });
        row = []; rw = 0;
      }
      row.push(i);
      rw += itemW;
    }
    if (row.length) {
      rowGroups.push({ idxs: row, totalW: rw - gap });
    }

    // FIXED: Calculate consistent offset based on WIDEST row
    const maxRowW = Math.max(...rowGroups.map(r => r.totalW));
    const consistentOffX = Math.max(0, Math.floor((areaW - maxRowW) / 2));
    const quantizedOffX = Math.round(consistentOffX / gap) * gap;

    for (const { idxs } of rowGroups) {
      const maxH = Math.max(...idxs.map(i => dims[i].h));
      
      // FIXED: Left-align all rows to same origin
      let cx = quantizedOffX;
      for (const i of idxs) {
        const vOff = Math.floor((maxH - dims[i].h) / 2);
        // Quantize positions
        const quantX = Math.round(cx / gap) * gap;
        const quantY = Math.round((cy + vOff) / gap) * gap;
        const { built } = buildAt(items[i], startX + quantX, startY + quantY);
        result.push(built);
        cx += dims[i].w + gap;
      }
      cy += maxH + (withLabels ? labelExtra : 0) + gap;
    }
  };

  // ── Display-Controls layout: waveform on left, controls on right ──
  if (useDisplayControls) {
    const display = displays[0];
    const displaySize = sizes[display.type] || sizes.waveform;
    const displayW = display.width || displaySize.w;

    // Left column: adaptive width based on section width.
    // For narrow sections (<350px), shrink display to allow 2-column knobs on the right.
    // For wider sections, display at natural width or 45% minimum.
    const knobW = (sizes.knob?.w || 48) + gap; // knob + gap = ~56px
    const minRightFor2Col = knobW * 2 + 4; // ~116px for 2 knobs side by side
    let leftW;
    if (areaW < 350 && displayW > areaW - minRightFor2Col - gap) {
      // Narrow section: shrink display so right column fits 2 knobs
      leftW = Math.max(120, areaW - minRightFor2Col - gap);
    } else {
      leftW = Math.max(displayW, Math.floor(areaW * 0.45));
    }
    const rightX = startX + leftW + gap;
    const rightW = areaW - leftW - gap;

    // ── LEFT COLUMN: displays stacked vertically ──
    let leftCy = 0;
    const leftDisplayComps = [];
    for (let di = 0; di < displays.length; di++) {
      const d = displays[di];
      const dSize = sizes[d.type] || sizes.waveform;
      const dH = d.height || dSize.h;
      const { built: dBuilt } = buildAt(d, startX, startY + leftCy);
      dBuilt.width = leftW;
      // Cap display height — proportional to available area
      // IMPORTANT: check 'medium' first since 'medium' is truthy and would match `compact ?`
      const displayMaxH = compact === 'medium' ? 150 : compact ? 120 : 200;
      dBuilt.height = Math.min(dH, Math.floor(areaH * 0.50), displayMaxH);
      result.push(dBuilt);
      leftDisplayComps.push(dBuilt);
      leftCy += dBuilt.height + gap;
    }

    // ── RIGHT COLUMN: hero + selectors + knobs ──
    const heroKnobs = knobs.filter(k => k.size === 'large');
    const normalKnobs = knobs.filter(k => k.size !== 'large');
    const normalKnobSize = sizes.knob;

    // Detect explicit groups from section definition
    const sectionGroups = [];
    if (section?.groups) {
      for (const g of section.groups) {
        const items = normalKnobs.filter(k => (g.items || []).includes(k.label));
        if (items.length > 0) sectionGroups.push({ name: g.name, items });
      }
    }
    const groupedKnobSet = new Set(sectionGroups.flatMap(g => g.items));
    const ungroupedNormal = normalKnobs.filter(k => !groupedKnobSet.has(k));

    let rightCy = 0;

    // Hero knob(s) + selectors in a horizontal band at top of right column
    const heroW = heroKnobs.length > 0
      ? Math.round((heroKnobs[0].width || sizes.knob.w) * 1.4) + gap : 0;
    const selW = selectors.length > 0
      ? (selectors[0].width || (sizes[selectors[0].type] || sizes.dropdown).w) + gap : 0;

    let heroMaxH = 0;
    if (heroKnobs.length > 0) {
      let hx = 0;
      for (const hk of heroKnobs) {
        const { built, w, h } = buildAt(hk, rightX + hx, startY);
        result.push(built);
        heroMaxH = Math.max(heroMaxH, h + labelExtra);
        hx += w + gap;
      }
    }

    let selMaxH = 0;
    if (selectors.length > 0) {
      let sy = 0;
      for (const sel of selectors) {
        const selSize = sizes[sel.type] || sizes.dropdown;
        const sw = sel.width || selSize.w;
        const sh = sel.height || selSize.h;
        const { built } = buildAt(sel, rightX + heroW, startY + sy);
        built.width = Math.min(sw, rightW - heroW);
        result.push(built);
        sy += sh + gap;
      }
      selMaxH = sy;
    }

    rightCy = Math.max(heroMaxH > 0 ? heroMaxH + gap : 0, selMaxH);

    // Helper to layout a batch of knobs in the right column grid.
    // Returns unplaced knobs when maxRows is exceeded (for full-width overflow below display).
    const rightMaxPerRow = Math.max(1, Math.floor((rightW + gap) / (normalKnobSize.w + gap)));
    const knobRowH = normalKnobSize.h + labelExtra + gap;
    const layoutRightKnobBatch = (batchKnobs, maxRows = Infinity) => {
      if (batchKnobs.length === 0) return [];
      const perRow = Math.min(rightMaxPerRow, batchKnobs.length || 1);
      const numRows = Math.min(Math.ceil(batchKnobs.length / perRow), maxRows);
      let kIdx = 0;
      for (let rr = 0; rr < numRows; rr++) {
        const remaining = batchKnobs.length - kIdx;
        const rowsLeft = numRows - rr;
        const count = Math.ceil(remaining / rowsLeft);
        for (let ki = 0; ki < count && kIdx < batchKnobs.length; ki++, kIdx++) {
          const { built } = buildAt(batchKnobs[kIdx], rightX + ki * (normalKnobSize.w + gap), startY + rightCy);
          result.push(built);
        }
        rightCy += knobRowH;
      }
      return batchKnobs.slice(kIdx); // Unplaced knobs for full-width overflow
    };
    // Max right-column rows that fit beside the display (not below it)
    const getRightRowsBesideDisplay = () => Math.max(1, Math.floor(Math.max(knobRowH, leftCy - rightCy) / knobRowH));

    if (sectionGroups.length >= 2) {
      // ── TOP-DISPLAY MODE: waveform+hero+utility top, grouped columns below ──
      let bottomGroups = sectionGroups;

      // Below both columns: remaining grouped sub-sections across full width
      cy = Math.max(leftCy, rightCy);

      if (bottomGroups.length > 0) {
        const numGroups = bottomGroups.length;
        const colGap = gap;
        const colW = Math.floor((areaW - (numGroups - 1) * colGap) / numGroups);
        const subLabelH = compact === 'medium' ? 10 : compact ? 9 : 12;

        const groupStartY = cy;
        let maxGroupH = 0;

        for (let gi = 0; gi < numGroups; gi++) {
          const group = bottomGroups[gi];
          const colX = startX + gi * (colW + colGap);
          let gcy = 0;

          // Sub-label — tight-fit, visible, bold (matches gold standard)
          const subLabelText = group.name.toUpperCase();
          const subLabelW = Math.min(colW, subLabelText.length * 7 + 10);
          result.push({
            type: 'label', label: subLabelText,
            x: colX, y: startY + groupStartY,
            width: subLabelW, height: subLabelH,
            color: theme.textColor, fontSize: compact === 'medium' ? 9 : 9,
            fontWeight: 500, zIndex: 3, opacity: 1, letterSpacing: '0.06em',
          });
          gcy += subLabelH + (compact === 'medium' ? 2 : 3);

          // Knobs in grid within column
          const maxPerRow = Math.max(1, Math.floor((colW + gap) / (normalKnobSize.w + gap)));
          const gKnobs = group.items;
          const perRow = Math.min(maxPerRow, gKnobs.length);
          const numKnobRows = Math.ceil(gKnobs.length / perRow);
          let kIdx = 0;

          for (let kr = 0; kr < numKnobRows; kr++) {
            const remaining = gKnobs.length - kIdx;
            const rowsLeft = numKnobRows - kr;
            const count = Math.ceil(remaining / rowsLeft);
            for (let ki = 0; ki < count && kIdx < gKnobs.length; ki++, kIdx++) {
              const { built } = buildAt(gKnobs[kIdx],
                colX + ki * (normalKnobSize.w + gap),
                startY + groupStartY + gcy);
              result.push(built);
            }
            gcy += normalKnobSize.h + labelExtra + gap;
          }

          maxGroupH = Math.max(maxGroupH, gcy);
        }

        cy = groupStartY + maxGroupH;
      }

    } else {
      // ── Standard display-controls: all normal knobs in right column ──
      // Detect prefix groups for sub-labels
      const knobPrefixMap = {};
      for (const k of normalKnobs) {
        const prefix = (k.label || '').split(/\s+/)[0];
        if (prefix && prefix.length >= 2) {
          if (!knobPrefixMap[prefix]) knobPrefixMap[prefix] = [];
          knobPrefixMap[prefix].push(k);
        }
      }
      const rightKnobGroups = Object.entries(knobPrefixMap)
        .filter(([, items]) => items.length >= 2)
        .map(([prefix, items]) => ({ prefix, items }));

      // Collect overflow knobs that don't fit beside the display
      const overflowKnobs = [];

      if (rightKnobGroups.length >= 1) {
        const EXPAND = {
          Rev: 'Reverb', Dly: 'Delay', Ch: 'Chorus', Fl: 'Flanger',
          Ph: 'Phaser', Dist: 'Distortion', Cmp: 'Compressor', Sat: 'Saturator',
          LFO: 'LFO', Env: 'Envelope', Mod: 'Modulation',
        };
        const subLabelH = compact === 'medium' ? 10 : compact ? 9 : 10;
        const groupedKnobSet = new Set(rightKnobGroups.flatMap(g => g.items));
        const ungroupedKnobs = normalKnobs.filter(k => !groupedKnobSet.has(k));
        if (ungroupedKnobs.length > 0) {
          overflowKnobs.push(...layoutRightKnobBatch(ungroupedKnobs, getRightRowsBesideDisplay()));
        }
        for (const group of rightKnobGroups) {
          const fullName = EXPAND[group.prefix] || group.prefix;
          const prefixText = fullName.toUpperCase();
          const prefixTextW = Math.min(rightW, prefixText.length * 7 + 10);
          result.push({
            type: 'label', label: prefixText,
            x: rightX, y: startY + rightCy,
            width: prefixTextW, height: subLabelH,
            color: theme.textColor, fontSize: compact === 'medium' ? 9 : 9,
            fontWeight: 500, zIndex: 10, opacity: 1, letterSpacing: '0.06em',
          });
          rightCy += subLabelH + gap;
          overflowKnobs.push(...layoutRightKnobBatch(group.items, getRightRowsBesideDisplay()));
        }
      } else {
        overflowKnobs.push(...layoutRightKnobBatch(normalKnobs, getRightRowsBesideDisplay()));
      }

      // Place utility buttons (non-On/Bypass) in right column alongside knobs
      // instead of a lonely full-width row at the bottom
      const onBtnPat = /\b(on|enable|bypass|power)\b/i;
      const utilButtons = toggles.filter(t => t.type === 'button' && !onBtnPat.test(t.label));
      if (utilButtons.length > 0) {
        // Add a small gap then place buttons in right column
        rightCy += gap;
        const btnDims = utilButtons.map(b => {
          const bs = sizes[b.type] || sizes.button;
          return { w: b.width || bs.w, h: b.height || bs.h };
        });
        const totalBtnW = btnDims.reduce((s, d) => s + d.w, 0) + (utilButtons.length - 1) * gap;
        const btnOffX = Math.max(0, Math.floor((rightW - totalBtnW) / 2));
        let bx = btnOffX;
        let maxBtnH = 0;
        for (let bi = 0; bi < utilButtons.length; bi++) {
          const { built } = buildAt(utilButtons[bi], rightX + bx, startY + rightCy);
          result.push(built);
          bx += btnDims[bi].w + gap;
          maxBtnH = Math.max(maxBtnH, btnDims[bi].h);
        }
        rightCy += maxBtnH + gap;
      }

      cy = Math.max(leftCy, rightCy);

      // ── Place overflow knobs in full-width rows below display ──
      // Knobs that didn't fit beside the display use the full section width
      if (overflowKnobs.length > 0) {
        const fullPerRow = Math.max(1, Math.floor((areaW + gap) / (normalKnobSize.w + gap)));
        const oRows = Math.ceil(overflowKnobs.length / fullPerRow);
        let oIdx = 0;
        for (let rr = 0; rr < oRows; rr++) {
          const remaining = overflowKnobs.length - oIdx;
          const count = Math.ceil(remaining / (oRows - rr));
          for (let ki = 0; ki < count && oIdx < overflowKnobs.length; ki++, oIdx++) {
            const { built } = buildAt(overflowKnobs[oIdx], startX + ki * (normalKnobSize.w + gap), startY + cy);
            result.push(built);
          }
          cy += knobRowH;
        }
      }
    }

    // Layout remaining types across full width
    const layoutFullRow = (items, withLabels) => {
      if (items.length === 0) return;
      const dims2 = items.map(c => {
        const s2 = sizes[c.type] || sizes.knob;
        return { w: c.width || s2.w, h: c.height || s2.h };
      });
      const totalW2 = dims2.reduce((s, d) => s + d.w, 0) + (items.length - 1) * gap;
      const maxH2 = Math.max(...dims2.map(d => d.h));
      const offX2 = Math.max(0, Math.floor((areaW - totalW2) / 2));
      let cx2 = offX2;
      for (let i = 0; i < items.length; i++) {
        const { built } = buildAt(items[i], startX + cx2, startY + cy);
        result.push(built);
        cx2 += dims2[i].w + gap;
      }
      cy += maxH2 + (withLabels ? labelExtra : 0) + gap;
    };

    layoutFullRow(verticals, true);
    // Skip toggles here — utility buttons already placed in right column,
    // On/Bypass buttons handled by finalizeSection header
    layoutFullRow(others, false);

    // ── Stretch left-column displays to fill vertical space ──
    // If right column is taller, expand waveforms to absorb 80% of the gap
    // so the display fills alongside the controls.
    if (leftCy < cy && leftDisplayComps.length > 0) {
      const extraH = cy - leftCy;
      const maxStretchPer = Math.min(Math.floor(extraH * 0.8 / leftDisplayComps.length), 80);
      const extraPer = Math.min(Math.floor(extraH / leftDisplayComps.length), maxStretchPer);
      let shift = 0;
      for (const dc of leftDisplayComps) {
        dc.y += shift;
        const newH = dc.height + extraPer;
        dc.height = Math.min(newH, 200); // Hard cap at 200px
        shift += extraPer;
      }
    }

  } else {
    // ── Standard grouped layout ──
    // Auto-detect label-prefix groups for organized sub-columns.
    // When a section has no display and controls share label prefixes
    // (e.g., "Rev Size", "Rev Mix", "Dly Time"), create organized sub-panel
    // columns instead of one flat undifferentiated row.
    const allGroupable = [...knobs, ...toggles];
    let usedGroupLayout = false;

    if (allGroupable.length >= 4) {
      const prefixMap = {};
      for (const comp of allGroupable) {
        const label = comp.label || '';
        const prefix = label.split(/\s+/)[0];
        if (prefix && prefix.length >= 2) {
          if (!prefixMap[prefix]) prefixMap[prefix] = [];
          prefixMap[prefix].push(comp);
        }
      }

      const groups = Object.entries(prefixMap)
        .filter(([, items]) => items.length >= 3)
        .map(([prefix, items]) => ({
          prefix,
          knobs: items.filter(c => c.type === 'knob'),
          buttons: items.filter(c => c.type === 'button' || c.type === 'led'),
        }));

      // Fuzzy match: include items whose label starts with a group prefix
      // (e.g., LED "Reverb" matches "Rev" group, LED "Delay" matches "Dly")
      // Also suppress sub-panels when section has many sliders (mixer-style)
      if (groups.length >= 2 && verticals.length < 3) {
        const groupedSet = new Set(groups.flatMap(g => [...g.knobs, ...g.buttons]));
        for (const comp of allGroupable) {
          if (groupedSet.has(comp)) continue;
          const label = (comp.label || '').toLowerCase();
          for (const g of groups) {
            if (label.startsWith(g.prefix.toLowerCase())) {
              if (comp.type === 'knob') g.knobs.push(comp);
              else g.buttons.push(comp);
              groupedSet.add(comp);
              break;
            }
          }
        }
        usedGroupLayout = true;

        // Expand abbreviated prefixes to readable sub-labels
        const EXPAND = {
          Rev: 'Reverb', Dly: 'Delay', Ch: 'Chorus', Fl: 'Flanger',
          Ph: 'Phaser', Dist: 'Distortion', Cmp: 'Compressor', Sat: 'Saturator',
        };

        const colGap = gap;
        const colW = Math.floor((areaW - (groups.length - 1) * colGap) / groups.length);
        const subLabelH = compact === 'medium' ? 10 : compact ? 9 : 12;
        const subPad = compact === 'medium' ? 4 : compact ? 3 : 6;

        // Displays and selectors across full width before grouped columns
        if (displays.length > 0) layoutMixedRow(displays, false);
        layoutMixedRow(selectors, false);

        const groupStartY = cy;
        let maxGroupH = 0;
        const subPanels = [];

        for (let gi = 0; gi < groups.length; gi++) {
          const group = groups[gi];
          const colX = startX + gi * (colW + colGap);
          let gcy = subPad;

          // Sub-panel background (height updated after layout)
          const subPanel = {
            type: 'panel', label: '',
            x: colX, y: startY + groupStartY,
            width: colW, height: 0,
            bgColor: theme.panelColor, borderColor: theme.panelBorder,
            borderRadius: compact === 'medium' ? 3 : 4,
            zIndex: 1, opacity: 0.4,
          };
          result.push(subPanel);
          subPanels.push(subPanel);

          // Sub-label — tight-fit, visible (matches gold standard)
          const fullName = EXPAND[group.prefix] || group.prefix;
          const subText = fullName.toUpperCase();
          const subTextW = Math.min(colW - subPad * 2, subText.length * 7 + 10);
          result.push({
            type: 'label', label: subText,
            x: colX + subPad, y: startY + groupStartY + gcy,
            width: subTextW, height: subLabelH,
            color: theme.textColor, fontSize: compact === 'medium' ? 9 : compact ? 9 : 9,
            fontWeight: 500, zIndex: 10, opacity: 1, letterSpacing: '0.06em',
          });
          gcy += subLabelH + subPad;

          // Knobs in grid within column
          const knobSize = sizes.knob;
          const innerColW = colW - subPad * 2;
          const maxPerRow = Math.max(1, Math.floor((innerColW + gap) / (knobSize.w + gap)));
          const gKnobs = group.knobs;
          const perRow = Math.min(maxPerRow, gKnobs.length || 1);
          const numKnobRows = Math.ceil(gKnobs.length / perRow);
          let kIdx = 0;

          for (let kr = 0; kr < numKnobRows; kr++) {
            const remaining = gKnobs.length - kIdx;
            const rowsLeft = numKnobRows - kr;
            const count = Math.ceil(remaining / rowsLeft);
            for (let ki = 0; ki < count && kIdx < gKnobs.length; ki++, kIdx++) {
              const { built } = buildAt(gKnobs[kIdx],
                colX + subPad + ki * (knobSize.w + gap),
                startY + groupStartY + gcy);
              result.push(built);
            }
            gcy += knobSize.h + labelExtra + gap;
          }

          // Buttons below knobs, centered
          if (group.buttons.length > 0) {
            const btnDims = group.buttons.map(b => {
              const bs = sizes[b.type] || sizes.button;
              return { w: b.width || bs.w, h: b.height || bs.h };
            });
            const totalBW = btnDims.reduce((s, d) => s + d.w, 0) + (group.buttons.length - 1) * gap;
            const bOffX = Math.max(0, Math.floor((innerColW - totalBW) / 2));
            let bx = bOffX;
            let maxBH = 0;
            for (let bi = 0; bi < group.buttons.length; bi++) {
              const { built } = buildAt(group.buttons[bi],
                colX + subPad + bx,
                startY + groupStartY + gcy);
              result.push(built);
              bx += btnDims[bi].w + gap;
              maxBH = Math.max(maxBH, btnDims[bi].h);
            }
            gcy += maxBH + gap;
          }

          gcy += subPad; // bottom padding
          subPanel.height = gcy;
          maxGroupH = Math.max(maxGroupH, gcy);
        }

        // Equalize sub-panel heights to the tallest group
        for (const sp of subPanels) sp.height = maxGroupH;

        cy = groupStartY + maxGroupH + gap;

        // Ungrouped components (if any)
        const allGroupedSet = new Set(groups.flatMap(g => [...g.knobs, ...g.buttons]));
        const ungroupedKnobs = knobs.filter(k => !allGroupedSet.has(k));
        const ungroupedToggles = toggles.filter(t => !allGroupedSet.has(t));

        if (ungroupedKnobs.length > 0) layoutUniformGrid(ungroupedKnobs, true);
        layoutMixedRow(verticals, true);
        if (ungroupedToggles.length > 0) layoutMixedRow(ungroupedToggles, false);
        layoutMixedRow(others, false);
      }
    }

    if (!usedGroupLayout) {
      // ── Check for slider-group layout (faders in row, knobs revealed on hover) ──
      const useSliderGroup = verticals.length >= 3 && knobs.length > 0 && displays.length === 0;

      if (useSliderGroup) {
        // 1. Selectors at top
        layoutMixedRow(selectors, false);

        // 2. Position sliders/meters in a centered horizontal row
        const vDims = verticals.map(v => {
          const s = sizes[v.type] || sizes.slider;
          const scale = v.size === 'large' ? 1.4 : v.size === 'small' ? 0.7 : 1;
          return { w: Math.round(s.w * scale), h: Math.round(s.h * scale) };
        });
        const totalVW = vDims.reduce((s, d) => s + d.w, 0) + (verticals.length - 1) * (gap + 8);
        const maxVH = Math.max(...vDims.map(d => d.h));
        const vOffX = Math.max(0, Math.floor((areaW - totalVW) / 2));
        let vx = vOffX;
        for (let vi = 0; vi < verticals.length; vi++) {
          const { built } = buildAt(verticals[vi], startX + vx, startY + cy);
          // Strip common suffixes for groupId
          const rawLabel = (verticals[vi].label || '').replace(/\s*(level|vol|volume|mix|amount)\s*$/i, '').trim();
          const gid = 'g_' + rawLabel.split(/\s+/)[0].toLowerCase();
          built.isGroupParent = true;
          built.groupId = gid;
          result.push(built);
          vx += vDims[vi].w + gap + 8;
        }
        cy += maxVH + labelExtra + gap;

        // 3. Match knobs/buttons to slider groups by label prefix
        const groupIds = new Map();
        for (const v of verticals) {
          const rawLabel = (v.label || '').replace(/\s*(level|vol|volume|mix|amount)\s*$/i, '').trim();
          const prefix = rawLabel.split(/\s+/)[0].toLowerCase();
          const gid = 'g_' + prefix;
          if (!groupIds.has(gid)) groupIds.set(gid, prefix);
        }
        const matchedKnobs = new Set();
        const matchedToggles = new Set();
        const knobGroups = new Map(); // gid → [knobs]
        const toggleGroups = new Map();
        for (const [gid, prefix] of groupIds) {
          knobGroups.set(gid, []);
          toggleGroups.set(gid, []);
        }
        for (const k of knobs) {
          const kl = (k.label || '').toLowerCase();
          for (const [gid, prefix] of groupIds) {
            if (kl.startsWith(prefix)) {
              knobGroups.get(gid).push(k);
              matchedKnobs.add(k);
              break;
            }
          }
        }
        for (const b of toggles) {
          const bl = (b.label || '').toLowerCase();
          for (const [gid, prefix] of groupIds) {
            if (bl.startsWith(prefix)) {
              toggleGroups.get(gid).push(b);
              matchedToggles.add(b);
              break;
            }
          }
        }

        // 4. Position each group's detail knobs (overlapping, hidden by default)
        // Center each group's items horizontally (consistent with centered sliders above)
        const detailY = cy;
        let maxDetailH = 0;
        let firstGroupId = null;
        for (const [gid] of groupIds) {
          const gKnobs = knobGroups.get(gid) || [];
          const gToggles = toggleGroups.get(gid) || [];
          const allItems = [...gKnobs, ...gToggles];
          if (allItems.length === 0) continue;
          if (!firstGroupId) firstGroupId = gid;
          // Compute total width for centering
          const itemDims = allItems.map(item => {
            const itemSize = sizes[item.type] || sizes.knob;
            const scale = item.size === 'large' ? 1.4 : item.size === 'small' ? 0.7 : 1;
            return { w: Math.round(itemSize.w * scale), h: Math.round(itemSize.h * scale) };
          });
          const totalItemW = itemDims.reduce((s, d) => s + d.w, 0) + (allItems.length - 1) * gap;
          const centerOffX = Math.max(0, Math.floor((areaW - totalItemW) / 2));
          let dx = centerOffX;
          let rowH = 0;
          for (let ii = 0; ii < allItems.length; ii++) {
            const item = allItems[ii];
            const { w, h } = itemDims[ii];
            if (dx + w > areaW && dx > centerOffX) { dx = centerOffX; cy += h + labelExtra + gap; }
            const { built } = buildAt(item, startX + dx, startY + detailY);
            built.groupId = gid;
            built.groupHidden = gid !== firstGroupId;
            if (gid === firstGroupId) built.groupDefault = true;
            result.push(built);
            dx += w + gap;
            rowH = Math.max(rowH, h + labelExtra);
          }
          maxDetailH = Math.max(maxDetailH, rowH);
        }
        cy = detailY + maxDetailH + gap;

        // 5. Unmatched knobs/buttons (always visible, centered)
        const unmatchedKnobs = knobs.filter(k => !matchedKnobs.has(k));
        const unmatchedToggles = toggles.filter(b => !matchedToggles.has(b));
        // Merge small toggles with knobs so they don't get a lonely row
        if (unmatchedKnobs.length > 0 && unmatchedToggles.length > 0) {
          layoutMixedRow([...unmatchedKnobs, ...unmatchedToggles], true);
        } else if (unmatchedKnobs.length > 0) {
          layoutMixedRow(unmatchedKnobs, true);
        } else if (unmatchedToggles.length > 0) {
          layoutMixedRow(unmatchedToggles, false);
        }
        layoutMixedRow(others, false);

      } else {
        // ── Standard flat layout ──
        // 1. Displays — FULL WIDTH at top (Serum-style: wavetable spans entire panel)
        for (const d of displays) {
          const dSize = sizes[d.type] || sizes.waveform;
          const dH = d.height || dSize.h;
          const { built } = buildAt(d, startX, startY + cy);
          built.width = areaW;
          built.height = dH;
          result.push(built);
          cy += dH + gap;
        }
        // 2. Selectors (dropdowns) — selection controls
        layoutMixedRow(selectors, false);
        // 3. Vertical controls (sliders, meters) — tall hero elements first
        layoutMixedRow(verticals, true);
        // 4. Knobs — main control grid, balanced symmetric rows
        //    Merge small toggles (buttons/LEDs) with knobs so they don't get a lonely row
        const onBtnPatFlat = /\b(on|enable|bypass|power)\b/i;
        const utilTogglesFlat = toggles.filter(t => t.type === 'button' && !onBtnPatFlat.test(t.label));
        const remainTogglesFlat = toggles.filter(t => !utilTogglesFlat.includes(t));
        if (knobs.length > 0 && utilTogglesFlat.length > 0) {
          layoutUniformGrid([...knobs, ...utilTogglesFlat], true);
        } else {
          layoutUniformGrid(knobs, true);
          if (utilTogglesFlat.length > 0) layoutMixedRow(utilTogglesFlat, false);
        }
        // 5. Remaining toggles (On/Bypass LEDs)
        if (remainTogglesFlat.length > 0) layoutMixedRow(remainTogglesFlat, false);
        // 6. Others
        layoutMixedRow(others, false);
      }
    }

    // Align small non-grouped sections: left-align rows then center the block.
    // Per-row centering causes misalignment when rows have different widths
    // (e.g., 90px dropdown centered differently from 136px knob row).
    if (!usedGroupLayout && result.length > 0 && result.length <= 12) {
      const minCompX = Math.min(...result.map(c => c.x));
      const maxCompRight = Math.max(...result.map(c => c.x + c.width));
      const totalContentW = maxCompRight - minCompX;
      // Left-align all rows to remove per-row centering differences
      const leftShift = startX - minCompX;
      for (const c of result) c.x += leftShift;
      // Stretch single dropdown to match widest row (cleaner look)
      const ddComps = result.filter(c => c.type === 'dropdown');
      if (ddComps.length === 1) {
        ddComps[0].width = Math.max(ddComps[0].width, totalContentW);
      }
      // Re-measure and center the whole block
      const newMaxRight = Math.max(...result.map(c => c.x + c.width));
      const newContentW = newMaxRight - startX;
      const centerShift = Math.floor((areaW - newContentW) / 2);
      for (const c of result) c.x += centerShift;
    }
  }

  // ── Vertically center content within available panel area ──
  // Skip centering in measurement pass (noCentering=true) — caller handles it
  if (!noCentering) {
    const contentHeight = cy > gap ? cy - gap : cy;
    if (contentHeight > 0 && contentHeight < areaH) {
      const yOffset = Math.floor((areaH - contentHeight) / 2);
      for (const comp of result) {
        comp.y += yOffset;
      }
    }
  }

  return result;
}

// ── Background prompt sanitizer ──────────────────────────────────────────────

function sanitizeBackgroundPrompt(prompt) {
  if (!prompt) return '';
  // Aggressively strip ALL instrument/panel/electronic references.
  // DALL-E renders synths if it sees these words — replace with pure material language.
  let clean = prompt
    .replace(/\b(moog|minimoog|mini\s*moog|synthesizer|synth|keyboard|piano|organ|guitar|drum machine|drum pad|bass station|prophet|jupiter|juno|oberheim|arp|buchla|roland|korg|neve|ssl|api)\b/gi, '')
    .replace(/\b(faceplate|control\s*panel|control\s*surface|rack\s*unit|audio\s*equipment|studio\s*gear|mixing\s*desk|console)\b/gi, 'surface')
    .replace(/\b(with\s+knobs|with\s+buttons|with\s+sliders|with\s+faders|with\s+controls|for\s+controls|ready\s+for\s+controls)\b/gi, '')
    .replace(/\b(knobs|buttons|sliders|faders|dials|switches|potentiometers|jacks|patch\s+points|controls|LEDs|meters|displays)\b/gi, '')
    .replace(/\b(empty\s+panel|bare\s+panel|panel\s+surface|empty\s+faceplate)\b/gi, 'flat surface')
    .replace(/\b(no\s+knobs|no\s+buttons|no\s+controls|no\s+sliders|no\s+UI|no\s+text|no\s+instruments)\b/gi, '')
    .replace(/\b(silk-screened|label\s+areas|section\s+dividers|ventilation|mounting\s+holes)\b/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
  // Ensure it ends with material-only instruction
  if (clean) {
    clean += ', flat surface material texture only, product photography top-down lighting, photorealistic 4k';
  }
  return clean;
}

// ── Color helpers ────────────────────────────────────────────────────────────

function adjustColorForBody(bgColor) {
  // Body should be CLEARLY distinct from background — high contrast
  const { r, g, b } = hexToRgb(bgColor);
  const avg = (r + g + b) / 3;
  if (avg < 80) {
    // Very dark bg: body needs to be substantially lighter
    return rgbToHex(
      Math.min(255, r + 100),
      Math.min(255, g + 95),
      Math.min(255, b + 90)
    );
  } else if (avg < 128) {
    // Medium-dark bg: moderately lighter
    return rgbToHex(
      Math.min(255, r + 80),
      Math.min(255, g + 75),
      Math.min(255, b + 70)
    );
  } else {
    // Light bg: body is noticeably darker
    return rgbToHex(
      Math.max(0, r - 70),
      Math.max(0, g - 70),
      Math.max(0, b - 70)
    );
  }
}

function adjustColorForAccent(accentColor) {
  // Accent ring/border: darker version of accent, still visible
  const { r, g, b } = hexToRgb(accentColor);
  return rgbToHex(
    Math.min(255, Math.round(r * 0.6 + 30)),
    Math.min(255, Math.round(g * 0.6 + 30)),
    Math.min(255, Math.round(b * 0.6 + 30))
  );
}

function darkenHex(hex, amount) {
  const { r, g, b } = hexToRgb(hex);
  return rgbToHex(r * (1 - amount), g * (1 - amount), b * (1 - amount));
}

function hexToRgb(hex) {
  hex = hex.replace('#', '');
  if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
  const n = parseInt(hex, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function rgbToHex(r, g, b) {
  const clamp = v => Math.max(0, Math.min(255, Math.round(v)));
  return '#' + [clamp(r), clamp(g), clamp(b)].map(v => v.toString(16).padStart(2, '0')).join('');
}

// ── Exports for reuse by sectionDetector / StructureView ────────────────────
export {
  layoutSectionContent, positionComponents, assessDensity, isSectionCompact, getSectionGap, finalizeSection,
  PADDING, COMPACT_PADDING, MEDIUM_PADDING,
  SECTION_GAP, COMPACT_SECTION_GAP, MEDIUM_SECTION_GAP,
  SECTION_LABEL_HEIGHT, COMPACT_SECTION_LABEL_HEIGHT, MEDIUM_SECTION_LABEL_HEIGHT,
  COMPONENT_GAP, COMPACT_COMPONENT_GAP, MEDIUM_COMPONENT_GAP,
  TITLE_BAR_HEIGHT,
  TYPE_SIZES, MEDIUM_TYPE_SIZES, COMPACT_TYPE_SIZES,
  generateTheme,
};
