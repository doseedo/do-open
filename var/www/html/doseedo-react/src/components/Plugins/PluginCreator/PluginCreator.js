import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import PluginCanvas from './PluginCanvas';
import ComponentPalette from './ComponentPalette';
import PropertyPanel from './PropertyPanel';
import MasterChat from './MasterChat';
import AudioTestPanel from './AudioTestPanel';
import DSPEditor from './DSPEditor';
import LayersPanel from './LayersPanel';
import ImageBrowser from './ImageBrowser';
import AssetBrowser from './AssetBrowser';
import useUndoRedo from './useUndoRedo';
import StructureView, { StructureTreePanel } from './StructureView';
import { buildSectionMap, detectGridStructure } from './sectionDetector';
import WebAudioDSPEngine from '../../../audio/WebAudioDSPEngine';
import { generateImage, generateFluxImage, searchImages } from '../../../services/chatAPI';
import { saveProject, loadProject, publishProject, generateCode, buildPlugin, buildPluginAutoFix, listMyProjects, deleteProject } from '../../../services/pluginProjectsAPI';
import { publishCreation, unpublishCreation } from '../../../services/communityAPI';
import ContextMenu from './ContextMenu';
import { PLUGIN_TEMPLATES } from './templates';
import { applyTheme } from './themes';
import { injectTexturePattern } from './svgSanitizer';
import { generateKnobSVG, generateSliderSVG, generateButtonSVG } from './svgComponentLibrary';
import { PluginContext } from "./PluginContext";
import { PluginParamControllerUI } from "./PluginParamControllerUI";
import ModMatrixWithSliders from "./ModMatrixWithSliders";
import ADSRDisplay from "./ADSRDisplay";
import MSEGEditor from "./MSEGEditor";
import EQCurveDisplay from "./EQCurveDisplay";
import styles from "./PluginCreator.module.css";

const GUIDE_THRESHOLD = 5; // px for smart guide snapping

const COMPONENT_DEFAULTS = {
  knob:     { width: 60,  height: 70,  color: '#667eea', label: 'Knob' },
  slider:   { width: 30,  height: 120, color: '#667eea', label: 'Slider' },
  button:   { width: 70,  height: 28,  color: '#764ba2', label: 'Button' },
  label:    { width: 100, height: 24,  color: '#ffffff', label: 'Label' },
  'mseg-editor': { width: 300, height: 150, color: '#667eea', label: 'MSEG' },
  'oscilloscope': { width: 200, height: 100, color: '#00ff88', label: 'Scope' },
  'adsr':   { width: 200, height: 100, color: '#e74c3c', label: 'ADSR' },
  'eq-curve': { width: 400, height: 150, color: '#3498db', label: 'EQ' },
  'mod-matrix': { width: 350, height: 250, color: '#9b59b6', label: 'Mod Matrix' },
  led:      { width: 12,  height: 12,  color: '#4caf50', label: '' },
  dropdown: { width: 120, height: 28,  color: '#667eea', label: 'Select' },
  image:    { width: 80,  height: 80,  color: '#444',    label: 'Image', image: '' },
  panel:    { width: 200, height: 150, color: '#2a2a4a', label: 'Panel', borderColor: 'rgba(255,255,255,0.1)', bgColor: 'rgba(255,255,255,0.03)' },
  meter:    { width: 24,  height: 100, color: '#4caf50', label: 'Meter' },
  waveform: { width: 180, height: 60,  color: '#667eea', label: 'Waveform' },
  'xy-pad':           { width: 120, height: 120, color: '#667eea', label: 'XY Pad' },
  'click-knob':       { width: 60,  height: 70,  color: '#667eea', label: 'Precision Knob' },
  'spectrum-analyzer':{ width: 200, height: 80,  color: '#4ecdc4', label: 'Spectrum' },
};

const ACCESS_KEY = '***REDACTED***';

let idCounter = 0;

// Mini canvas thumbnail for template previews
const TemplateThumbnail = ({ template }) => {
  const cfg = template.pluginConfig || {};
  const comps = template.components || [];
  if (comps.length === 0) return null;
  const w = cfg.width || 600;
  const h = cfg.height || 400;
  const scale = Math.min(120 / w, 70 / h);
  const COMP_COLORS = { knob: '#667eea', slider: '#667eea', button: '#764ba2', label: '#fff', led: '#4caf50', panel: 'rgba(255,255,255,0.08)', meter: '#4caf50', dropdown: '#667eea', image: '#555', 'xy-pad': '#667eea', 'wavetable_3d': '#4ecdc4', oscilloscope: '#4caf50', 'spectrum-analyzer': '#ff9800', adsr: '#667eea' };
  return (
    <div style={{
      width: w * scale, height: h * scale, position: 'relative',
      background: cfg.bgColor || '#1a1a2e', borderRadius: 4, overflow: 'hidden',
      border: '1px solid rgba(255,255,255,0.1)', margin: '0 auto',
    }}>
      {comps.map((c, i) => {
        const isCircle = ['knob', 'led', 'click-knob'].includes(c.type);
        return (
          <div key={i} style={{
            position: 'absolute',
            left: (c.x || 0) * scale, top: (c.y || 0) * scale,
            width: (c.width || 20) * scale, height: (c.height || 20) * scale,
            borderRadius: isCircle ? '50%' : 2,
            background: c.color || COMP_COLORS[c.type] || '#555',
            opacity: c.type === 'panel' ? 0.15 : c.type === 'label' ? 0.3 : 0.5,
          }} />
        );
      })}
    </div>
  );
};

const PluginCreator = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [unlocked, setUnlocked] = useState(() => sessionStorage.getItem('pc_auth') === '1');
  // Param controller for modulation routing (shared via context)
  const paramController = useMemo(() => new PluginParamControllerUI(), []);
  const [passInput, setPassInput] = useState('');
  const [passError, setPassError] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]); // multi-select: array of component ids
  const [rubberBand, setRubberBand] = useState(null); // {startX, startY, currentX, currentY}
  const [showLayers, setShowLayers] = useState(false);
  const [showImageBrowser, setShowImageBrowser] = useState(false);
  const [imageBrowserTarget, setImageBrowserTarget] = useState(null); // 'background' | componentId
  const [showDspEditor, setShowDspEditor] = useState(false);
  const [dspSubTab, setDspSubTab] = useState('graph'); // 'graph' | 'modmatrix' | 'adsr' | 'eq' | 'mseg'
  const [chatHistory, setChatHistory] = useState([]);
  const [dspConfig, setDspConfig] = useState(null);
  const [editorMode, setEditorMode] = useState('edit'); // 'edit' | 'test'
  const [paramValues, setParamValues] = useState({}); // componentId → 0-1 value
  const [snapToGrid, setSnapToGrid] = useState(false);
  const [gridSize, setGridSize] = useState(20);
  const [smartGuides, setSmartGuides] = useState([]); // [{type:'vertical'|'horizontal', position: number}]
  const [contextMenu, setContextMenu] = useState(null); // {x, y, items}
  const [activeTheme, setActiveTheme] = useState(null); // current theme object
  const [chatPanelTab, setChatPanelTab] = useState('designer'); // 'designer' | 'assets'
  const [showWelcome, setShowWelcome] = useState(true); // template gallery on first visit
  const [welcomeStep, setWelcomeStep] = useState('choose'); // 'choose' | 'templates'
  const clipboardRef = useRef([]); // copy/paste clipboard
  const [pluginConfig, setPluginConfig] = useState({
    name: 'My Plugin',
    width: 600,
    height: 400,
    bgColor: '#1a1a2e',
    titleBarColor: '#2d2d4e',
    bgImage: '',
  });

  // ── Save/Load state ──────────────────────────────────────────────────────
  const [projectId, setProjectId] = useState(null);
  const [saveStatus, setSaveStatus] = useState('idle'); // 'idle' | 'saving' | 'saved' | 'error'
  const [isPublished, setIsPublished] = useState(false);
  const [showPublishModal, setShowPublishModal] = useState(false);
  const [publishDesc, setPublishDesc] = useState('');
  const [publishTags, setPublishTags] = useState('');
  const [publishLoading, setPublishLoading] = useState(false);
  const [toast, setToast] = useState(null); // { message, link?, type: 'success'|'error' }
  const [codePreview, setCodePreview] = useState(null);
  const [showCodePreview, setShowCodePreview] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showProjectPicker, setShowProjectPicker] = useState(false);
  const [myProjects, setMyProjects] = useState([]);
  const [activeCodeFile, setActiveCodeFile] = useState(null);
  const [generatingCode, setGeneratingCode] = useState(false);
  const [buildingPlugin, setBuildingPlugin] = useState(false);
  const [buildError, setBuildError] = useState(null);
  const [buildBlob, setBuildBlob] = useState(null);
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const saveTimerRef = useRef(null);
  const lastSaveRef = useRef(null);
  const [isDirty, setIsDirty] = useState(false);
  const [chatPanelWidth, setChatPanelWidth] = useState(340);
  const isResizingRef = useRef(false);
  const resizeStartXRef = useRef(0);
  const resizeStartWidthRef = useRef(340);
  const [manualParamMap, setManualParamMap] = useState({});
  const [presets, setPresets] = useState([]);
  const [activePresetIdx, setActivePresetIdx] = useState(-1);
  const [showPresetPanel, setShowPresetPanel] = useState(false);
  const [presetNameInput, setPresetNameInput] = useState('');
  const [showPresetNameInput, setShowPresetNameInput] = useState(false);
  const pluginFrameRef = useRef(null);
  const [canvasZoom, setCanvasZoom] = useState(1.0);
  const pinchRef = useRef(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const [generatingImages, setGeneratingImages] = useState(new Set()); // component IDs currently generating images

  const { state: components, pushState: pushComponents, undo, redo, setState: setComponentsDirect, reset: resetComponents } = useUndoRedo([]);

  const snapValue = useCallback((val) => {
    if (!snapToGrid) return val;
    return Math.round(val / gridSize) * gridSize;
  }, [snapToGrid, gridSize]);

  // ── Reset all state for "New Project" ──────────────────────────────────
  const resetProject = useCallback(() => {
    setProjectId(null);
    setPluginConfig({ name: 'My Plugin', width: 600, height: 400, bgColor: '#1a1a2e', titleBarColor: '#2d2d4e', bgImage: '' });
    resetComponents([]);
    setDspConfig(null);
    setChatHistory([]);
    setPresets([]);
    setActivePresetIdx(-1);
    setParamValues({});
    setSelectedIds([]);
    setIsPublished(false);
    setActiveTheme(null);
    setShowWelcome(true);
    setWelcomeStep('choose');
    setIsDirty(false);
    setCodePreview(null);
    setBuildBlob(null);
    setBuildError(null);
    // Clear localStorage chat so old messages don't bleed into new project
    try { localStorage.removeItem('plugin_creator_chat_history'); } catch (e) { /* noop */ }
  }, [resetComponents]);

  // ── Toast helper ─────────────────────────────────────────────────────────
  const showToast = useCallback((message, link = null, type = 'success') => {
    setToast({ message, link, type });
    setTimeout(() => setToast(null), 4000);
  }, []);

  // ── Save project ─────────────────────────────────────────────────────────
  const handleSave = useCallback(async (silent = false) => {
    if (!silent) setSaveStatus('saving');
    try {
      // Capture thumbnail if canvas exists
      let thumbnailData = undefined;
      if (pluginFrameRef.current && components.length > 0) {
        try {
          const html2canvas = (await import('html2canvas')).default;
          const canvas = await html2canvas(pluginFrameRef.current, { scale: 0.5, backgroundColor: null, useCORS: true, logging: false });
          thumbnailData = canvas.toDataURL('image/jpeg', 0.6);
        } catch (e) { /* thumbnail capture is best-effort */ }
      }
      const dspWithPresets = dspConfig ? { ...dspConfig, presets: presets.length > 0 ? presets : undefined } : dspConfig;
      const projectData = {
        id: projectId || undefined,
        name: pluginConfig.name,
        plugin_config: pluginConfig,
        components,
        dsp_config: dspWithPresets,
        thumbnail_data: thumbnailData,
      };
      const result = await saveProject(projectData);
      if (result.id && !projectId) {
        setProjectId(result.id);
        setSearchParams({ project: result.id }, { replace: true });
      }
      setSaveStatus('saved');
      setIsDirty(false);
      lastSaveRef.current = Date.now();
      // Reset saved indicator after 2s
      setTimeout(() => setSaveStatus(prev => prev === 'saved' ? 'idle' : prev), 2000);
    } catch (err) {
      console.error('Save failed:', err);
      if (!silent) setSaveStatus('error');
      setTimeout(() => setSaveStatus(prev => prev === 'error' ? 'idle' : prev), 3000);
    }
  }, [projectId, pluginConfig, components, dspConfig, presets, setSearchParams]);

  // Auto-save debounced (5 seconds after last change)
  useEffect(() => {
    if (!unlocked) return;
    // Don't auto-save if nothing has changed or it's too early
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      // Only auto-save if there are components or dsp config (not an empty project)
      if (components.length > 0 || dspConfig) {
        handleSave(true);
      }
    }, 5000);
    return () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); };
  }, [pluginConfig, components, dspConfig, unlocked]); // eslint-disable-line react-hooks/exhaustive-deps

  // Warn before closing tab with unsaved work
  useEffect(() => {
    const handler = (e) => {
      if (components.length > 0 || dspConfig) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [components, dspConfig]);

  // Mark dirty on changes
  useEffect(() => {
    if (!unlocked) return;
    setIsDirty(true);
  }, [components, dspConfig, pluginConfig]); // eslint-disable-line react-hooks/exhaustive-deps

  // Resizable panel drag
  useEffect(() => {
    const onMouseMove = (e) => {
      if (!isResizingRef.current) return;
      const delta = e.clientX - resizeStartXRef.current;
      setChatPanelWidth(Math.max(200, Math.min(600, resizeStartWidthRef.current + delta)));
    };
    const onMouseUp = () => { isResizingRef.current = false; };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => { window.removeEventListener('mousemove', onMouseMove); window.removeEventListener('mouseup', onMouseUp); };
  }, []);

  // ── Load project on mount / reset when project param removed ──────────────
  const projectParam = searchParams.get('project');
  const prevProjectParamRef = useRef(projectParam);
  useEffect(() => {
    const prev = prevProjectParamRef.current;
    prevProjectParamRef.current = projectParam;
    // If project param was removed (e.g. navigated to /plugins/create), reset state
    if (prev && !projectParam) {
      resetProject();
      return;
    }
    if (projectParam && unlocked) {
      loadProject(projectParam).then(data => {
        setProjectId(data.id);
        setPluginConfig(data.plugin_config || {
          name: data.name || 'My Plugin',
          width: 600, height: 400, bgColor: '#1a1a2e', titleBarColor: '#2d2d4e', bgImage: '',
        });
        if (data.plugin_config?.name) {
          setPluginConfig(prev => ({ ...prev, name: data.plugin_config.name || data.name }));
        }
        if (data.components && data.components.length > 0) {
          pushComponents(data.components);
        }
        if (data.dsp_config) {
          setDspConfig(data.dsp_config);
          if (data.dsp_config.presets) setPresets(data.dsp_config.presets);
        }
        setIsPublished(data.is_public || false);
      }).catch(err => {
        console.error('Failed to load project:', err);
        showToast('Failed to load project: ' + (err.message || 'Unknown error'), null, 'error');
      });
    }
  }, [unlocked, projectParam, resetProject]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Desktop plugin clone integration ──────────────────────────────────────
  const dskPlugin = searchParams.get('dsk_plugin');
  const dskKey    = searchParams.get('dsk_key');

  useEffect(() => {
    if (!dskPlugin || !dskKey || !unlocked) return;
    fetch(`/api/plugins/${dskPlugin}`, { headers: { 'X-API-Key': dskKey } })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(plugin => {
        const content = typeof plugin.content === 'string'
          ? JSON.parse(plugin.content) : (plugin.content || {});
        const params = (content.components || []).slice(0, 16);
        const COLS = 4, KW = 80, KH = 90, PAD_X = 140, PAD_Y = 130, START_X = 30, START_Y = 80;
        const PALETTE = ['#4a9eff','#e06c9f','#ffd700','#00c8ff','#b464ff','#00e5a0','#ff6b6b','#94b8de'];
        const canvasComponents = params.map((p, i) => ({
          id: `dsk_${p.id || i}`,
          type: 'knob',
          label: p.label || p.id || `Param ${i + 1}`,
          x: START_X + (i % COLS) * PAD_X,
          y: START_Y + Math.floor(i / COLS) * PAD_Y,
          width: KW, height: KH,
          color: PALETTE[Math.floor(i / COLS) % PALETTE.length],
          min: p.min ?? 0, max: p.max ?? 1,
          default: p.default ?? 0, value: p.default ?? 0,
          unit: p.unit || '',
        }));
        setPluginConfig({
          name: content.name || plugin.name || 'My Plugin',
          width: 600,
          height: Math.max(400, START_Y + Math.ceil(params.length / COLS) * PAD_Y + 20),
          bgColor: '#1a1a2e', titleBarColor: '#2d2d4e', bgImage: '',
        });
        if (canvasComponents.length > 0) pushComponents(canvasComponents);
        if (content.dsp_config) setDspConfig(content.dsp_config);
        setShowWelcome(false);
        showToast(`Loaded from desktop: ${content.name || plugin.name}`, null, 'success');
      })
      .catch(err => showToast('Failed to load desktop plugin: ' + (err.message || 'Unknown'), null, 'error'));
  }, [unlocked, dskPlugin, dskKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Collect images from UI state ─────────────────────────────────────────
  const collectImages = useCallback(() => {
    const images = {};
    // Background image
    if (pluginConfig.bgImage) {
      images['bg_image.png'] = pluginConfig.bgImage; // data URL or https URL
    }
    // Component images
    for (const comp of components) {
      if (comp.type === 'image' && comp.image) {
        const safeId = (comp.id || '').replace(/[^a-zA-Z0-9_]/g, '_');
        images[`${safeId}_image.png`] = comp.image;
      }
    }
    return Object.keys(images).length > 0 ? images : null;
  }, [pluginConfig.bgImage, components]);

  // Auto-map UI component labels to DSP parameter ids
  const paramMapping = useMemo(() => {
    if (!dspConfig?.parameters || components.length === 0) return {};
    const mapping = {}; // paramId → componentId
    const usedCompIds = new Set(); // prevent double-binding
    const interactiveComps = components.filter(c => ['knob', 'slider', 'xy-pad'].includes(c.type));

    // Normalize: strip underscores, spaces, hyphens, lowercase
    const norm = (s) => (s || '').toLowerCase().replace(/[_\s-]/g, '');
    // Split into words for word-level matching
    const words = (s) => (s || '').toLowerCase().replace(/[_-]/g, ' ').trim().split(/\s+/).filter(Boolean);

    // Pass 1: exact matches (highest priority)
    for (const param of dspConfig.parameters) {
      const pNorm = norm(param.name || param.id);
      const pId = norm(param.id);
      for (const comp of interactiveComps) {
        if (usedCompIds.has(comp.id)) continue;
        const cNorm = norm(comp.label);
        if (cNorm === pNorm || cNorm === pId) {
          mapping[param.id] = comp.id;
          usedCompIds.add(comp.id);
          break;
        }
      }
    }

    // Pass 2: last-word match (e.g., DSP "osc_a_level" → knob "Level", DSP "amp_attack" → knob "Attack")
    for (const param of dspConfig.parameters) {
      if (mapping[param.id]) continue;
      const pWords = words(param.name || param.id);
      const lastWord = pWords[pWords.length - 1];
      if (!lastWord || lastWord.length < 3) continue;
      for (const comp of interactiveComps) {
        if (usedCompIds.has(comp.id)) continue;
        const cNorm = norm(comp.label);
        if (cNorm === lastWord) {
          mapping[param.id] = comp.id;
          usedCompIds.add(comp.id);
          break;
        }
      }
    }

    // Pass 3: partial/contains match (lowest priority, still deduplicated)
    for (const param of dspConfig.parameters) {
      if (mapping[param.id]) continue;
      const pNorm = norm(param.name || param.id);
      for (const comp of interactiveComps) {
        if (usedCompIds.has(comp.id)) continue;
        const cNorm = norm(comp.label);
        if (cNorm.length >= 3 && (cNorm.includes(pNorm) || pNorm.includes(cNorm))) {
          mapping[param.id] = comp.id;
          usedCompIds.add(comp.id);
          break;
        }
      }
    }

    // Manual overrides win over auto-matched
    for (const [paramId, compId] of Object.entries(manualParamMap)) {
      if (compId) mapping[paramId] = compId;
      else delete mapping[paramId];
    }

    return mapping;
  }, [dspConfig, components, manualParamMap]);

  // Reverse mapping: componentId → paramId
  const reverseMapping = useMemo(() => {
    const rev = {};
    for (const [paramId, compId] of Object.entries(paramMapping)) {
      rev[compId] = paramId;
    }
    return rev;
  }, [paramMapping]);

  // Manual param remapping callback
  const handleRemapParam = useCallback((compId, paramId) => {
    setManualParamMap(prev => {
      const next = { ...prev };
      // Remove any existing mapping pointing to this component
      for (const [pid, cid] of Object.entries(next)) {
        if (cid === compId) delete next[pid];
      }
      if (paramId) next[paramId] = compId;
      return next;
    });
  }, []);

  // ── Preset system ─────────────────────────────────────────────────────
  const handleSavePreset = useCallback(() => {
    if (!showPresetNameInput) {
      setPresetNameInput(`Preset ${presets.length + 1}`);
      setShowPresetNameInput(true);
      return;
    }
    const name = presetNameInput.trim() || `Preset ${presets.length + 1}`;
    const values = { ...paramValues };
    setPresets(prev => [...prev, { name, values }]);
    setActivePresetIdx(presets.length);
    setShowPresetNameInput(false);
    setPresetNameInput('');
  }, [paramValues, presets, showPresetNameInput, presetNameInput]);

  const handleLoadPreset = useCallback((idx) => {
    const preset = presets[idx];
    if (!preset) return;
    setActivePresetIdx(idx);
    // Apply to param values state
    setParamValues(prev => ({ ...prev, ...preset.values }));
    // Apply to engine
    if (engineRef.current) {
      engineRef.current.setState(preset.values);
    }
  }, [presets]);

  const handleDeletePreset = useCallback((idx) => {
    setPresets(prev => prev.filter((_, i) => i !== idx));
    if (activePresetIdx === idx) setActivePresetIdx(-1);
    else if (activePresetIdx > idx) setActivePresetIdx(prev => prev - 1);
  }, [activePresetIdx]);

  // ── Export canvas as PNG ──────────────────────────────────────────────
  const handleExportPNG = useCallback(async () => {
    if (!pluginFrameRef.current) return;
    try {
      const html2canvas = (await import('html2canvas')).default;
      const canvas = await html2canvas(pluginFrameRef.current, { scale: 2, backgroundColor: null, useCORS: true });
      const link = document.createElement('a');
      link.download = `${pluginConfig.name || 'plugin'}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
      showToast('PNG exported');
    } catch (err) {
      console.error('PNG export failed:', err);
      showToast('Export failed: ' + err.message, null, 'error');
    }
  }, [pluginConfig.name, showToast]);

  // ── Generate JUCE code ───────────────────────────────────────────────────
  const handleGenerateCode = useCallback(async () => {
    if (!dspConfig) return;
    setGeneratingCode(true);
    try {
      const uiLayout = {
        pluginConfig,
        components,
        // Enriched data for better JUCE export
        parameterMapping: Object.entries(paramMapping).map(([paramId, compId]) => {
          const comp = components.find(c => c.id === compId);
          const paramDef = dspConfig?.parameters?.find(p => p.id === paramId);
          return {
            paramId,
            componentId: compId,
            componentType: comp?.type,
            componentLabel: comp?.label,
            min: paramDef?.min ?? 0,
            max: paramDef?.max ?? 1,
            defaultValue: paramDef?.default ?? 0.5,
            unit: paramDef?.unit || '',
            skew: paramDef?.skew || 1,
            automatable: true,
          };
        }),
        presets: (dspConfig?.presets && dspConfig.presets.length > 0)
          ? dspConfig.presets
          : [{
              name: 'Default',
              values: Object.fromEntries(
                (dspConfig?.parameters || []).map(p => [
                  p.id,
                  p.default != null ? p.default : 0.5,
                ])
              ),
            }],
        modConnections: paramController?.getConnections?.() || [],
      };
      const images = collectImages();
      const result = await generateCode(dspConfig, uiLayout, images);
      setCodePreview(result);
      setBuildBlob(null);
      setShowCodePreview(true);
      setActiveCodeFile(Object.keys(result.files)[0] || null);
    } catch (err) {
      console.error('Code generation failed:', err);
    } finally {
      setGeneratingCode(false);
    }
  }, [dspConfig, pluginConfig, components, collectImages, paramMapping, paramController]);

  // ── Build plugin on Mac Mini (with AI auto-fix for compile errors) ──────
  const [buildStage, setBuildStage] = useState('');
  const handleBuildPlugin = useCallback(async () => {
    if (!codePreview?.files) return;
    setBuildingPlugin(true);
    setBuildError(null);
    setBuildBlob(null);
    setBuildStage('Starting build...');
    try {
      const blob = await buildPluginAutoFix(
        pluginConfig.name, codePreview.files, codePreview.images || null,
        {
          onLog: (msg) => setBuildStage(msg.slice(0, 80)),
          onStage: (stage) => setBuildStage(stage),
          onAutoFix: (fixedFiles) => {
            if (fixedFiles) setCodePreview(prev => ({ ...prev, files: fixedFiles }));
          },
        }
      );
      setBuildBlob(blob);
      // Trigger browser download
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${pluginConfig.name.replace(/[^a-zA-Z0-9_-]/g, '_')}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setBuildStage('');
    } catch (err) {
      console.error('Build failed:', err);
      setBuildError(err.message || 'Build failed');
      setBuildStage('');
    } finally {
      setBuildingPlugin(false);
    }
  }, [codePreview, pluginConfig.name]);

  // ── Download specific format from build ────────────────────────────────
  const handleDownloadFormat = useCallback(async (format) => {
    setShowDownloadMenu(false);

    // If we need to generate code first
    if (!codePreview?.files) {
      if (!dspConfig) {
        showToast('Design a DSP chain first', null, 'error');
        return;
      }
      showToast('Generating code first...', null, 'success');
      // trigger code gen, then build — user can re-click after
      handleGenerateCode();
      return;
    }

    // If no build blob yet, trigger build
    if (!buildBlob) {
      setBuildingPlugin(true);
      setBuildError(null);
      setBuildBlob(null);
      setBuildStage('Building plugin...');
      try {
        const blob = await buildPluginAutoFix(
          pluginConfig.name, codePreview.files, codePreview.images || null,
          {
            onLog: (msg) => setBuildStage(msg.slice(0, 80)),
            onStage: (stage) => setBuildStage(stage),
            onAutoFix: (fixedFiles) => {
              if (fixedFiles) setCodePreview(prev => ({ ...prev, files: fixedFiles }));
            },
          }
        );
        setBuildBlob(blob);
        setBuildStage('');
        setBuildingPlugin(false);
        // Now extract the format
        await extractAndDownload(blob, format);
      } catch (err) {
        console.error('Build failed:', err);
        setBuildError(err.message || 'Build failed');
        setBuildStage('');
        setBuildingPlugin(false);
      }
      return;
    }

    // Already have blob — extract
    await extractAndDownload(buildBlob, format);
  }, [buildBlob, codePreview, dspConfig, pluginConfig.name, handleGenerateCode, showToast]);

  const extractAndDownload = useCallback(async (blob, format) => {
    const JSZip = (await import('jszip')).default;
    const zip = await JSZip.loadAsync(blob);
    const safeName = pluginConfig.name.replace(/[^a-zA-Z0-9_-]/g, '_');

    if (format === 'all') {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${safeName}.zip`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
      return;
    }

    // Filter files by format
    const ext = format === 'vst3' ? '.vst3' : format === 'au' ? '.component' : '.pkg';
    const newZip = new JSZip();
    let found = false;
    zip.forEach((path, entry) => {
      if (format === 'pkg') {
        if (path.endsWith('.pkg')) { found = true; newZip.file(path, entry.async('blob')); }
      } else {
        if (path.includes(ext + '/') || path.endsWith(ext)) { found = true; newZip.file(path, entry.async('blob')); }
      }
    });

    if (!found) {
      showToast(`No ${format.toUpperCase()} found in build`, null, 'error');
      return;
    }

    const outBlob = await newZip.generateAsync({ type: 'blob' });
    const label = format === 'vst3' ? 'VST3' : format === 'au' ? 'AU' : 'Installer';
    const url = URL.createObjectURL(outBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${safeName}_${label}.zip`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [pluginConfig.name, showToast]);

  // ── Publish ──────────────────────────────────────────────────────────────
  const handlePublish = useCallback(async () => {
    setPublishLoading(true);
    try {
      if (!projectId) {
        await handleSave(false);
      }
      const pid = projectId;
      if (!pid) { setPublishLoading(false); return; }
      const tags = publishTags.split(',').map(t => t.trim()).filter(Boolean);
      const gcsResult = await publishProject(pid, {
        is_public: true,
        description: publishDesc || undefined,
        tags: tags.length ? tags : undefined,
      });
      const dspSummary = dspConfig ? {
        pluginType: dspConfig.pluginType || 'effect',
        nodeCount: (dspConfig.dspChain || []).length,
        paramCount: (dspConfig.parameters || []).length,
      } : null;
      await publishCreation({
        id: pid,
        creation_type: 'plugin',
        name: pluginConfig.name,
        slug: gcsResult?.slug || `plugin-${pid.substring(0, 8)}`,
        description: publishDesc || undefined,
        tags: tags.length ? tags : [],
        dsp_summary: dspSummary,
      }).catch(err => console.warn('DB publish (non-critical):', err));
      setIsPublished(true);
      setShowPublishModal(false);
      showToast('Published to community!', '/plugins', 'success');
    } catch (err) {
      console.error('Publish failed:', err);
      showToast('Publish failed: ' + (err.message || 'Unknown error'), null, 'error');
    } finally {
      setPublishLoading(false);
    }
  }, [projectId, publishDesc, publishTags, pluginConfig.name, dspConfig, handleSave, showToast]);

  const handleUnpublish = useCallback(async () => {
    if (!projectId) return;
    try {
      await publishProject(projectId, { is_public: false });
      await unpublishCreation(projectId).catch(() => {});
      setIsPublished(false);
    } catch (err) {
      console.error('Unpublish failed:', err);
    }
  }, [projectId]);

  const addComponent = useCallback((type) => {
    const defaults = COMPONENT_DEFAULTS[type];
    if (!defaults) return;
    const id = `comp-${++idCounter}-${Date.now()}`;
    const count = components.filter(c => c.type === type).length;
    const newComp = {
      id,
      type,
      x: snapValue(Math.round(pluginConfig.width / 2 - defaults.width / 2)),
      y: snapValue(Math.round(pluginConfig.height / 2 - defaults.height / 2)),
      ...defaults,
      label: defaults.label + (count > 0 ? ' ' + (count + 1) : ''),
      opacity: 1,
      rotation: 0,
      borderRadius: 0,
      fontSize: 13,
      zIndex: components.length + 1,
    };
    pushComponents([...components, newComp]);
    setSelectedIds([id]);
  }, [pluginConfig.width, pluginConfig.height, components, pushComponents, snapValue]);

  // Add a pre-configured component from Asset Browser
  const addComponentToCanvas = useCallback((componentData) => {
    const defaults = COMPONENT_DEFAULTS[componentData.type] || COMPONENT_DEFAULTS.knob;
    const id = `comp-${++idCounter}-${Date.now()}`;
    const count = components.filter(c => c.type === componentData.type).length;
    const newComp = {
      id,
      type: componentData.type,
      x: snapValue(Math.round(pluginConfig.width / 2 - (componentData.width || defaults.width) / 2)),
      y: snapValue(Math.round(pluginConfig.height / 2 - (componentData.height || defaults.height) / 2)),
      ...defaults,
      ...componentData,
      label: componentData.label || (defaults.label + (count > 0 ? ' ' + (count + 1) : '')),
      opacity: 1,
      rotation: 0,
      borderRadius: 0,
      fontSize: 13,
      zIndex: components.length + 1,
    };
    newComp.id = id; // ensure our generated ID
    pushComponents([...components, newComp]);
    setSelectedIds([id]);
  }, [pluginConfig.width, pluginConfig.height, components, pushComponents, snapValue]);

  const updateComponent = useCallback((id, updates) => {
    const updated = components.map(c => {
      if (c.id !== id) return c;
      const merged = { ...c, ...updates };
      if (snapToGrid && (updates.x !== undefined || updates.y !== undefined)) {
        if (updates.x !== undefined) merged.x = snapValue(updates.x);
        if (updates.y !== undefined) merged.y = snapValue(updates.y);
      }
      return merged;
    });
    pushComponents(updated);
  }, [components, pushComponents, snapToGrid, snapValue]);

  // Smart guide detection during drag
  const detectSmartGuides = useCallback((draggedId, dragX, dragY) => {
    const dragged = components.find(c => c.id === draggedId);
    if (!dragged) return { snappedX: dragX, snappedY: dragY, guides: [] };

    const dw = dragged.width;
    const dh = dragged.height;
    // 3 snap points per axis: left/center/right, top/middle/bottom
    const dragSnapX = [dragX, dragX + dw / 2, dragX + dw];
    const dragSnapY = [dragY, dragY + dh / 2, dragY + dh];

    let bestDeltaX = null, bestDistX = GUIDE_THRESHOLD + 1, guideXPos = null;
    let bestDeltaY = null, bestDistY = GUIDE_THRESHOLD + 1, guideYPos = null;

    for (const other of components) {
      if (other.id === draggedId) continue;
      const otherSnapX = [other.x, other.x + other.width / 2, other.x + other.width];
      const otherSnapY = [other.y, other.y + other.height / 2, other.y + other.height];

      for (const dx of dragSnapX) {
        for (const ox of otherSnapX) {
          const dist = Math.abs(dx - ox);
          if (dist < GUIDE_THRESHOLD && dist < bestDistX) {
            bestDistX = dist;
            bestDeltaX = ox - dx;
            guideXPos = ox;
          }
        }
      }
      for (const dy of dragSnapY) {
        for (const oy of otherSnapY) {
          const dist = Math.abs(dy - oy);
          if (dist < GUIDE_THRESHOLD && dist < bestDistY) {
            bestDistY = dist;
            bestDeltaY = oy - dy;
            guideYPos = oy;
          }
        }
      }
    }

    const snappedX = bestDeltaX !== null ? dragX + bestDeltaX : dragX;
    const snappedY = bestDeltaY !== null ? dragY + bestDeltaY : dragY;
    const guides = [];
    if (guideXPos !== null) guides.push({ type: 'vertical', position: guideXPos });
    if (guideYPos !== null) guides.push({ type: 'horizontal', position: guideYPos });
    return { snappedX, snappedY, guides };
  }, [components]);

  // Track drag start positions for multi-drag
  const dragStartRef = useRef(null);

  // Lightweight drag update that doesn't push to undo history on every pixel
  const updateComponentDrag = useCallback((id, updates) => {
    // Skip if component is locked
    const comp = components.find(c => c.id === id);
    if (comp?.locked) return;

    let finalX = updates.x;
    let finalY = updates.y;

    // Smart guides (always active, takes priority over grid snap)
    if (finalX !== undefined && finalY !== undefined) {
      const result = detectSmartGuides(id, finalX, finalY);
      if (result.guides.length > 0) {
        finalX = result.snappedX;
        finalY = result.snappedY;
        setSmartGuides(result.guides);
      } else {
        setSmartGuides([]);
        // Fall back to grid snap if enabled
        if (snapToGrid) {
          finalX = snapValue(finalX);
          finalY = snapValue(finalY);
        }
      }
    }

    // Multi-drag: if this component is part of selection, move all selected
    if (selectedIds.length > 1 && selectedIds.includes(id)) {
      if (!dragStartRef.current) {
        // Record start positions of all selected components synchronously
        dragStartRef.current = {};
        for (const c of components) {
          if (selectedIds.includes(c.id)) {
            dragStartRef.current[c.id] = { x: c.x, y: c.y };
          }
        }
      }
      const startPos = dragStartRef.current[id];
      if (startPos) {
        const deltaX = (finalX ?? 0) - startPos.x;
        const deltaY = (finalY ?? 0) - startPos.y;
        setComponentsDirect(prev => prev.map(c => {
          if (c.id === id) return { ...c, ...updates, x: finalX ?? c.x, y: finalY ?? c.y };
          if (selectedIds.includes(c.id) && dragStartRef.current[c.id]) {
            return { ...c, x: dragStartRef.current[c.id].x + deltaX, y: dragStartRef.current[c.id].y + deltaY };
          }
          return c;
        }));
      }
    } else {
      setComponentsDirect(prev => prev.map(c => {
        if (c.id !== id) return c;
        return { ...c, ...updates, x: finalX ?? c.x, y: finalY ?? c.y };
      }));
    }
  }, [setComponentsDirect, snapToGrid, snapValue, detectSmartGuides, selectedIds, components]);

  // Push undo state only when drag stops
  const commitDrag = useCallback(() => {
    setSmartGuides([]); // Clear guide lines
    dragStartRef.current = null; // Clear multi-drag tracking
    // Force push current state to undo history
    setComponentsDirect(prev => {
      pushComponents(prev);
      return prev;
    });
  }, [pushComponents, setComponentsDirect]);

  const deleteComponent = useCallback((id) => {
    pushComponents(components.filter(c => c.id !== id));
    setSelectedIds(prev => prev.filter(sid => sid !== id));
  }, [components, pushComponents]);

  // Delete all selected components
  const deleteSelected = useCallback(() => {
    if (selectedIds.length === 0) return;
    pushComponents(components.filter(c => !selectedIds.includes(c.id)));
    setSelectedIds([]);
  }, [selectedIds, components, pushComponents]);

  const duplicateComponent = useCallback((id) => {
    const comp = components.find(c => c.id === id);
    if (!comp) return;
    const newId = `comp-${++idCounter}-${Date.now()}`;
    const clone = { ...comp, id: newId, x: comp.x + 20, y: comp.y + 20 };
    pushComponents([...components, clone]);
    setSelectedIds([newId]);
  }, [components, pushComponents]);

  // Duplicate all selected components
  const duplicateSelected = useCallback(() => {
    if (selectedIds.length === 0) return;
    const newComps = [];
    const newIds = [];
    for (const sid of selectedIds) {
      const comp = components.find(c => c.id === sid);
      if (!comp) continue;
      const newId = `comp-${++idCounter}-${Date.now()}-${Math.random().toString(36).slice(2,6)}`;
      newComps.push({ ...comp, id: newId, x: comp.x + 20, y: comp.y + 20 });
      newIds.push(newId);
    }
    pushComponents([...components, ...newComps]);
    setSelectedIds(newIds);
  }, [selectedIds, components, pushComponents]);

  // Select component — supports shift-click for multi-select
  const selectComponent = useCallback((id, shiftKey) => {
    if (shiftKey) {
      setSelectedIds(prev =>
        prev.includes(id) ? prev.filter(sid => sid !== id) : [...prev, id]
      );
    } else {
      setSelectedIds([id]);
    }
  }, []);
  const deselectAll = useCallback(() => setSelectedIds([]), []);

  // ── Alignment & distribution tools ──────────────────────────────────────
  const alignComponents = useCallback((axis) => {
    if (selectedIds.length < 2) return;
    const sel = components.filter(c => selectedIds.includes(c.id));
    let updates;
    switch (axis) {
      case 'left': { const minX = Math.min(...sel.map(c => c.x)); updates = sel.map(c => ({ ...c, x: minX })); break; }
      case 'right': { const maxR = Math.max(...sel.map(c => c.x + c.width)); updates = sel.map(c => ({ ...c, x: maxR - c.width })); break; }
      case 'top': { const minY = Math.min(...sel.map(c => c.y)); updates = sel.map(c => ({ ...c, y: minY })); break; }
      case 'bottom': { const maxB = Math.max(...sel.map(c => c.y + c.height)); updates = sel.map(c => ({ ...c, y: maxB - c.height })); break; }
      case 'centerH': { const avgX = sel.reduce((s, c) => s + c.x + c.width / 2, 0) / sel.length; updates = sel.map(c => ({ ...c, x: Math.round(avgX - c.width / 2) })); break; }
      case 'centerV': { const avgY = sel.reduce((s, c) => s + c.y + c.height / 2, 0) / sel.length; updates = sel.map(c => ({ ...c, y: Math.round(avgY - c.height / 2) })); break; }
      case 'distributeH': {
        const sorted = [...sel].sort((a, b) => a.x - b.x);
        const minX = sorted[0].x;
        const maxR = sorted[sorted.length - 1].x + sorted[sorted.length - 1].width;
        const totalW = sorted.reduce((s, c) => s + c.width, 0);
        const gap = (maxR - minX - totalW) / Math.max(1, sorted.length - 1);
        let cx = minX;
        updates = sorted.map(c => { const u = { ...c, x: Math.round(cx) }; cx += c.width + gap; return u; });
        break;
      }
      case 'distributeV': {
        const sorted = [...sel].sort((a, b) => a.y - b.y);
        const minY = sorted[0].y;
        const maxB = sorted[sorted.length - 1].y + sorted[sorted.length - 1].height;
        const totalH = sorted.reduce((s, c) => s + c.height, 0);
        const gap = (maxB - minY - totalH) / Math.max(1, sorted.length - 1);
        let cy = minY;
        updates = sorted.map(c => { const u = { ...c, y: Math.round(cy) }; cy += c.height + gap; return u; });
        break;
      }
      default: return;
    }
    const updateMap = {};
    for (const u of updates) updateMap[u.id] = u;
    pushComponents(components.map(c => updateMap[c.id] || c));
  }, [selectedIds, components, pushComponents]);

  // ── JSON Export/Import ──────────────────────────────────────────────────
  const handleExportJSON = useCallback(() => {
    const data = { pluginConfig, components, dspConfig, presets };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${pluginConfig.name || 'plugin'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [pluginConfig, components, dspConfig, presets]);

  const handleImportJSON = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const data = JSON.parse(ev.target.result);
          if (data.pluginConfig) setPluginConfig(prev => ({ ...prev, ...data.pluginConfig }));
          if (data.components) pushComponents(data.components);
          if (data.dspConfig) setDspConfig(data.dspConfig);
          if (data.presets) setPresets(data.presets);
          showToast('Project imported');
        } catch (err) {
          showToast('Import failed: ' + err.message, null, 'error');
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }, [pushComponents, showToast]);

  // ── Delete project from picker ─────────────────────────────────────────
  const handleDeleteProjectFromPicker = useCallback(async (pid) => {
    try {
      await deleteProject(pid);
      setMyProjects(prev => prev.filter(p => (p.id || p.slug) !== pid));
      if (pid === projectId) {
        // Deleted current project — reset to new
        setProjectId(null);
        setSearchParams({}, { replace: true });
        setPluginConfig({ name: 'My Plugin', width: 600, height: 400, bgColor: '#1a1a2e', titleBarColor: '#2d2d4e', bgImage: '' });
        pushComponents([]);
        setDspConfig(null);
        showToast('Project deleted');
      } else {
        showToast('Project deleted');
      }
    } catch (err) {
      showToast('Delete failed: ' + (err.message || 'Unknown error'), null, 'error');
    }
    setDeleteConfirmId(null);
  }, [projectId, pushComponents, setSearchParams, showToast]);

  // ── Fit-to-view zoom ──────────────────────────────────────────────────
  const fitToView = useCallback(() => {
    const canvasBody = document.querySelector(`.${styles.canvasBody}`);
    if (!canvasBody) { setCanvasZoom(1); return; }
    const viewW = canvasBody.clientWidth - 40; // padding
    const viewH = canvasBody.clientHeight - 80;
    const fitZoom = Math.min(viewW / (pluginConfig.width || 600), viewH / (pluginConfig.height || 400), 3);
    setCanvasZoom(Math.max(0.25, Math.round(fitZoom * 100) / 100));
  }, [pluginConfig.width, pluginConfig.height]);

  // Pick the best DALL-E 3 size for a given aspect ratio
  const pickDalleSize = useCallback((w, h) => {
    const ratio = w / h;
    // DALL-E 3 supports: 1024x1024, 1792x1024 (landscape), 1024x1792 (portrait)
    if (ratio > 1.3) return '1792x1024';    // landscape plugin
    if (ratio < 0.77) return '1024x1792';   // portrait plugin
    return '1024x1024';                      // roughly square
  }, []);

  // Pick Flux aspect ratio from canvas dimensions
  const pickFluxAspectRatio = useCallback((w, h) => {
    const ratio = w / h;
    if (ratio > 1.9) return '21:9';
    if (ratio > 1.5) return '16:9';
    if (ratio > 1.2) return '4:3';
    if (ratio > 0.85) return '1:1';
    if (ratio > 0.65) return '3:4';
    return '9:16';
  }, []);

  // Resolve an image field that could be a URL string or {generate:...}/{search:...}
  // isBackground: if true, uses Flux (better prompt following) with fallback to DALL-E
  const resolveImageField = useCallback(async (field, isBackground = false) => {
    if (!field) return '';
    if (typeof field === 'string') return field;
    if (typeof field === 'object') {
      try {
        if (field.generate) {
          // Use Flux for backgrounds (much better prompt following than DALL-E)
          if (isBackground) {
            try {
              const aspectRatio = pickFluxAspectRatio(pluginConfig.width, pluginConfig.height);
              console.log('[PluginCreator] Generating background with Flux:', field.generate.slice(0, 80), '| aspect:', aspectRatio);
              const result = await generateFluxImage({ prompt: field.generate, aspect_ratio: aspectRatio });
              if (result.url) return result.url;
            } catch (fluxErr) {
              console.warn('[PluginCreator] Flux failed, falling back to DALL-E:', fluxErr.message);
            }
          }
          // Fallback: DALL-E for non-backgrounds or if Flux fails
          const size = isBackground ? pickDalleSize(pluginConfig.width, pluginConfig.height) : '1024x1024';
          const result = await generateImage({ prompt: field.generate, size });
          return result.url || '';
        }
        if (field.search) {
          const result = await searchImages({ query: field.search, per_page: 1 });
          return result.results?.[0]?.url || '';
        }
      } catch (err) {
        console.error('Image resolution failed:', err);
        return '';
      }
    }
    return '';
  }, [pickDalleSize, pickFluxAspectRatio, pluginConfig.width, pluginConfig.height]);

  // Apply AI-generated layout (with auto image generation/search)
  const applyLayout = useCallback(async (layout) => {
    // Apply plugin config immediately (minus bgImage which may need async resolution)
    if (layout.pluginConfig) {
      const configUpdate = { ...layout.pluginConfig };
      const bgImageField = configUpdate.bgImage;
      delete configUpdate.bgImage;
      setPluginConfig(prev => ({ ...prev, ...configUpdate }));

      // Resolve bgImage asynchronously if it's a generate/search command
      if (bgImageField) {
        if (typeof bgImageField === 'string') {
          setPluginConfig(prev => ({ ...prev, bgImage: bgImageField }));
        } else if (typeof bgImageField === 'object') {
          resolveImageField(bgImageField, true).then(url => {
            if (url) setPluginConfig(prev => ({ ...prev, bgImage: url }));
          });
        }
      }
    }

    if (layout.components && Array.isArray(layout.components)) {
      const mode = layout.mode || 'replace';

      // Warn before destructive replace when components exist
      // Skip confirm for auto-applied layouts (from MasterChat auto-apply)
      if (mode === 'replace' && components.length > 0 && !layout.skipConfirm) {
        const proceed = window.confirm(`This will replace all ${components.length} existing components. Continue?`);
        if (!proceed) return;
      }

      const newComps = layout.components.map((c) => {
        const defaults = COMPONENT_DEFAULTS[c.type] || COMPONENT_DEFAULTS.label;
        const comp = {
          id: `comp-${++idCounter}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          ...defaults,
          ...c,
          opacity: c.opacity ?? 1,
          rotation: c.rotation ?? 0,
          borderRadius: c.borderRadius ?? 0,
          fontSize: c.fontSize ?? 13,
          zIndex: c.zIndex ?? 1,
        };
        // Resolve svgStyle → generate SVG from parameterized library
        if (comp.svgStyle && !comp.svg && !comp.sprite) {
          const uid = comp.id.replace(/[^a-zA-Z0-9]/g, '').slice(0, 12);
          const params = {
            width: comp.width, height: comp.height,
            bodyColor: comp.bodyColor || comp.color || '#333',
            indicatorColor: comp.indicatorColor || comp.color || '#fff',
            accentColor: comp.accentColor || comp.color || '#888',
            uid, label: comp.label || '',
          };
          if (comp.type === 'knob') comp.svg = generateKnobSVG(comp.svgStyle, params);
          else if (comp.type === 'slider') comp.svg = generateSliderSVG(comp.svgStyle, params);
          else if (comp.type === 'button') comp.svg = generateButtonSVG(comp.svgStyle, params);
        }
        // Normalize image field — if it's an object, clear it now and resolve async
        if (comp.image && typeof comp.image === 'object') {
          const imageCmd = comp.image;
          comp.image = ''; // placeholder while loading
          const compId = comp.id;
          setGeneratingImages(prev => new Set([...prev, compId]));
          resolveImageField(imageCmd).then(url => {
            setGeneratingImages(prev => { const s = new Set(prev); s.delete(compId); return s; });
            if (url) {
              setComponentsDirect(prev => prev.map(cc =>
                cc.id === compId ? { ...cc, image: url } : cc
              ));
            }
          });
        }
        // Resolve sprite field async (same pattern as image)
        if (comp.sprite && typeof comp.sprite === 'object') {
          const spriteCmd = comp.sprite;
          comp.sprite = ''; // placeholder while loading
          const compId = comp.id;
          setGeneratingImages(prev => new Set([...prev, compId]));
          resolveImageField(spriteCmd).then(url => {
            setGeneratingImages(prev => { const s = new Set(prev); s.delete(compId); return s; });
            if (url) {
              setComponentsDirect(prev => prev.map(cc =>
                cc.id === compId ? { ...cc, sprite: url } : cc
              ));
            }
          });
        }
        // Resolve texture fields and inject into SVG
        if (comp.textures && comp.svg && typeof comp.textures === 'object') {
          const textureEntries = Object.entries(comp.textures);
          const compId = comp.id;
          Promise.all(textureEntries.map(async ([regionId, texCmd]) => {
            if (typeof texCmd === 'string') return [regionId, texCmd];
            const url = await resolveImageField(texCmd);
            return [regionId, url];
          })).then(resolved => {
            setComponentsDirect(prev => prev.map(cc => {
              if (cc.id !== compId || !cc.svg) return cc;
              let updatedSvg = cc.svg;
              for (const [regionId, url] of resolved) {
                if (url) updatedSvg = injectTexturePattern(updatedSvg, regionId, url);
              }
              return { ...cc, svg: updatedSvg };
            }));
          });
        }
        return comp;
      });
      if (mode === 'merge') {
        pushComponents([...components, ...newComps]);
      } else if (mode === 'patch') {
        // Match incoming to existing by label+type, merge properties, keep unmatched
        const patched = components.map(existing => {
          const match = newComps.find(n =>
            n.type === existing.type &&
            (n.label || '').toLowerCase() === (existing.label || '').toLowerCase()
          );
          if (match) {
            // Merge: incoming overrides position/size, existing keeps id + unspecified props
            return { ...existing, ...match, id: existing.id };
          }
          return existing;
        });
        // Add incoming components that didn't match any existing
        const added = newComps.filter(n =>
          !components.some(e =>
            e.type === n.type &&
            (e.label || '').toLowerCase() === (n.label || '').toLowerCase()
          )
        );
        pushComponents([...patched, ...added]);
      } else {
        pushComponents(newComps);
      }
      setSelectedIds([]);
    }
  }, [components, pushComponents, resolveImageField, setComponentsDirect]);

  // Open image browser targeting background or a specific component
  const openImageBrowser = useCallback((target) => {
    setImageBrowserTarget(target);
    setShowImageBrowser(true);
  }, []);

  const handleImageSelect = useCallback((url) => {
    if (imageBrowserTarget === 'background') {
      setPluginConfig(prev => ({ ...prev, bgImage: url }));
    } else if (imageBrowserTarget === 'new-component') {
      // Add as image component
      const id = `comp-${++idCounter}-${Date.now()}`;
      pushComponents([...components, {
        id,
        type: 'image',
        x: Math.round(pluginConfig.width / 2 - 40),
        y: Math.round(pluginConfig.height / 2 - 40),
        ...COMPONENT_DEFAULTS.image,
        image: url,
        opacity: 1, rotation: 0, borderRadius: 0, fontSize: 13,
        zIndex: components.length + 1,
      }]);
      setSelectedIds([id]);
    } else if (imageBrowserTarget) {
      // Apply to specific component
      updateComponent(imageBrowserTarget, { image: url });
    }
    setShowImageBrowser(false);
  }, [imageBrowserTarget, components, pushComponents, pluginConfig, updateComponent]);

  // Context menu handler
  const handleContextMenu = useCallback((e, componentId) => {
    e.preventDefault();
    e.stopPropagation();
    if (editorMode !== 'edit') return;

    const items = [];
    if (componentId) {
      // Select the component on right-click (add to selection if not already selected)
      if (!selectedIds.includes(componentId)) {
        setSelectedIds([componentId]);
      }
      const comp = components.find(c => c.id === componentId);
      const maxZ = Math.max(...components.map(c => c.zIndex || 0), 0);
      const minZ = Math.min(...components.map(c => c.zIndex || 0), 0);

      if (selectedIds.length > 1 && selectedIds.includes(componentId)) {
        // Multi-select context menu
        items.push(
          { label: `Duplicate ${selectedIds.length} items`, icon: 'fa-copy', shortcut: '\u2318D', action: () => duplicateSelected() },
          { label: `Delete ${selectedIds.length} items`, icon: 'fa-trash', shortcut: 'Del', action: () => deleteSelected() },
        );
      } else {
        items.push(
          { label: 'Duplicate', icon: 'fa-copy', shortcut: '\u2318D', action: () => duplicateComponent(componentId) },
          { label: 'Delete', icon: 'fa-trash', shortcut: 'Del', action: () => deleteComponent(componentId) },
          { separator: true },
          { label: 'Bring to Front', icon: 'fa-arrow-up', action: () => updateComponent(componentId, { zIndex: maxZ + 1 }), disabled: (comp?.zIndex || 0) >= maxZ },
          { label: 'Send to Back', icon: 'fa-arrow-down', action: () => updateComponent(componentId, { zIndex: minZ - 1 }), disabled: (comp?.zIndex || 0) <= minZ },
        );
        if (comp?.type === 'image' || comp?.type === 'panel') {
          items.push(
            { separator: true },
            { label: 'Set Image...', icon: 'fa-image', action: () => openImageBrowser(componentId) },
          );
        }
      }
    } else {
      // Canvas right-click
      items.push(
        { label: 'Select All', icon: 'fa-object-group', shortcut: '\u2318A', action: () => setSelectedIds(components.map(c => c.id)) },
        { separator: true },
        { label: 'Add Knob', icon: 'fa-circle-dot', action: () => addComponent('knob') },
        { label: 'Add Slider', icon: 'fa-sliders', action: () => addComponent('slider') },
        { label: 'Add Button', icon: 'fa-square', action: () => addComponent('button') },
        { label: 'Add Label', icon: 'fa-font', action: () => addComponent('label') },
        { label: 'Add Panel', icon: 'fa-square-full', action: () => addComponent('panel') },
      );
    }
    setContextMenu({ x: e.clientX, y: e.clientY, items });
  }, [editorMode, components, selectedIds, duplicateComponent, duplicateSelected, deleteComponent, deleteSelected, updateComponent, openImageBrowser, addComponent]);

  // Apply theme — recolors plugin config + all components
  const handleApplyTheme = useCallback((theme) => {
    const { configUpdate, recoloredComponents } = applyTheme(theme, pluginConfig, components);
    setPluginConfig(prev => ({ ...prev, ...configUpdate }));
    pushComponents(recoloredComponents);
    setActiveTheme(theme);
  }, [pluginConfig, components, pushComponents]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!unlocked) return;
      // Don't trigger shortcuts when typing in inputs
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        redo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault();
        setSelectedIds(components.map(c => c.id));
      } else if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSave(false);
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        e.preventDefault();
        if (selectedIds.length > 0) {
          clipboardRef.current = components.filter(c => selectedIds.includes(c.id)).map(c => ({ ...c }));
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
        e.preventDefault();
        if (clipboardRef.current.length > 0) {
          const pasted = clipboardRef.current.map(c => ({
            ...c,
            id: crypto.randomUUID(),
            x: c.x + 20,
            y: c.y + 20,
          }));
          pushComponents([...components, ...pasted]);
          setSelectedIds(pasted.map(c => c.id));
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'x') {
        e.preventDefault();
        if (selectedIds.length > 0) {
          clipboardRef.current = components.filter(c => selectedIds.includes(c.id)).map(c => ({ ...c }));
          pushComponents(components.filter(c => !selectedIds.includes(c.id)));
          setSelectedIds([]);
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        if (selectedIds.length > 1) duplicateSelected();
        else if (selectedIds.length === 1) duplicateComponent(selectedIds[0]);
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedIds.length > 0) deleteSelected();
      } else if (e.key === 'Escape') {
        setContextMenu(null);
        setSelectedIds([]);
      } else if (e.key === 'Tab') {
        e.preventDefault();
        if (components.length === 0) return;
        const currentIdx = selectedIds.length === 1 ? components.findIndex(c => c.id === selectedIds[0]) : -1;
        const delta = e.shiftKey ? -1 : 1;
        const nextIdx = (currentIdx + delta + components.length) % components.length;
        setSelectedIds([components[nextIdx].id]);
      }
      // 1/2/3 — switch editor modes
      else if (e.key === '1') {
        setEditorMode('edit');
      } else if (e.key === '2') {
        setEditorMode('test');
      } else if (e.key === '3') {
        setEditorMode('structure');
      }
      // Arrow key nudging — move all selected
      else if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key) && selectedIds.length > 0) {
        e.preventDefault();
        const step = e.shiftKey ? 10 : (snapToGrid ? gridSize : 1);
        const delta = { x: 0, y: 0 };
        if (e.key === 'ArrowUp') delta.y = -step;
        if (e.key === 'ArrowDown') delta.y = step;
        if (e.key === 'ArrowLeft') delta.x = -step;
        if (e.key === 'ArrowRight') delta.x = step;
        const updated = components.map(c => {
          if (!selectedIds.includes(c.id)) return c;
          return { ...c, x: c.x + delta.x, y: c.y + delta.y };
        });
        pushComponents(updated);
      }
      // ? key — toggle shortcuts overlay
      else if (e.key === '?') {
        setShowShortcuts(s => !s);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [unlocked, selectedIds, components, undo, redo, duplicateComponent, duplicateSelected, deleteComponent, deleteSelected, updateComponent, pushComponents, handleSave]);

  // ── Audio Engine ──────────────────────────────────────────────────────
  const engineRef = useRef(null);
  const [engineState, setEngineState] = useState(null); // state copy to trigger re-renders

  // Create/destroy engine when entering/leaving test mode
  useEffect(() => {
    if (editorMode === 'test') {
      const engine = new WebAudioDSPEngine(dspConfig || { parameters: [], dspChain: [] });
      engineRef.current = engine;
      setEngineState(engine); // trigger re-render so children get the engine

      // Pre-build graph so params can be applied before first note
      // (AudioContext will be suspended until first user gesture — that's fine)
      engine._buildGraph();

      // Initialize knob positions from DSP parameter defaults, then sync to engine
      if (dspConfig?.parameters) {
        const inits = {};
        for (const param of dspConfig.parameters) {
          const compId = paramMapping[param.id];
          if (compId && paramValues[compId] == null) {
            // Normalize default value to 0-1 range
            const min = param.min ?? 0, max = param.max ?? 1;
            const def = param.default ?? ((min + max) / 2);
            const norm = max !== min ? (def - min) / (max - min) : 0.5;
            inits[compId] = Math.max(0, Math.min(1, norm));
          }
        }
        if (Object.keys(inits).length > 0) {
          setParamValues(prev => ({ ...inits, ...prev }));
        }
        for (const param of dspConfig.parameters) {
          const compId = paramMapping[param.id];
          if (compId) {
            const val = inits[compId] ?? paramValues[compId] ?? 0.5;
            engine.setParameter(param.id, val);
          }
        }
      }

      return () => {
        engine.dispose();
        engineRef.current = null;
        setEngineState(null);
      };
    } else {
      if (engineRef.current) {
        engineRef.current.dispose();
        engineRef.current = null;
        setEngineState(null);
      }
    }
  }, [editorMode, dspConfig]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleParamChange = useCallback((compId, value) => {
    setParamValues(prev => ({ ...prev, [compId]: value }));
    // Forward to audio engine if mapped
    const paramId = reverseMapping[compId];
    if (paramId && engineRef.current) {
      engineRef.current.setParameter(paramId, value);
    }
  }, [reverseMapping]);

  const selectedComponent = selectedIds.length === 1 ? components.find(c => c.id === selectedIds[0]) : null;

  if (!unlocked) {
    const handleUnlock = (e) => {
      e.preventDefault();
      if (passInput === ACCESS_KEY) {
        sessionStorage.setItem('pc_auth', '1');
        setUnlocked(true);
      } else {
        setPassError(true);
      }
    };
    return (
      <div className={styles.creator} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 32, maxWidth: 480 }}>
          {/* What I'm Building */}
          <div style={{ textAlign: 'center' }}>
            <i className="fa-solid fa-puzzle-piece" style={{ fontSize: 36, color: 'rgba(186,156,255,0.6)', marginBottom: 16, display: 'block' }} />
            <h2 style={{ margin: '0 0 16px', fontSize: 24, fontWeight: 700 }}>What I'm Building</h2>
            <p style={{ margin: '0 0 8px', fontSize: 15, color: 'rgba(255,255,255,0.7)', lineHeight: 1.6 }}>
              A visual plugin creator that lets you design, prototype, and compile real VST3/AU audio plugins — entirely in your browser.
            </p>
            <p style={{ margin: '0 0 8px', fontSize: 15, color: 'rgba(255,255,255,0.7)', lineHeight: 1.6 }}>
              Built for musicians, producers, and audio developers who want to make custom effects and instruments without writing C++ from scratch.
            </p>
            <p style={{ margin: '0 0 0', fontSize: 15, color: 'rgba(255,255,255,0.7)', lineHeight: 1.6 }}>
              It eliminates the steep learning curve of audio plugin development — describe what you want, design the UI, and get a compiled plugin in minutes.
            </p>
          </div>

          {/* CTA */}
          <a
            href="mailto:arlo@doseedo.com?subject=Plugin%20Creator%20—%20Interested"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 10,
              padding: '12px 28px', borderRadius: 10, border: 'none',
              background: 'linear-gradient(135deg, #667eea, #764ba2)', color: '#fff',
              fontSize: 15, fontWeight: 600, cursor: 'pointer', textDecoration: 'none',
            }}
          >
            <i className="fa-solid fa-envelope" />
            Get in Touch
          </a>

          {/* Divider */}
          <div style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 12 }}>
            <hr style={{ flex: 1, border: 'none', borderTop: '1px solid rgba(255,255,255,0.08)' }} />
            <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.25)', textTransform: 'uppercase', letterSpacing: 1 }}>Early Access</span>
            <hr style={{ flex: 1, border: 'none', borderTop: '1px solid rgba(255,255,255,0.08)' }} />
          </div>

          {/* Password form */}
          <form onSubmit={handleUnlock} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, width: '100%', maxWidth: 320 }}>
            <i className="fa-solid fa-lock" style={{ fontSize: 20, color: 'rgba(186,156,255,0.4)' }} />
            <p style={{ margin: 0, fontSize: 13, color: 'rgba(255,255,255,0.4)', textAlign: 'center' }}>Have a password? Enter it below to try the creator.</p>
            <input
              type="password"
              value={passInput}
              onChange={(e) => { setPassInput(e.target.value); setPassError(false); }}
              placeholder="Password"
              autoFocus
              style={{
                width: '100%', padding: '12px 16px', fontSize: 14, borderRadius: 10,
                border: `1px solid ${passError ? 'rgba(255,100,100,0.5)' : 'rgba(186,156,255,0.3)'}`,
                background: 'rgba(255,255,255,0.06)', color: '#fff', outline: 'none',
              }}
            />
            {passError && <p style={{ margin: 0, fontSize: 13, color: 'rgba(255,100,100,0.8)' }}>Incorrect password</p>}
            <button type="submit" style={{
              width: '100%', padding: '10px 0', borderRadius: 10, border: 'none',
              background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.7)',
              fontSize: 14, fontWeight: 600, cursor: 'pointer',
            }}>
              Enter
            </button>
            <button type="button" onClick={() => navigate('/plugins')} style={{
              background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)',
              fontSize: 13, cursor: 'pointer',
            }}>
              Back to Plugins
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <PluginContext.Provider value={{ paramController, engineRef }}>
    <div className={styles.creator}>
      <div className={styles.topBar}>
        <button className={styles.backBtn} onClick={() => navigate('/plugins')}>
          <i className="fa-solid fa-arrow-left" /> Back
        </button>
        <div style={{ position: 'relative' }}>
          <button
            className={styles.backBtn}
            onClick={async () => {
              if (!showProjectPicker) {
                try {
                  const projects = await listMyProjects();
                  setMyProjects(Array.isArray(projects) ? projects : projects.projects || []);
                } catch { setMyProjects([]); }
              }
              setShowProjectPicker(!showProjectPicker);
            }}
            title="Switch project"
            style={{ marginLeft: 4, fontSize: 12 }}
          >
            <i className="fa-solid fa-folder-open" style={{ marginRight: 4 }} />
            {pluginConfig.name || 'Untitled'}
            <i className={`fa-solid fa-chevron-${showProjectPicker ? 'up' : 'down'}`} style={{ marginLeft: 4, fontSize: 9 }} />
          </button>
          {showProjectPicker && (
            <>
            <div style={{ position: 'fixed', inset: 0, zIndex: 199 }} onClick={() => setShowProjectPicker(false)} />
            <div style={{
              position: 'absolute', top: '100%', left: 0, zIndex: 200, marginTop: 4,
              background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10,
              minWidth: 240, maxHeight: 320, overflow: 'auto', boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
              padding: 6,
            }}>
              <div style={{ padding: '4px 10px', fontSize: 10, color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: 1 }}>
                My Projects
              </div>
              {myProjects.length === 0 && (
                <div style={{ padding: '12px 10px', fontSize: 12, color: 'rgba(255,255,255,0.3)' }}>No saved projects</div>
              )}
              {myProjects.map(p => {
                const pid = p.id || p.slug;
                return (
                  <div key={pid} style={{ position: 'relative' }}>
                    {deleteConfirmId === pid ? (
                      <div style={{
                        display: 'flex', alignItems: 'center', gap: 6, padding: '8px 10px',
                        background: 'rgba(244,67,54,0.1)', borderRadius: 6, fontSize: 12,
                      }}>
                        <i className="fa-solid fa-triangle-exclamation" style={{ color: '#f44336' }} />
                        <span style={{ color: 'rgba(255,255,255,0.7)', flex: 1 }}>Delete?</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDeleteProjectFromPicker(pid); }}
                          style={{ background: '#f44336', border: 'none', borderRadius: 4, color: '#fff', cursor: 'pointer', padding: '3px 8px', fontSize: 11 }}
                        >
                          Delete
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(null); }}
                          style={{ background: 'rgba(255,255,255,0.1)', border: 'none', borderRadius: 4, color: '#fff', cursor: 'pointer', padding: '3px 8px', fontSize: 11 }}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
                        <button
                          onClick={() => {
                            setShowProjectPicker(false);
                            navigate(`/plugins/create?project=${pid}`);
                          }}
                          style={{
                            display: 'flex', alignItems: 'center', gap: 8, flex: 1, padding: '8px 10px',
                            background: (pid === projectId) ? 'rgba(102,126,234,0.15)' : 'transparent',
                            border: 'none', borderRadius: '6px 0 0 6px', cursor: 'pointer', color: '#fff', fontSize: 13, textAlign: 'left',
                          }}
                        >
                          {p.thumbnail_url ? (
                            <img src={p.thumbnail_url} alt="" style={{ width: 32, height: 24, borderRadius: 4, objectFit: 'cover' }} />
                          ) : (
                            <div style={{ width: 32, height: 24, borderRadius: 4, background: 'rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                              <i className="fa-solid fa-puzzle-piece" style={{ fontSize: 10, color: 'rgba(255,255,255,0.2)' }} />
                            </div>
                          )}
                          <div style={{ flex: 1, overflow: 'hidden' }}>
                            <div style={{ fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {p.name || 'Untitled'}
                            </div>
                            {p.updated_at && (
                              <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)' }}>
                                {new Date(p.updated_at).toLocaleDateString()}
                              </div>
                            )}
                          </div>
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(pid); }}
                          title="Delete project"
                          style={{
                            background: 'none', border: 'none', cursor: 'pointer', padding: '8px 8px',
                            color: 'rgba(255,255,255,0.2)', fontSize: 11, borderRadius: '0 6px 6px 0',
                          }}
                          onMouseEnter={e => e.currentTarget.style.color = '#f44336'}
                          onMouseLeave={e => e.currentTarget.style.color = 'rgba(255,255,255,0.2)'}
                        >
                          <i className="fa-solid fa-trash" />
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
              <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', marginTop: 4, paddingTop: 4 }}>
                <button
                  onClick={() => { setShowProjectPicker(false); resetProject(); navigate('/plugins/create'); }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '8px 10px',
                    background: 'transparent', border: 'none', borderRadius: 6, cursor: 'pointer',
                    color: '#667eea', fontSize: 13,
                  }}
                >
                  <i className="fa-solid fa-plus" /> New Project
                </button>
              </div>
            </div>
            </>
          )}
        </div>
        <div className={styles.topBarActions}>
          <div className={styles.modeToggle}>
            <button
              className={`${styles.modeBtn} ${editorMode === 'edit' ? styles.modeBtnActive : ''}`}
              onClick={() => setEditorMode('edit')}
              title="Edit mode — drag, resize, select components"
            >
              <i className="fa-solid fa-pen" /> Edit
            </button>
            <button
              className={`${styles.modeBtn} ${editorMode === 'test' ? styles.modeBtnActive : ''}`}
              onClick={() => { setEditorMode('test'); setSelectedIds([]); }}
              title="Test mode — interact with knobs, sliders, buttons"
            >
              <i className="fa-solid fa-play" /> Test
            </button>
            <button
              className={`${styles.modeBtn} ${editorMode === 'structure' ? styles.modeBtnActive : ''}`}
              onClick={() => { setEditorMode('structure'); setSelectedIds([]); }}
              title="Structure mode — view and resize sections"
            >
              <i className="fa-solid fa-sitemap" /> Structure
            </button>
          </div>
          <div className={styles.topBarDivider} />
          <button
            className={`${styles.topBarBtn} ${snapToGrid ? styles.topBarBtnActive : ''}`}
            onClick={() => setSnapToGrid(prev => !prev)}
            title="Snap to grid"
          >
            <i className="fa-solid fa-border-all" />
          </button>
          {snapToGrid && (
            <select
              className={styles.gridSelect}
              value={gridSize}
              onChange={(e) => setGridSize(parseInt(e.target.value))}
            >
              <option value={10}>10px</option>
              <option value={20}>20px</option>
              <option value={40}>40px</option>
            </select>
          )}
          <button
            className={`${styles.topBarBtn} ${showLayers ? styles.topBarBtnActive : ''}`}
            onClick={() => setShowLayers(prev => !prev)}
            title="Layers panel"
          >
            <i className="fa-solid fa-layer-group" />
          </button>
          <button
            className={styles.topBarBtn}
            onClick={() => openImageBrowser('new-component')}
            title="Image browser"
          >
            <i className="fa-solid fa-image" />
          </button>
          <button className={styles.topBarBtn} onClick={undo} title="Undo (Ctrl+Z)">
            <i className="fa-solid fa-rotate-left" />
          </button>
          <button className={styles.topBarBtn} onClick={redo} title="Redo (Ctrl+Shift+Z)">
            <i className="fa-solid fa-rotate-right" />
          </button>
          {/* Alignment tools (visible when 2+ selected) */}
          {selectedIds.length >= 2 && (
            <>
              <div className={styles.topBarDivider} />
              <button className={styles.topBarBtn} onClick={() => alignComponents('left')} title="Align left">
                <i className="fa-solid fa-align-left" />
              </button>
              <button className={styles.topBarBtn} onClick={() => alignComponents('centerH')} title="Align center horizontal">
                <i className="fa-solid fa-align-center" />
              </button>
              <button className={styles.topBarBtn} onClick={() => alignComponents('right')} title="Align right">
                <i className="fa-solid fa-align-right" />
              </button>
              <button className={styles.topBarBtn} onClick={() => alignComponents('top')} title="Align top" style={{ transform: 'rotate(-90deg)' }}>
                <i className="fa-solid fa-align-left" />
              </button>
              <button className={styles.topBarBtn} onClick={() => alignComponents('centerV')} title="Align center vertical" style={{ transform: 'rotate(-90deg)' }}>
                <i className="fa-solid fa-align-center" />
              </button>
              <button className={styles.topBarBtn} onClick={() => alignComponents('bottom')} title="Align bottom" style={{ transform: 'rotate(-90deg)' }}>
                <i className="fa-solid fa-align-right" />
              </button>
              {selectedIds.length >= 3 && (
                <>
                  <button className={styles.topBarBtn} onClick={() => alignComponents('distributeH')} title="Distribute horizontal spacing">
                    <i className="fa-solid fa-grip-lines-vertical" />
                  </button>
                  <button className={styles.topBarBtn} onClick={() => alignComponents('distributeV')} title="Distribute vertical spacing">
                    <i className="fa-solid fa-grip-lines" />
                  </button>
                </>
              )}
            </>
          )}
          <div className={styles.topBarDivider} />
          {/* Save */}
          <button
            className={`${styles.topBarBtn} ${saveStatus === 'saved' ? styles.topBarBtnActive : ''}`}
            onClick={() => handleSave(false)}
            disabled={saveStatus === 'saving'}
            title={saveStatus === 'saved' ? 'Saved' : saveStatus === 'saving' ? 'Saving...' : isDirty ? 'Unsaved changes — Save project' : 'Save project'}
            style={{ position: 'relative' }}
          >
            <i className={`fa-solid ${saveStatus === 'saving' ? 'fa-spinner fa-spin' : saveStatus === 'saved' ? 'fa-check' : saveStatus === 'error' ? 'fa-exclamation-triangle' : 'fa-floppy-disk'}`} />
            {isDirty && saveStatus !== 'saving' && saveStatus !== 'saved' && (
              <span style={{
                position: 'absolute', top: 4, right: 4, width: 6, height: 6,
                borderRadius: '50%', background: '#f5a623', border: '1px solid rgba(0,0,0,0.3)',
              }} />
            )}
          </button>
          {/* Plugin type badge */}
          {dspConfig && (
            <span
              style={{
                fontSize: 10, padding: '2px 8px', borderRadius: 10, cursor: 'pointer',
                background: dspConfig.pluginType === 'instrument' ? 'rgba(255,152,0,0.2)' : dspConfig.pluginType === 'midi_effect' ? 'rgba(156,39,176,0.2)' : 'rgba(76,175,80,0.2)',
                color: dspConfig.pluginType === 'instrument' ? '#ffb74d' : dspConfig.pluginType === 'midi_effect' ? '#ce93d8' : '#81c784',
              }}
              title="Plugin type (set via chat or DSP editor)"
              onClick={() => {
                const types = ['effect', 'instrument', 'midi_effect'];
                const cur = types.indexOf(dspConfig.pluginType || 'effect');
                const next = types[(cur + 1) % types.length];
                setDspConfig(prev => ({ ...prev, pluginType: next }));
              }}
            >
              <i className={`fa-solid ${dspConfig.pluginType === 'instrument' ? 'fa-piano-keyboard' : dspConfig.pluginType === 'midi_effect' ? 'fa-music' : 'fa-wave-square'}`} style={{ marginRight: 4 }} />
              {dspConfig.pluginType === 'instrument' ? 'Instrument' : dspConfig.pluginType === 'midi_effect' ? 'MIDI FX' : 'Effect'}
            </span>
          )}
          {/* Generate Code */}
          <button
            className={styles.topBarBtn}
            onClick={handleGenerateCode}
            disabled={!dspConfig || generatingCode}
            title={dspConfig ? 'Generate JUCE C++ code' : 'Design a DSP chain first'}
          >
            <i className={`fa-solid ${generatingCode ? 'fa-spinner fa-spin' : 'fa-code'}`} />
          </button>
          {/* Download Plugin */}
          <div style={{ position: 'relative' }}>
            <button
              className={styles.topBarBtn}
              onClick={() => setShowDownloadMenu(prev => !prev)}
              disabled={buildingPlugin}
              title={buildingPlugin ? (buildStage || 'Building...') : buildBlob ? 'Download plugin' : 'Build & download plugin'}
              style={buildBlob ? { color: '#81c784', borderColor: 'rgba(129,199,132,0.3)' } : {}}
            >
              <i className={`fa-solid ${buildingPlugin ? 'fa-spinner fa-spin' : 'fa-download'}`} />
            </button>
            {showDownloadMenu && (
              <div
                style={{
                  position: 'absolute', top: '100%', right: 0, marginTop: 4, width: 180,
                  background: 'rgba(20,20,40,0.95)', border: '1px solid rgba(186,156,255,0.2)',
                  borderRadius: 10, padding: '6px 0', zIndex: 1000, backdropFilter: 'blur(12px)',
                  boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
                }}
                onMouseLeave={() => setShowDownloadMenu(false)}
              >
                <div style={{ padding: '4px 12px 6px', fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.35)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  {buildBlob ? 'Download' : 'Build & Download'}
                </div>
                {[
                  { key: 'vst3', label: 'VST3', icon: 'fa-plug' },
                  { key: 'au', label: 'Audio Unit', icon: 'fa-music' },
                  { key: 'pkg', label: 'macOS Installer', icon: 'fa-box' },
                  { key: 'all', label: 'All Formats (.zip)', icon: 'fa-file-zipper' },
                ].map(opt => (
                  <button
                    key={opt.key}
                    onClick={() => handleDownloadFormat(opt.key)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 8, width: '100%',
                      padding: '8px 14px', background: 'none', border: 'none',
                      color: 'rgba(255,255,255,0.7)', fontSize: 12, cursor: 'pointer',
                      textAlign: 'left', transition: 'background 0.15s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(186,156,255,0.12)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'none'}
                  >
                    <i className={`fa-solid ${opt.icon}`} style={{ width: 16, textAlign: 'center', opacity: 0.6 }} />
                    {opt.label}
                  </button>
                ))}
                {!codePreview?.files && (
                  <div style={{ padding: '6px 14px', fontSize: 10, color: 'rgba(255,255,255,0.3)', borderTop: '1px solid rgba(255,255,255,0.06)', marginTop: 4 }}>
                    Will generate code & build first
                  </div>
                )}
                {buildingPlugin && (
                  <div style={{ padding: '6px 14px', fontSize: 10, color: 'rgba(186,156,255,0.6)', borderTop: '1px solid rgba(255,255,255,0.06)', marginTop: 4 }}>
                    <i className="fa-solid fa-spinner fa-spin" style={{ marginRight: 4 }} />
                    {buildStage || 'Building...'}
                  </div>
                )}
              </div>
            )}
          </div>
          {/* Export PNG */}
          <button
            className={styles.topBarBtn}
            onClick={handleExportPNG}
            title="Export canvas as PNG"
          >
            <i className="fa-solid fa-file-image" />
          </button>
          {/* JSON Export/Import */}
          <button className={styles.topBarBtn} onClick={handleExportJSON} title="Export project as JSON">
            <i className="fa-solid fa-file-export" />
          </button>
          <button className={styles.topBarBtn} onClick={handleImportJSON} title="Import project from JSON">
            <i className="fa-solid fa-file-import" />
          </button>
          {/* Presets */}
          <div style={{ position: 'relative' }}>
            <button
              className={`${styles.topBarBtn} ${showPresetPanel ? styles.topBarBtnActive : ''}`}
              onClick={() => setShowPresetPanel(!showPresetPanel)}
              title="Presets"
            >
              <i className="fa-solid fa-sliders" />
            </button>
            {showPresetPanel && (
              <div style={{
                position: 'absolute', top: '100%', right: 0, marginTop: 4, width: 220,
                background: 'rgba(20,20,40,0.95)', border: '1px solid rgba(186,156,255,0.2)',
                borderRadius: 10, padding: '8px 0', zIndex: 1000, backdropFilter: 'blur(12px)',
                boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
              }}>
                <div style={{ padding: '4px 12px 8px', fontSize: 11, fontWeight: 600, color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Presets
                </div>
                {presets.length === 0 && (
                  <div style={{ padding: '8px 12px', fontSize: 12, color: 'rgba(255,255,255,0.3)' }}>No presets saved</div>
                )}
                {presets.map((p, i) => (
                  <div key={i} style={{
                    display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px',
                    cursor: 'pointer', fontSize: 12, color: i === activePresetIdx ? '#ba9cff' : 'rgba(255,255,255,0.7)',
                    background: i === activePresetIdx ? 'rgba(186,156,255,0.1)' : 'transparent',
                  }}
                    onClick={() => handleLoadPreset(i)}
                  >
                    <i className={`fa-solid ${i === activePresetIdx ? 'fa-circle-check' : 'fa-circle'}`} style={{ fontSize: 9 }} />
                    <span style={{ flex: 1 }}>{p.name}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDeletePreset(i); }}
                      style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.2)', cursor: 'pointer', fontSize: 10, padding: '2px 4px' }}
                      title="Delete preset"
                    >
                      <i className="fa-solid fa-xmark" />
                    </button>
                  </div>
                ))}
                <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', margin: '4px 0' }} />
                {showPresetNameInput ? (
                  <div style={{ display: 'flex', gap: 4, padding: '4px 12px', alignItems: 'center' }}>
                    <input
                      type="text"
                      value={presetNameInput}
                      onChange={(e) => setPresetNameInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSavePreset();
                        if (e.key === 'Escape') { setShowPresetNameInput(false); setPresetNameInput(''); }
                      }}
                      autoFocus
                      placeholder="Preset name..."
                      style={{
                        flex: 1, padding: '4px 8px', fontSize: 11, borderRadius: 4,
                        border: '1px solid rgba(186,156,255,0.3)', background: 'rgba(255,255,255,0.08)',
                        color: '#fff', outline: 'none',
                      }}
                    />
                    <button onClick={handleSavePreset} style={{ background: 'none', border: 'none', color: '#ba9cff', cursor: 'pointer', fontSize: 11, padding: '4px 6px' }}>
                      <i className="fa-solid fa-check" />
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={handleSavePreset}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '6px 12px',
                      background: 'none', border: 'none', color: '#ba9cff', fontSize: 12, cursor: 'pointer', textAlign: 'left',
                    }}
                  >
                    <i className="fa-solid fa-plus" /> Save Current
                  </button>
                )}
              </div>
            )}
          </div>
          {/* Publish */}
          <button
            className={`${styles.topBarBtn} ${isPublished ? styles.topBarBtnActive : ''}`}
            onClick={() => isPublished ? handleUnpublish() : setShowPublishModal(true)}
            title={isPublished ? 'Published (click to unpublish)' : 'Publish to community'}
          >
            <i className={`fa-solid ${isPublished ? 'fa-globe' : 'fa-share-nodes'}`} />
          </button>
        </div>
      </div>

      <div className={styles.creatorLayout}>
        {/* Chat Panel */}
        <div className={styles.chatPanel} style={{ width: chatPanelWidth, flexShrink: 0 }}>
          <div className={styles.chatTabBar}>
            <button
              className={chatPanelTab === 'designer' ? styles.chatTabActive : styles.chatTab}
              onClick={() => setChatPanelTab('designer')}
            >
              <i className="fa-solid fa-bolt" /> Designer
            </button>
            <button
              className={chatPanelTab === 'assets' ? styles.chatTabActive : styles.chatTab}
              onClick={() => setChatPanelTab('assets')}
            >
              <i className="fa-solid fa-shapes" /> Assets
            </button>
          </div>
          <div style={{ display: chatPanelTab === 'designer' ? 'contents' : 'none' }}>
            <MasterChat
              pluginConfig={pluginConfig}
              components={components}
              dspConfig={dspConfig}
              onApplyLayout={applyLayout}
              onApplyDsp={setDspConfig}
              onOpenImageBrowser={openImageBrowser}
              chatHistory={chatHistory}
              onChatHistoryChange={setChatHistory}
              activeTheme={activeTheme}
              onApplyTheme={handleApplyTheme}
            />
          </div>
          {chatPanelTab === 'assets' && (
            <AssetBrowser onAddToCanvas={addComponentToCanvas} />
          )}
        </div>

        {/* Resize Handle */}
        <div
          className={styles.resizeHandle}
          onMouseDown={(e) => {
            e.preventDefault();
            isResizingRef.current = true;
            resizeStartXRef.current = e.clientX;
            resizeStartWidthRef.current = chatPanelWidth;
          }}
        />

        {/* Canvas Panel */}
        <div className={styles.canvasPanel}>
          {/* UI Editor / DSP Editor toggle */}
          <div className={styles.canvasToggle}>
            <button
              className={`${styles.canvasToggleBtn} ${!showDspEditor ? styles.canvasToggleBtnActive : ''}`}
              onClick={() => setShowDspEditor(false)}
            >
              <i className="fa-solid fa-palette" /> UI Editor
            </button>
            <button
              className={`${styles.canvasToggleBtn} ${showDspEditor ? styles.canvasToggleBtnActive : ''}`}
              onClick={() => setShowDspEditor(true)}
            >
              <i className="fa-solid fa-diagram-project" /> DSP Editor
            </button>
          </div>

          {/* Welcome / Template Gallery */}
          {showWelcome && components.length === 0 && !projectParam && !showDspEditor && (
            <div className={styles.welcomeOverlay} onClick={() => { setShowWelcome(false); setWelcomeStep('choose'); }}>
              <div onClick={e => e.stopPropagation()} style={{
                background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 16, padding: welcomeStep === 'templates' ? '20px 24px' : '32px 40px',
                boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
                maxWidth: welcomeStep === 'templates' ? 520 : 360, width: '90%',
                maxHeight: '80vh', overflowY: 'auto',
              }}>
                {welcomeStep === 'choose' ? (
                  <>
                    <div style={{ textAlign: 'center', marginBottom: 24 }}>
                      <i className="fa-solid fa-wand-magic-sparkles" style={{ fontSize: 24, color: 'rgba(186,156,255,0.7)', marginBottom: 10, display: 'block' }} />
                      <h2 style={{ margin: '0 0 6px', fontSize: 18, fontWeight: 700, color: '#fff' }}>New Plugin</h2>
                      <p style={{ margin: 0, fontSize: 12, color: 'rgba(255,255,255,0.4)' }}>How would you like to start?</p>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                      <button
                        onClick={() => { setShowWelcome(false); setWelcomeStep('choose'); }}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 14, padding: '16px 20px',
                          background: 'rgba(102,126,234,0.08)', border: '1px solid rgba(102,126,234,0.2)',
                          borderRadius: 12, color: '#fff', cursor: 'pointer', textAlign: 'left',
                          transition: 'all 0.15s',
                        }}
                        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(102,126,234,0.15)'; e.currentTarget.style.borderColor = 'rgba(102,126,234,0.4)'; }}
                        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(102,126,234,0.08)'; e.currentTarget.style.borderColor = 'rgba(102,126,234,0.2)'; }}
                      >
                        <i className="fa-solid fa-plus" style={{ fontSize: 18, color: '#667eea', width: 28 }} />
                        <div>
                          <div style={{ fontWeight: 600, fontSize: 14 }}>From Scratch</div>
                          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', marginTop: 2 }}>Blank canvas — design with chat or palette</div>
                        </div>
                      </button>
                      <button
                        onClick={() => setWelcomeStep('templates')}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 14, padding: '16px 20px',
                          background: 'rgba(186,156,255,0.08)', border: '1px solid rgba(186,156,255,0.2)',
                          borderRadius: 12, color: '#fff', cursor: 'pointer', textAlign: 'left',
                          transition: 'all 0.15s',
                        }}
                        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(186,156,255,0.15)'; e.currentTarget.style.borderColor = 'rgba(186,156,255,0.4)'; }}
                        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(186,156,255,0.08)'; e.currentTarget.style.borderColor = 'rgba(186,156,255,0.2)'; }}
                      >
                        <i className="fa-solid fa-shapes" style={{ fontSize: 18, color: '#ba9cff', width: 28 }} />
                        <div>
                          <div style={{ fontWeight: 600, fontSize: 14 }}>From Template</div>
                          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', marginTop: 2 }}>Start with a pre-built plugin layout</div>
                        </div>
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <button
                        onClick={() => setWelcomeStep('choose')}
                        style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.5)', cursor: 'pointer', padding: '4px 6px', fontSize: 13 }}
                      >
                        <i className="fa-solid fa-arrow-left" />
                      </button>
                      <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600, color: '#fff' }}>Choose a Template</h3>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 8 }}>
                      {PLUGIN_TEMPLATES.map((t, i) => (
                        <button
                          key={i}
                          className={styles.welcomeCard}
                          onClick={() => {
                            setPluginConfig(prev => ({ ...prev, ...t.pluginConfig }));
                            const compsWithIds = (t.components || []).map(c => ({
                              ...c,
                              id: c.id || `comp-${++idCounter}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
                              zIndex: c.zIndex ?? 1,
                              opacity: c.opacity ?? 1,
                              rotation: c.rotation ?? 0,
                              borderRadius: c.borderRadius ?? 0,
                            }));
                            pushComponents(compsWithIds);
                            if (t.dspConfig) setDspConfig(t.dspConfig);
                            setShowWelcome(false);
                            setWelcomeStep('choose');
                          }}
                        >
                          {t.components?.length > 0 ? (
                            <TemplateThumbnail template={t} />
                          ) : (
                            <i className={`fa-solid ${t.icon}`} style={{ fontSize: 18, color: t.color || '#ba9cff' }} />
                          )}
                          <div style={{ fontWeight: 600, fontSize: 11 }}>{t.name}</div>
                          <div style={{ fontSize: 9, opacity: 0.5, lineHeight: 1.3 }}>{t.description}</div>
                          {t.dspConfig?.pluginType && (
                            <span className={styles.welcomeBadge}>
                              {t.dspConfig.pluginType === 'instrument' ? 'Instrument' : 'Effect'}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Empty canvas hint */}
          {components.length === 0 && !showWelcome && !showDspEditor && (
            <div className={styles.emptyCanvasHint}>
              <i className="fa-solid fa-wand-magic-sparkles" style={{ fontSize: 24, marginBottom: 8, opacity: 0.3 }} />
              <p>Describe your plugin in the chat, or add components from the palette</p>
            </div>
          )}

          {showDspEditor ? (
            <div style={{ flex: 1, minHeight: 0, borderRadius: 12, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.08)', display: 'flex', flexDirection: 'column' }}>
              {/* DSP sub-tabs */}
              <div style={{ display: 'flex', gap: 4, padding: '6px 8px', background: 'rgba(0,0,0,0.3)', borderBottom: '1px solid rgba(255,255,255,0.08)', flexShrink: 0 }}>
                {['graph','modmatrix','adsr','eq','mseg'].map(tab => (
                  <button key={tab} onClick={() => setDspSubTab(tab)} style={{ padding: '3px 10px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 11, fontWeight: 600, background: dspSubTab === tab ? 'rgba(102,126,234,0.8)' : 'rgba(255,255,255,0.07)', color: '#fff' }}>
                    {tab === 'graph' ? 'DSP Graph' : tab === 'modmatrix' ? 'Mod Matrix' : tab === 'adsr' ? 'ADSR' : tab === 'eq' ? 'EQ' : 'MSEG'}
                  </button>
                ))}
              </div>
              <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
                {dspSubTab === 'graph' && <DSPEditor dspConfig={dspConfig} onUpdateDsp={setDspConfig} />}
                {dspSubTab === 'modmatrix' && <ModMatrixWithSliders dspConfig={dspConfig} onUpdateDsp={setDspConfig} />}
                {dspSubTab === 'adsr' && (
                  <ADSRDisplay
                    attack={paramValues.attack ?? 0.01}
                    decay={paramValues.decay ?? 0.3}
                    sustain={paramValues.sustain ?? 0.7}
                    release={paramValues.release ?? 0.5}
                    width={480} height={160}
                    onChange={(key, val) => {
                      handleParamChange(key, val);
                      if (engineRef.current) engineRef.current.setParameter(key, val);
                    }}
                  />
                )}
                {dspSubTab === 'eq' && (
                  <EQCurveDisplay
                    onChange={(band, key, val) => {
                      const paramKey = `eq_band${band}_${key}`;
                      handleParamChange(paramKey, val);
                      if (engineRef.current) engineRef.current.setParameter(paramKey, val);
                    }}
                  />
                )}
                {dspSubTab === 'mseg' && (
                  <MSEGEditor
                    onChange={(points) => {
                      handleParamChange('mseg_points', points);
                    }}
                  />
                )}
              </div>
            </div>
          ) : (
          <>
          {/* Plugin name & config */}
          <div className={styles.configBar}>
            <span className={styles.configLabel}>Name</span>
            <input
              className={styles.configInput}
              value={pluginConfig.name}
              onChange={(e) => setPluginConfig(prev => ({ ...prev, name: e.target.value }))}
            />
            <span className={styles.configLabel}>Size</span>
            <input
              className={styles.configInputSmall}
              type="number"
              value={pluginConfig.width}
              onChange={(e) => setPluginConfig(prev => ({ ...prev, width: Math.max(200, parseInt(e.target.value) || 200) }))}
              min={200}
              max={1600}
            />
            <span style={{ color: 'rgba(255,255,255,0.25)', fontSize: 12 }}>&times;</span>
            <input
              className={styles.configInputSmall}
              type="number"
              value={pluginConfig.height}
              onChange={(e) => setPluginConfig(prev => ({ ...prev, height: Math.max(150, parseInt(e.target.value) || 150) }))}
              min={150}
              max={1200}
            />
            <span className={styles.configLabel}>BG</span>
            <input
              type="color"
              value={pluginConfig.bgColor}
              onChange={(e) => setPluginConfig(prev => ({ ...prev, bgColor: e.target.value }))}
              className={styles.colorPicker}
            />
            <button
              className={styles.bgImageBtn}
              onClick={() => openImageBrowser('background')}
              title="Set background image"
            >
              <i className="fa-solid fa-image" />
            </button>
            {pluginConfig.bgImage && (
              <div className={styles.bgThumbWrap}>
                <div
                  className={styles.bgThumb}
                  style={{ backgroundImage: `url(${pluginConfig.bgImage})` }}
                  title={pluginConfig.bgImage}
                />
                <button
                  className={styles.bgClearBtn}
                  onClick={() => setPluginConfig(prev => ({ ...prev, bgImage: '' }))}
                  title="Clear background image"
                >
                  <i className="fa-solid fa-xmark" />
                </button>
              </div>
            )}
          </div>

          {/* Component Palette (edit mode only) */}
          {editorMode === 'edit' && <ComponentPalette onAddComponent={addComponent} />}

          {/* Zoom controls */}
          <div className={styles.zoomBar}>
            <button onClick={() => setCanvasZoom(z => Math.max(0.25, z - 0.1))} title="Zoom out" className={styles.zoomBtn}>
              <i className="fa-solid fa-minus" />
            </button>
            <span className={styles.zoomLabel} onClick={() => setCanvasZoom(1)} title="Reset to 100%">
              {Math.round(canvasZoom * 100)}%
            </span>
            <button onClick={() => setCanvasZoom(z => Math.min(3, z + 0.1))} title="Zoom in" className={styles.zoomBtn}>
              <i className="fa-solid fa-plus" />
            </button>
            <button onClick={fitToView} title="Fit to view" className={styles.zoomBtn} style={{ marginLeft: 4 }}>
              <i className="fa-solid fa-expand" />
            </button>
          </div>

          {/* Canvas + Property Panel side by side */}
          <div className={styles.canvasWithProps}>
          {/* Plugin Canvas — always renders, StructureView overlays on top */}
          <div style={{ flex: 1, minWidth: 0, overflow: 'auto' }}>
          <div
            style={{ transform: `scale(${canvasZoom})`, transformOrigin: 'top left', display: 'inline-block', touchAction: 'pan-x pan-y' }}
            onWheel={(e) => {
              if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                setCanvasZoom(z => Math.max(0.25, Math.min(3, z - e.deltaY * 0.002)));
              }
            }}
            onTouchStart={(e) => {
              if (e.touches.length === 2) {
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                pinchRef.current = { dist: Math.hypot(dx, dy), zoom: canvasZoom };
              }
            }}
            onTouchMove={(e) => {
              if (e.touches.length === 2 && pinchRef.current) {
                e.preventDefault();
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                const dist = Math.hypot(dx, dy);
                const scale = dist / pinchRef.current.dist;
                setCanvasZoom(Math.max(0.25, Math.min(3, pinchRef.current.zoom * scale)));
              }
            }}
            onTouchEnd={() => { pinchRef.current = null; }}
          >
            <PluginCanvas
              config={pluginConfig}
              components={components}
              selectedIds={editorMode === 'structure' ? [] : selectedIds}
              onSelect={selectComponent}
              onDeselect={deselectAll}
              onUpdateComponent={updateComponentDrag}
              onDragStop={commitDrag}
              snapToGrid={snapToGrid}
              gridSize={gridSize}
              editorMode={editorMode}
              paramValues={paramValues}
              onParamChange={handleParamChange}
              smartGuides={smartGuides}
              onContextMenu={handleContextMenu}
              rubberBand={rubberBand}
              onRubberBandChange={setRubberBand}
              onRubberBandSelect={setSelectedIds}
              reverseParamMap={reverseMapping}
              engine={engineState}
              frameRef={pluginFrameRef}
              generatingImages={generatingImages}
              overlay={editorMode === 'structure' ? (
                <StructureView
                  config={pluginConfig}
                  components={components}
                  selectedIds={selectedIds}
                  onSelect={selectComponent}
                  onDeselect={deselectAll}
                  onUpdateComponents={(updated) => setComponentsDirect(() => updated)}
                  onCommitResize={commitDrag}
                />
              ) : null}
            />
          </div>

          {/* Audio Test Panel (test mode only) */}
          {editorMode === 'test' && (
            <AudioTestPanel
              engine={engineState}
              dspConfig={dspConfig}
              paramValues={paramValues}
              paramMapping={paramMapping}
              components={components}
              onParamChange={handleParamChange}
            />
          )}

          {/* Structure Tree Panel (structure mode) */}
          {editorMode === 'structure' && (() => {
            const { sections, orphans } = buildSectionMap(components);
            return (
              <StructureTreePanel
                sections={sections}
                orphans={orphans}
                selectedIds={selectedIds}
                onSelect={selectComponent}
                pluginConfig={pluginConfig}
              />
            );
          })()}
          </div>

          {/* Property Panel (edit + structure modes) — right side */}
          {(editorMode === 'edit' || editorMode === 'structure') && selectedComponent && (
            <PropertyPanel
              component={selectedComponent}
              onUpdate={(updates) => updateComponent(selectedIds[0], updates)}
              onDelete={() => deleteComponent(selectedIds[0])}
              onDuplicate={() => duplicateComponent(selectedIds[0])}
              onOpenImageBrowser={() => openImageBrowser(selectedIds[0])}
              paramMapping={paramMapping}
              dspConfig={dspConfig}
              onRemapParam={handleRemapParam}
            />
          )}
          </div>
          </>
          )}
        </div>

        {/* Layers Panel */}
        {showLayers && (
          <LayersPanel
            components={components}
            selectedIds={selectedIds}
            onSelect={(id) => selectComponent(id)}
            onUpdateComponent={updateComponent}
            onClose={() => setShowLayers(false)}
          />
        )}
      </div>

      {/* Image Browser Modal */}
      {showImageBrowser && (
        <ImageBrowser
          onSelect={handleImageSelect}
          onClose={() => setShowImageBrowser(false)}
        />
      )}

      {/* Publish Modal */}
      {showPublishModal && (
        <div className={styles.modalOverlay} onClick={() => setShowPublishModal(false)}>
          <div className={styles.modalContent} onClick={e => e.stopPropagation()}>
            <h3 style={{ margin: '0 0 12px', fontSize: 16, fontWeight: 600 }}>
              <i className="fa-solid fa-share-nodes" style={{ marginRight: 8 }} />
              Publish to Community
            </h3>
            <p style={{ margin: '0 0 16px', fontSize: 13, color: 'rgba(255,255,255,0.5)' }}>
              Share your plugin design with the community. Others can view, download, and build your plugin.
            </p>
            <label style={{ fontSize: 12, color: 'rgba(255,255,255,0.4)', marginBottom: 4, display: 'block' }}>Description</label>
            <textarea
              value={publishDesc}
              onChange={e => setPublishDesc(e.target.value)}
              placeholder="Describe your plugin..."
              style={{
                width: '100%', padding: '10px 12px', fontSize: 13, borderRadius: 8, marginBottom: 12,
                border: '1px solid rgba(186,156,255,0.2)', background: 'rgba(255,255,255,0.06)',
                color: '#fff', outline: 'none', resize: 'vertical', minHeight: 60,
              }}
            />
            <label style={{ fontSize: 12, color: 'rgba(255,255,255,0.4)', marginBottom: 4, display: 'block' }}>Tags (comma-separated)</label>
            <input
              value={publishTags}
              onChange={e => setPublishTags(e.target.value)}
              placeholder="delay, reverb, synth, lo-fi..."
              style={{
                width: '100%', padding: '10px 12px', fontSize: 13, borderRadius: 8, marginBottom: 16,
                border: '1px solid rgba(186,156,255,0.2)', background: 'rgba(255,255,255,0.06)',
                color: '#fff', outline: 'none',
              }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowPublishModal(false)}
                style={{
                  padding: '8px 16px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)',
                  background: 'transparent', color: 'rgba(255,255,255,0.6)', fontSize: 13, cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handlePublish}
                disabled={publishLoading}
                style={{
                  padding: '8px 20px', borderRadius: 8, border: 'none',
                  background: 'linear-gradient(135deg, #667eea, #764ba2)', color: '#fff',
                  fontSize: 13, fontWeight: 600, cursor: publishLoading ? 'wait' : 'pointer',
                  opacity: publishLoading ? 0.6 : 1, display: 'flex', alignItems: 'center', gap: 6,
                }}
              >
                {publishLoading ? <><i className="fa-solid fa-spinner fa-spin" /> Publishing...</> : 'Publish'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Context Menu */}
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          items={contextMenu.items}
          onClose={() => setContextMenu(null)}
        />
      )}

      {/* Keyboard Shortcuts Overlay */}
      {showShortcuts && (
        <div className={styles.modalOverlay} onClick={() => setShowShortcuts(false)}>
          <div onClick={e => e.stopPropagation()} style={{
            background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 12, padding: '24px 32px', maxWidth: 420, width: '90%',
            boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3 style={{ margin: 0, color: '#fff', fontSize: 16 }}>Keyboard Shortcuts</h3>
              <button onClick={() => setShowShortcuts(false)} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.5)', cursor: 'pointer', fontSize: 18 }}>&times;</button>
            </div>
            {[
              ['Ctrl+S', 'Save'],
              ['Ctrl+Z', 'Undo'],
              ['Ctrl+Shift+Z', 'Redo'],
              ['Ctrl+C', 'Copy selected'],
              ['Ctrl+V', 'Paste'],
              ['Ctrl+X', 'Cut selected'],
              ['Ctrl+A', 'Select all'],
              ['Ctrl+D', 'Duplicate selected'],
              ['Delete / Backspace', 'Delete selected'],
              ['Escape', 'Deselect all'],
              ['Arrow keys', 'Nudge 1px'],
              ['Shift + Arrow keys', 'Nudge 10px'],
              ['Tab / Shift+Tab', 'Select next/prev component'],
              ['1 / 2 / 3', 'Switch to Edit / Test / Structure mode'],
              ['Ctrl+Wheel', 'Zoom in/out'],
              ['?', 'Toggle this overlay'],
            ].map(([key, desc]) => (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                <kbd style={{ background: 'rgba(255,255,255,0.08)', padding: '2px 8px', borderRadius: 4, fontSize: 12, fontFamily: 'monospace', color: '#ba9cff' }}>{key}</kbd>
                <span style={{ color: 'rgba(255,255,255,0.7)', fontSize: 13 }}>{desc}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Code Preview Modal */}
      {showCodePreview && codePreview && (
        <div className={styles.modalOverlay} onClick={() => setShowCodePreview(false)}>
          <div className={styles.codeModalContent} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>
                <i className="fa-solid fa-code" style={{ marginRight: 8 }} />
                Generated JUCE C++ Code
              </h3>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <button
                  onClick={() => {
                    if (!codePreview?.files) return;
                    const JSZip = window.JSZip;
                    if (!JSZip) {
                      // Inline zip creation using Blob
                      const parts = Object.entries(codePreview.files).map(([fname, content]) =>
                        `// === ${fname} ===\n\n${content}\n\n`
                      ).join('\n');
                      const blob = new Blob([parts], { type: 'text/plain' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${(pluginConfig.name || 'Plugin').replace(/[^a-zA-Z0-9_-]/g, '_')}_source.txt`;
                      document.body.appendChild(a); a.click(); document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                      return;
                    }
                    const zip = new JSZip();
                    const src = zip.folder('Source');
                    Object.entries(codePreview.files).forEach(([fname, content]) => {
                      if (fname === 'CMakeLists.txt') zip.file(fname, content);
                      else src.file(fname, content);
                    });
                    zip.generateAsync({ type: 'blob' }).then(blob => {
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${(pluginConfig.name || 'Plugin').replace(/[^a-zA-Z0-9_-]/g, '_')}_source.zip`;
                      document.body.appendChild(a); a.click(); document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    });
                  }}
                  style={{
                    padding: '6px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.15)', fontSize: 13, fontWeight: 600,
                    cursor: 'pointer', background: 'rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.7)',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}
                  title="Download raw JUCE C++ project source files"
                >
                  <i className="fa-solid fa-download" />
                  Source
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(codePreview.files[activeCodeFile] || '').then(() => {
                      showToast('Copied to clipboard!');
                    });
                  }}
                  style={{
                    padding: '6px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.15)', fontSize: 13, fontWeight: 600,
                    cursor: 'pointer', background: 'rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.7)',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}
                  title="Copy current file to clipboard"
                >
                  <i className="fa-solid fa-clipboard" />
                  Copy
                </button>
                <button
                  onClick={handleBuildPlugin}
                  disabled={buildingPlugin}
                  style={{
                    padding: '6px 14px', borderRadius: 8, border: 'none', fontSize: 13, fontWeight: 600,
                    cursor: buildingPlugin ? 'wait' : 'pointer',
                    background: buildingPlugin ? 'rgba(102,126,234,0.2)' : 'linear-gradient(135deg, #667eea, #764ba2)',
                    color: '#fff', display: 'flex', alignItems: 'center', gap: 6,
                  }}
                  title="Compile VST3 + AU on build server"
                >
                  <i className={`fa-solid ${buildingPlugin ? 'fa-spinner fa-spin' : 'fa-hammer'}`} />
                  {buildingPlugin ? (buildStage || 'Building...') : 'Build Plugin'}
                </button>
                <button
                  onClick={() => setShowCodePreview(false)}
                  style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.5)', fontSize: 18, cursor: 'pointer' }}
                >
                  <i className="fa-solid fa-xmark" />
                </button>
              </div>
            </div>
            {buildError && (
              <div style={{
                padding: '8px 12px', marginBottom: 8, borderRadius: 8,
                background: 'rgba(255,80,80,0.15)', border: '1px solid rgba(255,80,80,0.3)',
                color: 'rgba(255,150,150,1)', fontSize: 12,
              }}>
                <i className="fa-solid fa-triangle-exclamation" style={{ marginRight: 6 }} />
                {buildError}
              </div>
            )}
            <div style={{ display: 'flex', gap: 4, marginBottom: 8, flexWrap: 'wrap' }}>
              {Object.keys(codePreview.files).map(fname => (
                <button
                  key={fname}
                  onClick={() => setActiveCodeFile(fname)}
                  style={{
                    padding: '4px 10px', borderRadius: 6, border: 'none', fontSize: 12, cursor: 'pointer',
                    background: activeCodeFile === fname ? 'rgba(102,126,234,0.3)' : 'rgba(255,255,255,0.06)',
                    color: activeCodeFile === fname ? '#667eea' : 'rgba(255,255,255,0.5)',
                  }}
                >
                  {fname}
                </button>
              ))}
            </div>
            <pre style={{
              flex: 1, overflow: 'auto', padding: 16, borderRadius: 8,
              background: 'rgba(0,0,0,0.4)', fontSize: 12, lineHeight: 1.5,
              color: 'rgba(255,255,255,0.8)', margin: 0, whiteSpace: 'pre-wrap',
            }}>
              <code>{codePreview.files[activeCodeFile] || ''}</code>
            </pre>
          </div>
        </div>
      )}
      {/* Toast */}
      {toast && (
        <div style={{
          position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
          background: toast.type === 'error' ? 'rgba(244,67,54,0.95)' : 'rgba(76,175,80,0.95)',
          color: '#fff', padding: '10px 20px', borderRadius: 10, zIndex: 9999,
          fontSize: 14, fontWeight: 500, boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          display: 'flex', alignItems: 'center', gap: 10, animation: 'fadeInHint 0.3s ease',
        }}>
          <i className={`fa-solid ${toast.type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-check'}`} />
          {toast.message}
          {toast.link && (
            <a href={toast.link} style={{ color: '#fff', fontWeight: 700, textDecoration: 'underline' }}>View</a>
          )}
        </div>
      )}
    </div>
    </PluginContext.Provider>
  );
};

export default PluginCreator;
