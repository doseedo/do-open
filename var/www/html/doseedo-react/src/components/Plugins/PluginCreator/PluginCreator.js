import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import PluginCanvas from './PluginCanvas';
import ComponentPalette from './ComponentPalette';
import PropertyPanel from './PropertyPanel';
import CreatorChat from './CreatorChat';
import BackendChat from './BackendChat';
import AudioTestPanel from './AudioTestPanel';
import DSPEditor from './DSPEditor';
import LayersPanel from './LayersPanel';
import ImageBrowser from './ImageBrowser';
import useUndoRedo from './useUndoRedo';
import WebAudioDSPEngine from '../../../audio/WebAudioDSPEngine';
import { generateImage, searchImages } from '../../../services/chatAPI';
import { saveProject, loadProject, publishProject, generateCode } from '../../../services/pluginProjectsAPI';
import styles from './PluginCreator.module.css';

const COMPONENT_DEFAULTS = {
  knob:     { width: 60,  height: 70,  color: '#667eea', label: 'Knob' },
  slider:   { width: 30,  height: 120, color: '#667eea', label: 'Slider' },
  button:   { width: 70,  height: 28,  color: '#764ba2', label: 'Button' },
  label:    { width: 100, height: 24,  color: '#ffffff', label: 'Label' },
  led:      { width: 12,  height: 12,  color: '#4caf50', label: '' },
  dropdown: { width: 120, height: 28,  color: '#667eea', label: 'Select' },
  image:    { width: 80,  height: 80,  color: '#444',    label: 'Image', image: '' },
  panel:    { width: 200, height: 150, color: '#2a2a4a', label: 'Panel', borderColor: 'rgba(255,255,255,0.1)', bgColor: 'rgba(255,255,255,0.03)' },
  meter:    { width: 24,  height: 100, color: '#4caf50', label: 'Meter' },
  waveform: { width: 180, height: 60,  color: '#667eea', label: 'Waveform' },
  'xy-pad': { width: 120, height: 120, color: '#667eea', label: 'XY Pad' },
};

const ACCESS_KEY = '***REDACTED***';

let idCounter = 0;

const PluginCreator = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [unlocked, setUnlocked] = useState(() => sessionStorage.getItem('pc_auth') === '1');
  const [passInput, setPassInput] = useState('');
  const [passError, setPassError] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [showLayers, setShowLayers] = useState(false);
  const [showImageBrowser, setShowImageBrowser] = useState(false);
  const [imageBrowserTarget, setImageBrowserTarget] = useState(null); // 'background' | componentId
  const [activeTab, setActiveTab] = useState('ui'); // 'ui' | 'backend'
  const [dspConfig, setDspConfig] = useState(null);
  const [editorMode, setEditorMode] = useState('edit'); // 'edit' | 'test'
  const [paramValues, setParamValues] = useState({}); // componentId → 0-1 value
  const [snapToGrid, setSnapToGrid] = useState(false);
  const [gridSize, setGridSize] = useState(20);
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
  const [codePreview, setCodePreview] = useState(null);
  const [showCodePreview, setShowCodePreview] = useState(false);
  const [activeCodeFile, setActiveCodeFile] = useState(null);
  const [generatingCode, setGeneratingCode] = useState(false);
  const saveTimerRef = useRef(null);
  const lastSaveRef = useRef(null);

  const { state: components, pushState: pushComponents, undo, redo, setState: setComponentsDirect } = useUndoRedo([]);

  const snapValue = useCallback((val) => {
    if (!snapToGrid) return val;
    return Math.round(val / gridSize) * gridSize;
  }, [snapToGrid, gridSize]);

  // ── Save project ─────────────────────────────────────────────────────────
  const handleSave = useCallback(async (silent = false) => {
    if (!silent) setSaveStatus('saving');
    try {
      const projectData = {
        id: projectId || undefined,
        name: pluginConfig.name,
        plugin_config: pluginConfig,
        components,
        dsp_config: dspConfig,
      };
      const result = await saveProject(projectData);
      if (result.id && !projectId) {
        setProjectId(result.id);
        setSearchParams({ project: result.id }, { replace: true });
      }
      setSaveStatus('saved');
      lastSaveRef.current = Date.now();
      // Reset saved indicator after 2s
      setTimeout(() => setSaveStatus(prev => prev === 'saved' ? 'idle' : prev), 2000);
    } catch (err) {
      console.error('Save failed:', err);
      if (!silent) setSaveStatus('error');
      setTimeout(() => setSaveStatus(prev => prev === 'error' ? 'idle' : prev), 3000);
    }
  }, [projectId, pluginConfig, components, dspConfig, setSearchParams]);

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

  // ── Load project on mount ────────────────────────────────────────────────
  useEffect(() => {
    const pid = searchParams.get('project');
    if (pid && unlocked) {
      loadProject(pid).then(data => {
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
        }
        setIsPublished(data.is_public || false);
      }).catch(err => {
        console.error('Failed to load project:', err);
      });
    }
  }, [unlocked]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Generate JUCE code ───────────────────────────────────────────────────
  const handleGenerateCode = useCallback(async () => {
    if (!dspConfig) return;
    setGeneratingCode(true);
    try {
      const uiLayout = { pluginConfig, components };
      const result = await generateCode(dspConfig, uiLayout);
      setCodePreview(result);
      setShowCodePreview(true);
      setActiveCodeFile(Object.keys(result.files)[0] || null);
    } catch (err) {
      console.error('Code generation failed:', err);
    } finally {
      setGeneratingCode(false);
    }
  }, [dspConfig, pluginConfig, components]);

  // ── Publish ──────────────────────────────────────────────────────────────
  const handlePublish = useCallback(async () => {
    if (!projectId) {
      await handleSave(false);
    }
    const pid = projectId;
    if (!pid) return;
    try {
      const tags = publishTags.split(',').map(t => t.trim()).filter(Boolean);
      await publishProject(pid, {
        is_public: true,
        description: publishDesc || undefined,
        tags: tags.length ? tags : undefined,
      });
      setIsPublished(true);
      setShowPublishModal(false);
    } catch (err) {
      console.error('Publish failed:', err);
    }
  }, [projectId, publishDesc, publishTags, handleSave]);

  const handleUnpublish = useCallback(async () => {
    if (!projectId) return;
    try {
      await publishProject(projectId, { is_public: false });
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
    setSelectedId(id);
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

  // Lightweight drag update that doesn't push to undo history on every pixel
  const updateComponentDrag = useCallback((id, updates) => {
    setComponentsDirect(prev => prev.map(c => {
      if (c.id !== id) return c;
      const merged = { ...c, ...updates };
      if (snapToGrid) {
        if (updates.x !== undefined) merged.x = snapValue(updates.x);
        if (updates.y !== undefined) merged.y = snapValue(updates.y);
      }
      return merged;
    }));
  }, [setComponentsDirect, snapToGrid, snapValue]);

  // Push undo state only when drag stops
  const commitDrag = useCallback(() => {
    // Force push current state to undo history
    setComponentsDirect(prev => {
      pushComponents(prev);
      return prev;
    });
  }, [pushComponents, setComponentsDirect]);

  const deleteComponent = useCallback((id) => {
    pushComponents(components.filter(c => c.id !== id));
    setSelectedId(prev => prev === id ? null : prev);
  }, [components, pushComponents]);

  const duplicateComponent = useCallback((id) => {
    const comp = components.find(c => c.id === id);
    if (!comp) return;
    const newId = `comp-${++idCounter}-${Date.now()}`;
    const clone = { ...comp, id: newId, x: comp.x + 20, y: comp.y + 20 };
    pushComponents([...components, clone]);
    setSelectedId(newId);
  }, [components, pushComponents]);

  const selectComponent = useCallback((id) => setSelectedId(id), []);
  const deselectAll = useCallback(() => setSelectedId(null), []);

  // Resolve an image field that could be a URL string or {generate:...}/{search:...}
  const resolveImageField = useCallback(async (field) => {
    if (!field) return '';
    if (typeof field === 'string') return field;
    if (typeof field === 'object') {
      try {
        if (field.generate) {
          const result = await generateImage({ prompt: field.generate, size: '1024x1024' });
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
  }, []);

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
          resolveImageField(bgImageField).then(url => {
            if (url) setPluginConfig(prev => ({ ...prev, bgImage: url }));
          });
        }
      }
    }

    if (layout.components && Array.isArray(layout.components)) {
      const mode = layout.mode || 'replace';
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
        // Normalize image field — if it's an object, clear it now and resolve async
        if (comp.image && typeof comp.image === 'object') {
          const imageCmd = comp.image;
          comp.image = ''; // placeholder while loading
          const compId = comp.id;
          resolveImageField(imageCmd).then(url => {
            if (url) {
              // Update the component's image once resolved
              setComponentsDirect(prev => prev.map(cc =>
                cc.id === compId ? { ...cc, image: url } : cc
              ));
            }
          });
        }
        return comp;
      });
      if (mode === 'merge') {
        pushComponents([...components, ...newComps]);
      } else {
        pushComponents(newComps);
      }
      setSelectedId(null);
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
      setSelectedId(id);
    } else if (imageBrowserTarget) {
      // Apply to specific component
      updateComponent(imageBrowserTarget, { image: url });
    }
    setShowImageBrowser(false);
  }, [imageBrowserTarget, components, pushComponents, pluginConfig, updateComponent]);

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
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        if (selectedId) duplicateComponent(selectedId);
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedId) deleteComponent(selectedId);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [unlocked, selectedId, undo, redo, duplicateComponent, deleteComponent]);

  // ── Audio Engine ──────────────────────────────────────────────────────
  const engineRef = useRef(null);

  // Auto-map UI component labels to DSP parameter ids
  const paramMapping = useMemo(() => {
    if (!dspConfig?.parameters || components.length === 0) return {};
    const mapping = {}; // paramId → componentId
    const interactiveComps = components.filter(c => ['knob', 'slider', 'xy-pad'].includes(c.type));

    for (const param of dspConfig.parameters) {
      const paramName = (param.name || param.id || '').toLowerCase().replace(/[_\s-]/g, '');
      let bestMatch = null;
      for (const comp of interactiveComps) {
        const compLabel = (comp.label || '').toLowerCase().replace(/[_\s-]/g, '');
        if (compLabel === paramName || compLabel === param.id?.toLowerCase()) {
          bestMatch = comp.id;
          break;
        }
        // Partial match
        if (!bestMatch && (compLabel.includes(paramName) || paramName.includes(compLabel))) {
          bestMatch = comp.id;
        }
      }
      if (bestMatch) mapping[param.id] = bestMatch;
    }
    return mapping;
  }, [dspConfig, components]);

  // Reverse mapping: componentId → paramId
  const reverseMapping = useMemo(() => {
    const rev = {};
    for (const [paramId, compId] of Object.entries(paramMapping)) {
      rev[compId] = paramId;
    }
    return rev;
  }, [paramMapping]);

  // Create/destroy engine when entering/leaving test mode
  useEffect(() => {
    if (editorMode === 'test') {
      const engine = new WebAudioDSPEngine(dspConfig || { parameters: [], dspChain: [] });
      engineRef.current = engine;
      return () => {
        engine.dispose();
        engineRef.current = null;
      };
    } else {
      if (engineRef.current) {
        engineRef.current.dispose();
        engineRef.current = null;
      }
    }
  }, [editorMode, dspConfig]);

  const handleParamChange = useCallback((compId, value) => {
    setParamValues(prev => ({ ...prev, [compId]: value }));
    // Forward to audio engine if mapped
    const paramId = reverseMapping[compId];
    if (paramId && engineRef.current) {
      engineRef.current.setParameter(paramId, value);
    }
  }, [reverseMapping]);

  const selectedComponent = components.find(c => c.id === selectedId);

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
        <form onSubmit={handleUnlock} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, maxWidth: 320 }}>
          <i className="fa-solid fa-lock" style={{ fontSize: 32, color: 'rgba(186,156,255,0.5)' }} />
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600 }}>Plugin Creator</h2>
          <p style={{ margin: 0, fontSize: 13, color: 'rgba(255,255,255,0.4)', textAlign: 'center' }}>This tool is in early access. Enter the password to continue.</p>
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
            background: 'linear-gradient(135deg, #667eea, #764ba2)', color: '#fff',
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
    );
  }

  return (
    <div className={styles.creator}>
      <div className={styles.topBar}>
        <button className={styles.backBtn} onClick={() => navigate('/plugins')}>
          <i className="fa-solid fa-arrow-left" /> Back
        </button>
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
              onClick={() => { setEditorMode('test'); setSelectedId(null); }}
              title="Test mode — interact with knobs, sliders, buttons"
            >
              <i className="fa-solid fa-play" /> Test
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
          <div className={styles.topBarDivider} />
          {/* Save */}
          <button
            className={`${styles.topBarBtn} ${saveStatus === 'saved' ? styles.topBarBtnActive : ''}`}
            onClick={() => handleSave(false)}
            disabled={saveStatus === 'saving'}
            title={saveStatus === 'saved' ? 'Saved' : saveStatus === 'saving' ? 'Saving...' : 'Save project'}
          >
            <i className={`fa-solid ${saveStatus === 'saving' ? 'fa-spinner fa-spin' : saveStatus === 'saved' ? 'fa-check' : saveStatus === 'error' ? 'fa-exclamation-triangle' : 'fa-floppy-disk'}`} />
          </button>
          {/* Generate Code */}
          <button
            className={styles.topBarBtn}
            onClick={handleGenerateCode}
            disabled={!dspConfig || generatingCode}
            title={dspConfig ? 'Generate JUCE C++ code' : 'Design a DSP chain first'}
          >
            <i className={`fa-solid ${generatingCode ? 'fa-spinner fa-spin' : 'fa-code'}`} />
          </button>
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
        <div className={styles.chatPanel}>
          <div className={styles.chatTabBar}>
            <button
              className={activeTab === 'ui' ? styles.chatTabActive : styles.chatTab}
              onClick={() => setActiveTab('ui')}
            >
              <i className="fa-solid fa-wand-magic-sparkles" /> UI Designer
            </button>
            <button
              className={activeTab === 'backend' ? styles.chatTabActive : styles.chatTab}
              onClick={() => setActiveTab('backend')}
            >
              <i className="fa-solid fa-microchip" /> Backend Coder
            </button>
            <button
              className={activeTab === 'dsp' ? styles.chatTabActive : styles.chatTab}
              onClick={() => setActiveTab('dsp')}
            >
              <i className="fa-solid fa-diagram-project" /> DSP Editor
            </button>
          </div>
          {activeTab === 'ui' ? (
            <CreatorChat
              pluginConfig={pluginConfig}
              components={components}
              dspContext={dspConfig}
              onApplyLayout={applyLayout}
              onOpenImageBrowser={openImageBrowser}
            />
          ) : activeTab === 'backend' ? (
            <BackendChat
              pluginConfig={pluginConfig}
              components={components}
              dspConfig={dspConfig}
              onApplyDsp={setDspConfig}
            />
          ) : (
            <DSPEditor
              dspConfig={dspConfig}
              onUpdateDsp={setDspConfig}
            />
          )}
        </div>

        {/* Canvas Panel */}
        <div className={styles.canvasPanel}>
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
              max={1200}
            />
            <span style={{ color: 'rgba(255,255,255,0.25)', fontSize: 12 }}>&times;</span>
            <input
              className={styles.configInputSmall}
              type="number"
              value={pluginConfig.height}
              onChange={(e) => setPluginConfig(prev => ({ ...prev, height: Math.max(150, parseInt(e.target.value) || 150) }))}
              min={150}
              max={800}
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

          {/* Plugin Canvas */}
          <PluginCanvas
            config={pluginConfig}
            components={components}
            selectedId={selectedId}
            onSelect={selectComponent}
            onDeselect={deselectAll}
            onUpdateComponent={updateComponentDrag}
            onDragStop={commitDrag}
            snapToGrid={snapToGrid}
            gridSize={gridSize}
            editorMode={editorMode}
            paramValues={paramValues}
            onParamChange={handleParamChange}
          />

          {/* Audio Test Panel (test mode only) */}
          {editorMode === 'test' && (
            <AudioTestPanel
              engine={engineRef.current}
              dspConfig={dspConfig}
              paramValues={paramValues}
              paramMapping={paramMapping}
              components={components}
              onParamChange={handleParamChange}
            />
          )}

          {/* Property Panel (edit mode only) */}
          {editorMode === 'edit' && selectedComponent && (
            <PropertyPanel
              component={selectedComponent}
              onUpdate={(updates) => updateComponent(selectedId, updates)}
              onDelete={() => deleteComponent(selectedId)}
              onDuplicate={() => duplicateComponent(selectedId)}
              onOpenImageBrowser={() => openImageBrowser(selectedId)}
            />
          )}
        </div>

        {/* Layers Panel */}
        {showLayers && (
          <LayersPanel
            components={components}
            selectedId={selectedId}
            onSelect={selectComponent}
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
                style={{
                  padding: '8px 20px', borderRadius: 8, border: 'none',
                  background: 'linear-gradient(135deg, #667eea, #764ba2)', color: '#fff',
                  fontSize: 13, fontWeight: 600, cursor: 'pointer',
                }}
              >
                Publish
              </button>
            </div>
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
              <button
                onClick={() => setShowCodePreview(false)}
                style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.5)', fontSize: 18, cursor: 'pointer' }}
              >
                <i className="fa-solid fa-xmark" />
              </button>
            </div>
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
    </div>
  );
};

export default PluginCreator;
