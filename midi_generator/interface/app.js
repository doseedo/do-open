// ==================== STATE MANAGEMENT ====================
const state = {
    currentFile: null,
    originalParams: null,
    currentParams: null,
    modifiedParams: new Set(),
    viewMode: 'tree'
};

// ==================== CONFIGURATION ====================
const API_BASE_URL = 'http://localhost:5001';

// Parameter schema for controls
const PARAMETER_SCHEMA = {
    level1: {
        'tempo.bpm': { type: 'number', min: 40, max: 240, step: 1, unit: 'BPM' },
        'time_signature': { type: 'select', options: ['4/4', '3/4', '6/8', '5/4', '7/8', '12/8'] },
        'key.tonic': { type: 'select', options: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] },
        'key.mode': { type: 'select', options: ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian'] },
        'genre.primary': { type: 'select', options: ['jazz', 'classical', 'rock', 'electronic', 'pop', 'hiphop', 'latin'] },
        'structure.form': { type: 'select', options: ['AABA', 'verse_chorus', 'through_composed', 'binary', 'ternary'] },
        'energy.level': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
        'complexity.overall': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' }
    },
    level2: {
        harmony: {
            'chord_density': { type: 'number', min: 0, max: 20, step: 0.1, unit: '' },
            'complexity': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'chromaticism': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'tension': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'voicing_spread': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'progression_predictability': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' }
        },
        melody: {
            'note_density': { type: 'number', min: 0, max: 50, step: 0.1, unit: 'notes/measure' },
            'range_semitones': { type: 'number', min: 0, max: 48, step: 1, unit: 'semitones' },
            'contour_smoothness': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'rhythmic_complexity': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'repetition': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' }
        },
        rhythm: {
            'subdivision': { type: 'select', options: ['quarter', 'eighth', 'triplet', 'sixteenth'] },
            'syncopation': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'groove_consistency': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'polyrhythm': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'swing_amount': { type: 'number', min: 0.5, max: 0.75, step: 0.01, unit: '' }
        },
        dynamics: {
            'overall_level': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' },
            'range': { type: 'number', min: 0, max: 1, step: 0.01, unit: '' }
        },
        texture: {
            'polyphony': { type: 'number', min: 1, max: 32, step: 1, unit: 'voices' },
            'density': { type: 'number', min: 0, max: 50, step: 0.1, unit: 'notes/sec' }
        }
    }
};

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeUpload();
    initializeControls();
    initializeEventListeners();
    renderEmptyState();
});

// ==================== FILE UPLOAD ====================
function initializeUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const clearBtn = document.getElementById('clearFile');

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFileUpload(file);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && (file.name.endsWith('.mid') || file.name.endsWith('.midi'))) {
            handleFileUpload(file);
        } else {
            showToast('Please upload a valid MIDI file (.mid or .midi)', 'error');
        }
    });

    // Clear file
    clearBtn.addEventListener('click', () => {
        state.currentFile = null;
        state.originalParams = null;
        state.currentParams = null;
        state.modifiedParams.clear();
        fileInput.value = '';
        document.getElementById('fileInfo').classList.add('hidden');
        document.getElementById('uploadArea').classList.remove('hidden');
        renderEmptyState();
        updateStats();
    });
}

async function handleFileUpload(file) {
    state.currentFile = file;

    // Update UI
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileInfo').classList.remove('hidden');
    document.getElementById('currentFile').textContent = file.name;

    // Show loading
    showLoading(true);

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);

        // Send to backend
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to analyze MIDI file');
        }

        const data = await response.json();

        // Store parameters
        state.originalParams = JSON.parse(JSON.stringify(data));
        state.currentParams = data;
        state.modifiedParams.clear();

        // Render visualization
        renderHierarchy(data);
        renderControls(data);
        updateStats();

        showToast('MIDI file analyzed successfully!', 'success');
    } catch (error) {
        console.error('Error analyzing MIDI:', error);
        showToast('Error analyzing MIDI file. Make sure the backend server is running.', 'error');
    } finally {
        showLoading(false);
    }
}

// ==================== RENDERING ====================
function renderEmptyState() {
    // Level 1
    document.getElementById('level1Nodes').innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">Upload a MIDI file to see parameters</p>';

    // Level 2
    ['harmony', 'melody', 'rhythm', 'dynamics', 'texture'].forEach(category => {
        document.getElementById(`${category}Nodes`).innerHTML = '';
    });

    // Level 3
    document.getElementById('level3Nodes').innerHTML = '';
}

function renderHierarchy(params) {
    renderLevel1(params.level1_global);
    renderLevel2(params.level2_universal);
    renderLevel3(params.level3_genre_specific);
}

function renderLevel1(level1) {
    const container = document.getElementById('level1Nodes');
    container.innerHTML = '';

    Object.entries(level1).forEach(([key, value]) => {
        const node = createParameterNode(key, value, 'level1');
        container.appendChild(node);
    });
}

function renderLevel2(level2) {
    // Render each category
    Object.entries(level2).forEach(([category, params]) => {
        const container = document.getElementById(`${category}Nodes`);
        container.innerHTML = '';

        Object.entries(params).forEach(([key, value]) => {
            const node = createParameterNode(key, value, 'level2', category);
            container.appendChild(node);
        });
    });
}

function renderLevel3(level3) {
    const container = document.getElementById('level3Nodes');
    container.innerHTML = '';

    Object.entries(level3).forEach(([category, params]) => {
        if (typeof params === 'object') {
            Object.entries(params).forEach(([key, value]) => {
                const node = createParameterNode(key, value, 'level3', category);
                container.appendChild(node);
            });
        }
    });
}

function createParameterNode(key, value, level, category = null) {
    const node = document.createElement('div');
    node.className = 'param-node';
    node.dataset.key = key;
    node.dataset.level = level;
    if (category) node.dataset.category = category;

    const name = document.createElement('div');
    name.className = 'param-name';
    name.textContent = formatParamName(key);

    const valueDiv = document.createElement('div');
    valueDiv.className = `param-value ${typeof value === 'string' ? 'string' : ''}`;
    valueDiv.textContent = formatParamValue(value);

    node.appendChild(name);
    node.appendChild(valueDiv);

    // Click to highlight corresponding control
    node.addEventListener('click', () => {
        highlightControl(key, level, category);
    });

    return node;
}

// ==================== CONTROLS ====================
function renderControls(params) {
    // Level 1
    renderLevel1Controls(params.level1_global);

    // Level 2
    renderLevel2Controls(params.level2_universal);

    // Level 3
    renderLevel3Controls(params.level3_genre_specific);
}

function renderLevel1Controls(level1) {
    const container = document.querySelector('#level1Controls .controls-list');
    container.innerHTML = '';

    Object.entries(level1).forEach(([key, value]) => {
        const control = createParameterControl(key, value, 'level1', null);
        container.appendChild(control);
    });
}

function renderLevel2Controls(level2) {
    Object.entries(level2).forEach(([category, params]) => {
        const container = document.getElementById(`${category}Controls`);
        container.innerHTML = '';

        Object.entries(params).forEach(([key, value]) => {
            const control = createParameterControl(key, value, 'level2', category);
            container.appendChild(control);
        });
    });
}

function renderLevel3Controls(level3) {
    const container = document.querySelector('#level3Controls .controls-list');
    container.innerHTML = '';

    Object.entries(level3).forEach(([category, params]) => {
        if (typeof params === 'object') {
            // Add category header
            const categoryHeader = document.createElement('h4');
            categoryHeader.className = 'subsection-title';
            categoryHeader.textContent = `${formatParamName(category)}`;
            container.appendChild(categoryHeader);

            Object.entries(params).forEach(([key, value]) => {
                const control = createParameterControl(key, value, 'level3', category);
                container.appendChild(control);
            });
        }
    });
}

function createParameterControl(key, value, level, category) {
    const control = document.createElement('div');
    control.className = 'param-control';
    control.dataset.key = key;
    control.dataset.level = level;
    if (category) control.dataset.category = category;

    // Header
    const header = document.createElement('div');
    header.className = 'param-control-header';

    const name = document.createElement('div');
    name.className = 'param-control-name';
    name.textContent = formatParamName(key);

    const valueDisplay = document.createElement('div');
    valueDisplay.className = 'param-control-value';
    valueDisplay.textContent = formatParamValue(value);

    header.appendChild(name);
    header.appendChild(valueDisplay);
    control.appendChild(header);

    // Get schema for this parameter
    const schema = getParameterSchema(key, level, category);

    if (schema) {
        let input;

        if (schema.type === 'number') {
            input = document.createElement('input');
            input.type = 'range';
            input.min = schema.min;
            input.max = schema.max;
            input.step = schema.step;
            input.value = value;

            input.addEventListener('input', (e) => {
                const newValue = parseFloat(e.target.value);
                valueDisplay.textContent = formatParamValue(newValue) + (schema.unit ? ` ${schema.unit}` : '');
                updateParameter(key, newValue, level, category);
            });
        } else if (schema.type === 'select') {
            input = document.createElement('select');
            schema.options.forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option;
                opt.selected = option === value;
                input.appendChild(opt);
            });

            input.addEventListener('change', (e) => {
                const newValue = e.target.value;
                valueDisplay.textContent = newValue;
                updateParameter(key, newValue, level, category);
            });
        }

        if (input) {
            control.appendChild(input);
        }
    }

    return control;
}

function getParameterSchema(key, level, category) {
    if (level === 'level1') {
        return PARAMETER_SCHEMA.level1[key];
    } else if (level === 'level2' && category) {
        return PARAMETER_SCHEMA.level2[category]?.[key];
    }
    return null;
}

// ==================== PARAMETER UPDATES ====================
function updateParameter(key, value, level, category) {
    // Update state
    if (level === 'level1') {
        state.currentParams.level1_global[key] = value;
    } else if (level === 'level2' && category) {
        state.currentParams.level2_universal[category][key] = value;
    } else if (level === 'level3' && category) {
        if (!state.currentParams.level3_genre_specific[category]) {
            state.currentParams.level3_genre_specific[category] = {};
        }
        state.currentParams.level3_genre_specific[category][key] = value;
    }

    // Mark as modified
    const paramId = `${level}-${category || 'global'}-${key}`;
    state.modifiedParams.add(paramId);

    // Update visualization
    updateParameterNode(key, value, level, category);

    // Mark control as modified
    const control = document.querySelector(`.param-control[data-key="${key}"][data-level="${level}"]`);
    if (control) control.classList.add('modified');

    // Update stats
    updateStats();
}

function updateParameterNode(key, value, level, category) {
    const node = document.querySelector(`.param-node[data-key="${key}"][data-level="${level}"]`);
    if (node) {
        const valueDiv = node.querySelector('.param-value');
        valueDiv.textContent = formatParamValue(value);
        node.classList.add('modified');
    }
}

// ==================== UTILITIES ====================
function formatParamName(key) {
    return key
        .split(/[._]/)
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function formatParamValue(value) {
    if (typeof value === 'number') {
        return value % 1 === 0 ? value.toString() : value.toFixed(2);
    }
    return value.toString();
}

function highlightControl(key, level, category) {
    // Remove previous highlights
    document.querySelectorAll('.param-control.active').forEach(el => {
        el.classList.remove('active');
    });

    // Highlight new control
    const control = document.querySelector(
        `.param-control[data-key="${key}"][data-level="${level}"]${category ? `[data-category="${category}"]` : ''}`
    );

    if (control) {
        control.classList.add('active');
        control.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function updateStats() {
    document.getElementById('modifiedParams').textContent = state.modifiedParams.size;

    if (state.currentParams?.metadata) {
        const meta = state.currentParams.metadata;
        document.getElementById('duration').textContent = `${meta.duration_seconds.toFixed(1)}s`;
        document.getElementById('noteCount').textContent = meta.total_notes;
    }
}

// ==================== EVENT LISTENERS ====================
function initializeControls() {
    // View mode buttons
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.viewMode = e.target.dataset.view;
            // TODO: Implement different view modes
        });
    });

    // Reset button
    document.getElementById('resetBtn').addEventListener('click', () => {
        if (state.originalParams && confirm('Reset all parameters to original values?')) {
            state.currentParams = JSON.parse(JSON.stringify(state.originalParams));
            state.modifiedParams.clear();
            renderHierarchy(state.currentParams);
            renderControls(state.currentParams);
            updateStats();
            showToast('Parameters reset to original values', 'success');
        }
    });

    // Export button
    document.getElementById('exportBtn').addEventListener('click', exportParameters);

    // Search functionality
    document.getElementById('paramSearch').addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        document.querySelectorAll('.param-control').forEach(control => {
            const name = control.querySelector('.param-control-name').textContent.toLowerCase();
            control.style.display = name.includes(query) ? 'block' : 'none';
        });
    });

    // Collapsible sections
    document.querySelectorAll('.control-section-title').forEach(title => {
        title.addEventListener('click', () => {
            title.parentElement.classList.toggle('collapsed');
        });
    });
}

function initializeEventListeners() {
    // Add any additional event listeners here
}

// ==================== EXPORT ====================
function exportParameters() {
    if (!state.currentParams) {
        showToast('No parameters to export', 'warning');
        return;
    }

    const exportData = {
        original: state.originalParams,
        modified: state.currentParams,
        changes: Array.from(state.modifiedParams),
        timestamp: new Date().toISOString(),
        file: state.currentFile?.name || 'unknown'
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${state.currentFile?.name.replace('.mid', '')}_parameters.json`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Parameters exported successfully!', 'success');
}

// ==================== UI HELPERS ====================
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
