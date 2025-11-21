#!/usr/bin/env python3
"""
Flask API Server for MIDI Parameter Visualization Interface

This server provides endpoints for:
- MIDI file upload and analysis
- Real-time parameter extraction
- Parameter modification and export

Run with: python3 server.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path to import midi_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.parameters.hierarchical_extractor import HierarchicalParameterExtractor

# ==================== FLASK APP SETUP ====================
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mid', 'midi'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize extractor
extractor = HierarchicalParameterExtractor(verbose=False)

# ==================== HELPER FUNCTIONS ====================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def serialize_params(params):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(params, dict):
        return {k: serialize_params(v) for k, v in params.items()}
    elif isinstance(params, list):
        return [serialize_params(item) for item in params]
    elif hasattr(params, 'item'):  # numpy types
        return params.item()
    else:
        return params

# ==================== API ENDPOINTS ====================

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'MIDI Parameter Visualization API',
        'version': '1.0.0',
        'endpoints': {
            '/analyze': 'POST - Upload and analyze MIDI file',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/analyze', methods=['POST'])
def analyze_midi():
    """
    Analyze uploaded MIDI file and return hierarchical parameters

    Expected: multipart/form-data with 'file' field
    Returns: JSON with parameter hierarchy
    """
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .mid and .midi files are allowed'}), 400

    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        print(f"Analyzing MIDI file: {filename}")

        # Extract parameters
        params = extractor.extract_from_midi(temp_path)

        # Serialize for JSON (convert numpy types)
        params_serialized = serialize_params(params)

        # Clean up temp file
        os.remove(temp_path)

        print(f"✓ Successfully analyzed: {filename}")

        return jsonify(params_serialized)

    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"✗ Error analyzing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'error': 'Failed to analyze MIDI file',
            'details': str(e)
        }), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple MIDI files in batch

    Expected: multipart/form-data with multiple 'files' fields
    Returns: JSON array of parameter hierarchies
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    results = []
    errors = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            errors.append({
                'file': file.filename,
                'error': 'Invalid file'
            })
            continue

        try:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)

            # Extract parameters
            params = extractor.extract_from_midi(temp_path)
            params_serialized = serialize_params(params)

            results.append({
                'file': filename,
                'params': params_serialized
            })

            # Clean up
            os.remove(temp_path)

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)

            errors.append({
                'file': filename,
                'error': str(e)
            })

    return jsonify({
        'results': results,
        'errors': errors,
        'total': len(files),
        'successful': len(results),
        'failed': len(errors)
    })

@app.route('/export', methods=['POST'])
def export_parameters():
    """
    Export modified parameters

    Expected: JSON with parameter data
    Returns: JSON file download
    """
    try:
        data = request.get_json()

        # Validate data
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Create export filename
        filename = data.get('filename', 'parameters') + '.json'

        return jsonify({
            'success': True,
            'data': data
        })

    except Exception as e:
        return jsonify({
            'error': 'Export failed',
            'details': str(e)
        }), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'max_size': f'{MAX_FILE_SIZE / (1024 * 1024)}MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error',
        'details': str(error)
    }), 500

# ==================== MAIN ====================

def main():
    """Start the Flask server"""
    print("=" * 80)
    print("MIDI Parameter Visualization API Server")
    print("=" * 80)
    print()
    print("Server starting...")
    print(f"  - Upload folder: {UPLOAD_FOLDER}")
    print(f"  - Max file size: {MAX_FILE_SIZE / (1024 * 1024)}MB")
    print(f"  - Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print()
    print("API Endpoints:")
    print("  - GET  /          - Health check")
    print("  - POST /analyze   - Analyze single MIDI file")
    print("  - POST /batch-analyze - Analyze multiple MIDI files")
    print("  - POST /export    - Export parameters")
    print()
    print("Server running on http://localhost:5001")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()

    # Run server (use port 5001 to avoid conflict with macOS AirPlay)
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )

if __name__ == '__main__':
    main()
