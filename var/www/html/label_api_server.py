#!/usr/bin/env python3
"""
Simple Flask API for audio labeling interface.
Saves labels to /var/www/html/audio_labels.json

Run: python label_api_server.py (runs on port 8095)
"""
import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

LABELS_FILE = '/var/www/html/audio_labels.json'

def load_labels():
    """Load existing labels from file."""
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_labels(labels):
    """Save labels to file."""
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)

@app.route('/api/labels', methods=['POST'])
def add_label():
    """Add a new label."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        filename = data.get('filename')
        group = data.get('group')
        timestamp = data.get('timestamp', datetime.now().isoformat())

        if not filename or not group:
            return jsonify({'error': 'filename and group are required'}), 400

        labels = load_labels()

        # Update existing label or add new one
        found = False
        for label in labels:
            if label.get('filename') == filename:
                label['group'] = group
                label['timestamp'] = timestamp
                found = True
                break

        if not found:
            labels.append({
                'filename': filename,
                'group': group,
                'timestamp': timestamp
            })

        save_labels(labels)

        return jsonify({'success': True, 'message': f'Label saved: {filename} -> {group}'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/labels', methods=['GET'])
def get_labels():
    """Get all labels."""
    try:
        labels = load_labels()
        return jsonify(labels), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/labels/<filename>', methods=['GET'])
def get_label(filename):
    """Get label for a specific file."""
    try:
        labels = load_labels()
        for label in labels:
            if label.get('filename') == filename:
                return jsonify(label), 200
        return jsonify({'error': 'Label not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/labels/<filename>', methods=['DELETE'])
def delete_label(filename):
    """Delete a label."""
    try:
        labels = load_labels()
        labels = [l for l in labels if l.get('filename') != filename]
        save_labels(labels)
        return jsonify({'success': True, 'message': f'Label deleted: {filename}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # Initialize empty labels file if it doesn't exist
    if not os.path.exists(LABELS_FILE):
        save_labels([])

    print(f"Starting Label API server on port 8095...")
    print(f"Labels will be saved to: {LABELS_FILE}")
    app.run(host='0.0.0.0', port=8095, debug=False)
