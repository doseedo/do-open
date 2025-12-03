#!/usr/bin/env python3
"""
Hierarchy Server for Transform Editor
======================================

Simple HTTP server that exposes grammar hierarchy from NPZ checkpoints
for the web-based transform editor.

Includes round-trip validation: upload MIDI -> encode -> decode -> download.

Usage:
    python hierarchy_server.py [checkpoint.npz] [--port 8765]
"""

import http.server
import socketserver
import json
import os
import sys
import urllib.parse
import tempfile
import base64
import numpy as np
from pathlib import Path
from io import BytesIO

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / '1_approaches' / 'transform_based'))

from grammar.level_api import GrammarHierarchy
from validation.live_round_trip import (
    load_midi_notes,
    encode_with_grammar,
    decode_tokens,
    save_reconstructed_midi,
    compare_notes,
)


class HierarchyHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for grammar hierarchy API."""

    grammar = None  # Will be set by server
    checkpoint_path = None
    canonicals = None  # Canonical patterns from checkpoint

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # API endpoints
        if path == '/api/hierarchy':
            self.send_hierarchy()
        elif path == '/api/stats':
            self.send_stats()
        elif path == '/api/level':
            level = query.get('level', ['motif'])[0]
            limit = int(query.get('limit', [100])[0])
            self.send_level_patterns(level, limit)
        elif path == '/api/depth':
            depth = int(query.get('depth', [3])[0])
            limit = int(query.get('limit', [100])[0])
            self.send_depth_patterns(depth, limit)
        elif path == '/api/rule':
            rule_id = query.get('id', ['0'])[0]
            self.send_rule_details(rule_id)
        elif path == '/api/expand':
            rule_id = query.get('id', ['0'])[0]
            self.send_rule_expansion(rule_id)
        elif path == '/api/subtree':
            rule_id = query.get('id', ['0'])[0]
            max_depth = int(query.get('max_depth', [3])[0])
            self.send_subtree(rule_id, max_depth)
        elif path == '/api/canonicals':
            self.send_canonicals()
        else:
            # Serve static files
            super().do_GET()

    def do_POST(self):
        """Handle POST requests for file uploads."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/api/round-trip':
            self.handle_round_trip()
        elif path == '/api/encode':
            self.handle_encode()
        else:
            self.send_json({'error': 'Unknown POST endpoint'}, 404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_hierarchy(self):
        """Send full hierarchy export."""
        if self.grammar is None:
            self.send_json({'error': 'No checkpoint loaded'}, 500)
            return

        data = self.grammar.to_json_export()
        data['checkpoint'] = self.checkpoint_path
        self.send_json(data)

    def send_stats(self):
        """Send level statistics."""
        if self.grammar is None:
            self.send_json({'error': 'No checkpoint loaded'}, 500)
            return

        stats = self.grammar.get_level_stats()
        data = {
            'max_depth': self.grammar.get_max_depth(),
            'rule_count': self.grammar.get_rule_count(),
            'checkpoint': self.checkpoint_path,
            'levels': {
                depth: {
                    'semantic': s.semantic_level,
                    'count': s.rule_count,
                    'avg_length': round(s.avg_expansion_length, 1),
                    'min_length': s.min_expansion_length,
                    'max_length': s.max_expansion_length,
                }
                for depth, s in stats.items()
            }
        }
        self.send_json(data)

    def send_level_patterns(self, level: str, limit: int):
        """Send patterns at a semantic level."""
        if self.grammar is None:
            self.send_json({'error': 'No checkpoint loaded'}, 500)
            return

        patterns = self.grammar.get_patterns_at_level(level)[:limit]
        data = {
            'level': level,
            'count': len(self.grammar.by_level.get(level, [])),
            'patterns': [
                {
                    'id': p.id,
                    'depth': p.depth,
                    'expansion_length': p.expansion_length,
                    'child_count': p.child_rule_count,
                    'terminal_count': p.terminal_count,
                }
                for p in patterns
            ]
        }
        self.send_json(data)

    def send_depth_patterns(self, depth: int, limit: int):
        """Send patterns at a specific depth."""
        if self.grammar is None:
            self.send_json({'error': 'No checkpoint loaded'}, 500)
            return

        patterns = self.grammar.get_patterns_at_depth(depth)[:limit]
        data = {
            'depth': depth,
            'count': len(self.grammar.by_depth.get(depth, [])),
            'patterns': [
                {
                    'id': p.id,
                    'depth': p.depth,
                    'expansion_length': p.expansion_length,
                    'child_count': p.child_rule_count,
                    'terminal_count': p.terminal_count,
                }
                for p in patterns
            ]
        }
        self.send_json(data)

    def send_rule_details(self, rule_id: str):
        """Send details for a specific rule."""
        if self.grammar is None:
            self.send_json({'error': 'No checkpoint loaded'}, 500)
            return

        rule = self.grammar.get_rule(rule_id)
        if rule is None:
            self.send_json({'error': f'Rule {rule_id} not found'}, 404)
            return

        expansion = self.grammar.expand_rule(rule_id)
        children = self.grammar.get_children(rule_id)
        parents = self.grammar.get_parent_rules(rule_id)

        data = {
            'id': rule.id,
            'depth': rule.depth,
            'expansion_length': rule.expansion_length,
            'terminal_count': rule.terminal_count,
            'child_rule_count': rule.child_rule_count,
            'children': [{'id': c.id, 'depth': c.depth, 'length': c.expansion_length} for c in children],
            'parents': [{'id': p.id, 'depth': p.depth, 'length': p.expansion_length} for p in parents[:20]],
            'expansion_preview': expansion[:100],
            'raw': rule.raw[:20] if len(rule.raw) > 20 else rule.raw,
        }
        self.send_json(data)

    def send_rule_expansion(self, rule_id: str):
        """Send full expansion for a rule."""
        if self.grammar is None:
            self.send_json({'error': 'No checkpoint loaded'}, 500)
            return

        expansion = self.grammar.expand_rule(rule_id)
        data = {
            'id': rule_id,
            'length': len(expansion),
            'expansion': expansion,
        }
        self.send_json(data)

    def send_subtree(self, rule_id: str, max_depth: int):
        """Send rule subtree."""
        if self.grammar is None:
            self.send_json({'error': 'No checkpoint loaded'}, 500)
            return

        def serialize_subtree(tree):
            if 'terminal' in tree:
                return tree
            result = {
                'id': tree['rule'].id,
                'depth': tree['rule'].depth,
                'expansion_length': tree['rule'].expansion_length,
            }
            if isinstance(tree.get('children'), list):
                result['children'] = [serialize_subtree(c) for c in tree['children']]
            else:
                result['children'] = tree.get('children', [])
            return result

        subtree = self.grammar.get_subtree(rule_id, max_depth)
        data = serialize_subtree(subtree)
        self.send_json(data)

    def send_canonicals(self):
        """Send canonical patterns for encoding."""
        if self.canonicals is None:
            self.send_json({'error': 'No canonicals loaded'}, 500)
            return

        # Send summary of canonicals
        data = {
            'count': len(self.canonicals),
            'patterns': [
                {
                    'id': i,
                    'pitch_classes': c.get('pitch_classes', []),
                    'length': len(c.get('pitch_classes', [])),
                }
                for i, c in enumerate(self.canonicals[:100])
            ]
        }
        self.send_json(data)

    def handle_round_trip(self):
        """Handle round-trip validation request."""
        if self.canonicals is None:
            self.send_json({'error': 'No canonicals loaded'}, 500)
            return

        try:
            # Read uploaded MIDI file
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            # Parse JSON body
            data = json.loads(body.decode('utf-8'))
            midi_base64 = data.get('midi_data', '')

            if not midi_base64:
                self.send_json({'error': 'No MIDI data provided'}, 400)
                return

            # Decode base64 MIDI
            midi_bytes = base64.b64decode(midi_base64)

            # Save to temp file for mido to read
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
                f.write(midi_bytes)
                temp_midi_path = f.name

            try:
                # Load original notes
                original_notes, tpb = load_midi_notes(temp_midi_path)

                # Encode using existing grammar
                tokens, covered_notes = encode_with_grammar(original_notes, self.canonicals)

                # Decode back
                reconstructed_notes = decode_tokens(tokens, self.canonicals)

                # Compare
                metrics = compare_notes(original_notes, reconstructed_notes)

                # Save reconstructed MIDI to temp file
                with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
                    recon_path = f.name

                save_reconstructed_midi(reconstructed_notes, recon_path, tpb)

                # Read back and encode as base64
                with open(recon_path, 'rb') as f:
                    recon_base64 = base64.b64encode(f.read()).decode('utf-8')

                os.unlink(recon_path)

                # Build response
                response = {
                    'success': True,
                    'original_notes': len(original_notes),
                    'reconstructed_notes': len(reconstructed_notes),
                    'patterns_used': len(tokens),
                    'canonicals_matched': len(set(
                        t.get('pattern_idx') for t in tokens if t.get('type') == 'INTRO'
                    )),
                    'metrics': metrics,
                    'encoding': tokens[:50],  # First 50 tokens
                    'reconstructed_midi': recon_base64,
                }
                self.send_json(response)

            finally:
                os.unlink(temp_midi_path)

        except Exception as e:
            import traceback
            self.send_json({
                'error': str(e),
                'traceback': traceback.format_exc()
            }, 500)

    def handle_encode(self):
        """Handle encode-only request (no decode)."""
        if self.canonicals is None:
            self.send_json({'error': 'No canonicals loaded'}, 500)
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            midi_base64 = data.get('midi_data', '')

            if not midi_base64:
                self.send_json({'error': 'No MIDI data provided'}, 400)
                return

            midi_bytes = base64.b64decode(midi_base64)

            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
                f.write(midi_bytes)
                temp_midi_path = f.name

            try:
                original_notes, tpb = load_midi_notes(temp_midi_path)
                tokens, covered_notes = encode_with_grammar(original_notes, self.canonicals)

                # Note names for display
                NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

                response = {
                    'success': True,
                    'original_notes': len(original_notes),
                    'covered_notes': len(covered_notes),
                    'coverage': len(covered_notes) / len(original_notes) if original_notes else 0,
                    'patterns_used': len(tokens),
                    'encoding': [
                        {
                            **t,
                            'canonical_notes': [
                                NOTE_NAMES[pc] for pc in
                                self.canonicals[t['pattern_idx']].get('pitch_classes', [])
                            ] if t.get('type') == 'INTRO' and 0 <= t.get('pattern_idx', -1) < len(self.canonicals) else None
                        }
                        for t in tokens
                    ],
                }
                self.send_json(response)

            finally:
                os.unlink(temp_midi_path)

        except Exception as e:
            import traceback
            self.send_json({
                'error': str(e),
                'traceback': traceback.format_exc()
            }, 500)


def run_server(checkpoint_path: str, port: int = 8765, directory: str = None):
    """Run the hierarchy server."""
    # Load grammar
    print(f"Loading checkpoint: {checkpoint_path}")
    grammar = GrammarHierarchy.from_checkpoint(checkpoint_path)
    print(f"Loaded {grammar.get_rule_count()} rules, max depth {grammar.get_max_depth()}")

    # Load canonicals for encoding
    ckpt = np.load(checkpoint_path, allow_pickle=True)
    canonicals = json.loads(str(ckpt['canonical_patterns_json'][0]))
    print(f"Loaded {len(canonicals)} canonical patterns for encoding")

    HierarchyHandler.grammar = grammar
    HierarchyHandler.checkpoint_path = checkpoint_path
    HierarchyHandler.canonicals = canonicals

    # Change to web interface directory
    if directory:
        os.chdir(directory)
    else:
        os.chdir(Path(__file__).parent)

    with socketserver.TCPServer(("", port), HierarchyHandler) as httpd:
        print(f"\nServer running at http://localhost:{port}")
        print(f"Open transform_editor.html in your browser")
        print(f"\nAPI endpoints:")
        print(f"  GET  /api/hierarchy  - Full hierarchy export")
        print(f"  GET  /api/stats      - Level statistics")
        print(f"  GET  /api/level?level=motif&limit=100")
        print(f"  GET  /api/depth?depth=3&limit=100")
        print(f"  GET  /api/rule?id=123")
        print(f"  GET  /api/expand?id=123")
        print(f"  GET  /api/subtree?id=123&max_depth=3")
        print(f"  GET  /api/canonicals - List canonical patterns")
        print(f"  POST /api/encode     - Encode MIDI file")
        print(f"  POST /api/round-trip - Full round-trip test")
        print(f"\nPress Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hierarchy Server for Transform Editor')
    parser.add_argument('checkpoint', nargs='?', default='../../1_approaches/transform_based/checkpoint_v2.npz',
                        help='Path to checkpoint NPZ file')
    parser.add_argument('--port', type=int, default=8765, help='Port to run server on')
    args = parser.parse_args()

    run_server(args.checkpoint, args.port)
