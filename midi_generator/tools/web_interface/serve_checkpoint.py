#!/usr/bin/env python3
"""
Simple HTTP server that serves checkpoint.npz data as JSON for the transform editor.

Usage:
    python serve_checkpoint.py /path/to/checkpoint.npz [port]

Then open http://localhost:8765/editor in browser
"""

import sys
import json
import ast
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
import urllib.parse
from collections import defaultdict

# Global checkpoint data
CHECKPOINT_DATA = None
CHECKPOINT_PATH = None
TRANSFORM_RELATIONS = None  # Computed on load
PATTERN_EDITS = {}  # pattern_id -> modified pattern data


def load_checkpoint(path: str) -> dict:
    """Load checkpoint.npz and convert to JSON-serializable format."""
    print(f"Loading checkpoint: {path}")
    data = np.load(path, allow_pickle=True)

    # Parse rhythm patterns
    rhythm_patterns = {}
    for i, (pid, phash, pattern) in enumerate(zip(
        data['rhythm_pattern_ids'],
        data['rhythm_pattern_hashes'],
        data['rhythm_patterns']
    )):
        rhythm_patterns[pid] = {
            'id': pid,
            'hash': int(phash),
            'pattern': pattern.tolist() if hasattr(pattern, 'tolist') else list(pattern),
            'length': len(pattern)
        }

    # Parse contour patterns
    contour_patterns = {}
    for i, (pid, phash, pattern) in enumerate(zip(
        data['contour_pattern_ids'],
        data['contour_pattern_hashes'],
        data['contour_patterns']
    )):
        contour_patterns[pid] = {
            'id': pid,
            'hash': int(phash),
            'pattern': pattern.tolist() if hasattr(pattern, 'tolist') else list(pattern),
            'length': len(pattern)
        }

    # Parse objects
    objects_str = data['objects_json'][0]
    objects = ast.literal_eval(objects_str)

    # Compute statistics
    rhythm_usage = {}
    contour_usage = {}
    for obj in objects:
        rid = obj.get('rhythm_id', '')
        cid = obj.get('contour_id', '')
        rhythm_usage[rid] = rhythm_usage.get(rid, 0) + 1
        contour_usage[cid] = contour_usage.get(cid, 0) + 1

    # Add usage counts to patterns
    for pid, count in rhythm_usage.items():
        if pid in rhythm_patterns:
            rhythm_patterns[pid]['usage_count'] = count
    for pid, count in contour_usage.items():
        if pid in contour_patterns:
            contour_patterns[pid]['usage_count'] = count

    print(f"  Loaded {len(rhythm_patterns)} rhythm patterns")
    print(f"  Loaded {len(contour_patterns)} contour patterns")
    print(f"  Loaded {len(objects)} objects")

    # Load transform compositions and edges from checkpoint (if present)
    rhythm_compositions = {}
    contour_compositions = {}
    rhythm_edges = []
    contour_edges = []

    if 'rhythm_compositions_json' in data.files:
        try:
            rhythm_compositions = ast.literal_eval(data['rhythm_compositions_json'][0])
            print(f"  Loaded {len(rhythm_compositions)} rhythm compositions from checkpoint")
        except Exception as e:
            print(f"  Warning: Failed to load rhythm compositions: {e}")

    if 'contour_compositions_json' in data.files:
        try:
            contour_compositions = ast.literal_eval(data['contour_compositions_json'][0])
            print(f"  Loaded {len(contour_compositions)} contour compositions from checkpoint")
        except Exception as e:
            print(f"  Warning: Failed to load contour compositions: {e}")

    if 'rhythm_edges_json' in data.files:
        try:
            rhythm_edges = ast.literal_eval(data['rhythm_edges_json'][0])
            print(f"  Loaded {len(rhythm_edges)} rhythm edges from checkpoint")
        except Exception as e:
            print(f"  Warning: Failed to load rhythm edges: {e}")

    if 'contour_edges_json' in data.files:
        try:
            contour_edges = ast.literal_eval(data['contour_edges_json'][0])
            print(f"  Loaded {len(contour_edges)} contour edges from checkpoint")
        except Exception as e:
            print(f"  Warning: Failed to load contour edges: {e}")

    # Compute piece distribution for each pattern
    rhythm_pieces = defaultdict(set)
    contour_pieces = defaultdict(set)
    for obj in objects:
        piece_id = obj.get('piece_id', '')
        rid = obj.get('rhythm_id', '')
        cid = obj.get('contour_id', '')
        if rid:
            rhythm_pieces[rid].add(piece_id)
        if cid:
            contour_pieces[cid].add(piece_id)

    # Add piece counts to patterns
    for pid in rhythm_patterns:
        rhythm_patterns[pid]['piece_count'] = len(rhythm_pieces.get(pid, set()))
    for pid in contour_patterns:
        contour_patterns[pid]['piece_count'] = len(contour_pieces.get(pid, set()))

    unique_pieces = set(o.get('piece_id', '') for o in objects)

    return {
        'checkpoint_path': path,
        'rhythm_patterns': rhythm_patterns,
        'contour_patterns': contour_patterns,
        'objects': objects,
        'rhythm_compositions': rhythm_compositions,
        'contour_compositions': contour_compositions,
        'rhythm_edges': rhythm_edges,
        'contour_edges': contour_edges,
        'stats': {
            'rhythm_pattern_count': len(rhythm_patterns),
            'contour_pattern_count': len(contour_patterns),
            'object_count': len(objects),
            'unique_pieces': len(unique_pieces),
            'unique_tracks': len(set(o.get('track_id', '') for o in objects)),
            'piece_names': list(unique_pieces)[:50],
            'rhythm_composition_count': len(rhythm_compositions),
            'contour_composition_count': len(contour_compositions),
        }
    }


def compute_transform_relations(patterns: dict, pattern_type: str, limit: int = 200) -> dict:
    """Compute transform relations between top patterns.

    Returns: {
        'edges': [{'from': 'R1', 'to': 'R2', 'transform': 'transpose', 'param': 5}, ...],
        'compositions': {'R3': {'type': 'repeat', 'children': ['R1'], 'param': 2}, ...}
    }
    """
    # Sort by usage and take top patterns
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1].get('usage_count', 0), reverse=True)[:limit]
    pattern_dict = {pid: np.array(pdata['pattern']) for pid, pdata in sorted_patterns}

    edges = []
    compositions = {}

    # Build length index
    by_length = defaultdict(list)
    for pid, parr in pattern_dict.items():
        by_length[len(parr)].append((pid, parr))

    # Check transforms between patterns
    for pid, parr in pattern_dict.items():
        length = len(parr)

        # Check for repeat structure (A = repeat(B, n))
        for n in [2, 3, 4]:
            if length % n == 0:
                chunk_len = length // n
                chunk = parr[:chunk_len]
                is_repeat = all(np.allclose(parr[i*chunk_len:(i+1)*chunk_len], chunk, atol=0.01)
                               for i in range(1, n))
                if is_repeat:
                    # Find matching child pattern
                    for child_pid, child_arr in by_length.get(chunk_len, []):
                        if np.allclose(child_arr, chunk, atol=0.01):
                            compositions[pid] = {
                                'type': 'repeat',
                                'children': [child_pid],
                                'param': n
                            }
                            edges.append({
                                'from': child_pid, 'to': pid,
                                'transform': 'repeat', 'param': n
                            })
                            break
                    break

        # Check for transpose relations (contour only)
        if pattern_type == 'contour' and pid not in compositions:
            for other_pid, other_arr in pattern_dict.items():
                if other_pid == pid or len(other_arr) != length:
                    continue
                diff = parr - other_arr
                if np.std(diff) < 0.01:  # Constant difference
                    shift = int(np.round(np.mean(diff)))
                    if 0 < abs(shift) <= 12:
                        edges.append({
                            'from': other_pid, 'to': pid,
                            'transform': 'transpose', 'param': shift
                        })
                        break

        # Check for retrograde
        if pid not in compositions:
            for other_pid, other_arr in pattern_dict.items():
                if other_pid == pid or len(other_arr) != length:
                    continue
                if np.allclose(parr, other_arr[::-1], atol=0.01):
                    edges.append({
                        'from': other_pid, 'to': pid,
                        'transform': 'retrograde', 'param': 1
                    })
                    break

        # Check for concat (A = B + C)
        if pid not in compositions and length >= 4:
            for split in [length // 2, length // 3, length * 2 // 3]:
                if split < 2 or length - split < 2:
                    continue
                left = parr[:split]
                right = parr[split:]

                left_match = None
                right_match = None

                for cid, carr in by_length.get(split, []):
                    if np.allclose(carr, left, atol=0.01):
                        left_match = cid
                        break

                for cid, carr in by_length.get(length - split, []):
                    if np.allclose(carr, right, atol=0.01):
                        right_match = cid
                        break

                if left_match and right_match and left_match != right_match:
                    compositions[pid] = {
                        'type': 'concat',
                        'children': [left_match, right_match],
                        'param': split
                    }
                    edges.append({
                        'from': left_match, 'to': pid,
                        'transform': 'concat_left', 'param': split
                    })
                    edges.append({
                        'from': right_match, 'to': pid,
                        'transform': 'concat_right', 'param': split
                    })
                    break

    return {
        'edges': edges,
        'compositions': compositions,
        'node_count': len(pattern_dict),
        'edge_count': len(edges)
    }


class CheckpointHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves checkpoint data as JSON."""

    def __init__(self, *args, **kwargs):
        # Set directory to web_interface folder
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_POST(self):
        """Handle POST requests for pattern edits."""
        global PATTERN_EDITS
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/api/pattern_edit':
            pattern_id = data.get('pattern_id')
            pattern_data = data.get('pattern')
            pattern_type = data.get('type', 'rhythm')  # 'rhythm' or 'contour'

            if pattern_id is None or pattern_data is None:
                self.send_error(400, 'Missing pattern_id or pattern')
                return

            # Store the edit
            key = f'{pattern_type}:{pattern_id}'
            PATTERN_EDITS[key] = pattern_data

            # Also update the in-memory checkpoint data so subsequent reads see the edit
            patterns_key = f'{pattern_type}_patterns'
            if CHECKPOINT_DATA and pattern_id in CHECKPOINT_DATA.get(patterns_key, {}):
                CHECKPOINT_DATA[patterns_key][pattern_id]['pattern'] = pattern_data

            self.send_json({
                'success': True,
                'pattern_id': pattern_id,
                'type': pattern_type,
                'edit_count': len(PATTERN_EDITS)
            })

        elif path == '/api/clear_edits':
            PATTERN_EDITS = {}
            self.send_json({'success': True, 'message': 'All edits cleared'})

        else:
            self.send_error(404, 'Unknown POST endpoint')

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/api/checkpoint':
            self.send_checkpoint_data()
        elif path == '/api/checkpoint/stats':
            self.send_checkpoint_stats()
        elif path == '/api/checkpoint/rhythm_patterns':
            self.send_rhythm_patterns()
        elif path == '/api/checkpoint/contour_patterns':
            self.send_contour_patterns()
        elif path.startswith('/api/checkpoint/rhythm/'):
            pattern_id = path.split('/')[-1]
            self.send_pattern('rhythm', pattern_id)
        elif path.startswith('/api/checkpoint/contour/'):
            pattern_id = path.split('/')[-1]
            self.send_pattern('contour', pattern_id)
        elif path == '/api/checkpoint/objects':
            # Send paginated objects
            query = urllib.parse.parse_qs(parsed.query)
            offset = int(query.get('offset', [0])[0])
            limit = int(query.get('limit', [100])[0])
            self.send_objects(offset, limit)
        elif path == '/api/checkpoint/rhythm_transforms':
            self.send_transforms('rhythm')
        elif path == '/api/checkpoint/contour_transforms':
            self.send_transforms('contour')
        elif path == '/editor':
            # Redirect to the editor HTML
            self.send_response(302)
            self.send_header('Location', '/checkpoint_editor.html')
            self.end_headers()
        else:
            # Serve static files
            super().do_GET()

    def send_json(self, data):
        """Send JSON response."""
        content = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)

    def send_checkpoint_data(self):
        """Send full checkpoint (warning: large!)."""
        if CHECKPOINT_DATA is None:
            self.send_error(404, 'No checkpoint loaded')
            return
        # Send a summary, not full data (too large)
        summary = {
            'checkpoint_path': CHECKPOINT_DATA['checkpoint_path'],
            'stats': CHECKPOINT_DATA['stats'],
            'rhythm_pattern_ids': list(CHECKPOINT_DATA['rhythm_patterns'].keys())[:100],
            'contour_pattern_ids': list(CHECKPOINT_DATA['contour_patterns'].keys())[:100],
        }
        self.send_json(summary)

    def send_checkpoint_stats(self):
        """Send checkpoint statistics."""
        if CHECKPOINT_DATA is None:
            self.send_error(404, 'No checkpoint loaded')
            return
        self.send_json(CHECKPOINT_DATA['stats'])

    def send_rhythm_patterns(self):
        """Send all rhythm patterns (just metadata, not full arrays)."""
        if CHECKPOINT_DATA is None:
            self.send_error(404, 'No checkpoint loaded')
            return
        patterns = []
        for pid, pdata in CHECKPOINT_DATA['rhythm_patterns'].items():
            patterns.append({
                'id': pid,
                'length': pdata['length'],
                'usage_count': pdata.get('usage_count', 0),
                'pattern': pdata['pattern']  # Include full pattern for visualization
            })
        # Sort by usage count descending
        patterns.sort(key=lambda x: x['usage_count'], reverse=True)
        self.send_json(patterns)

    def send_contour_patterns(self):
        """Send all contour patterns."""
        if CHECKPOINT_DATA is None:
            self.send_error(404, 'No checkpoint loaded')
            return
        patterns = []
        for pid, pdata in CHECKPOINT_DATA['contour_patterns'].items():
            patterns.append({
                'id': pid,
                'length': pdata['length'],
                'usage_count': pdata.get('usage_count', 0),
                'pattern': pdata['pattern']
            })
        patterns.sort(key=lambda x: x['usage_count'], reverse=True)
        self.send_json(patterns)

    def send_pattern(self, pattern_type, pattern_id):
        """Send a single pattern."""
        if CHECKPOINT_DATA is None:
            self.send_error(404, 'No checkpoint loaded')
            return

        patterns = CHECKPOINT_DATA['rhythm_patterns'] if pattern_type == 'rhythm' else CHECKPOINT_DATA['contour_patterns']
        if pattern_id not in patterns:
            self.send_error(404, f'Pattern {pattern_id} not found')
            return

        pattern_data = patterns[pattern_id]

        # Find objects using this pattern
        key = 'rhythm_id' if pattern_type == 'rhythm' else 'contour_id'
        using_objects = [o for o in CHECKPOINT_DATA['objects'] if o.get(key) == pattern_id][:50]

        response = {
            **pattern_data,
            'sample_objects': using_objects
        }
        self.send_json(response)

    def send_objects(self, offset, limit):
        """Send paginated objects."""
        if CHECKPOINT_DATA is None:
            self.send_error(404, 'No checkpoint loaded')
            return

        objects = CHECKPOINT_DATA['objects']
        total = len(objects)
        page = objects[offset:offset + limit]

        self.send_json({
            'total': total,
            'offset': offset,
            'limit': limit,
            'objects': page
        })

    def send_transforms(self, pattern_type):
        """Send transform relations for patterns - prefer checkpoint data, fallback to compute."""
        global TRANSFORM_RELATIONS
        if CHECKPOINT_DATA is None:
            self.send_error(404, 'No checkpoint loaded')
            return

        cache_key = f'{pattern_type}_transforms'
        if TRANSFORM_RELATIONS is None:
            TRANSFORM_RELATIONS = {}

        if cache_key not in TRANSFORM_RELATIONS:
            # Try to use checkpoint data first
            compositions_key = f'{pattern_type}_compositions'
            edges_key = f'{pattern_type}_edges'

            if CHECKPOINT_DATA.get(compositions_key) or CHECKPOINT_DATA.get(edges_key):
                # Use precomputed data from checkpoint
                compositions = CHECKPOINT_DATA.get(compositions_key, {})
                edges_raw = CHECKPOINT_DATA.get(edges_key, [])

                # Convert edge list format [from, to, transform, param] to dict format
                edges = [
                    {'from': e[0], 'to': e[1], 'transform': e[2], 'param': e[3]}
                    for e in edges_raw
                ]

                print(f"Using {pattern_type} transform relations from checkpoint: {len(edges)} edges, {len(compositions)} compositions")
                TRANSFORM_RELATIONS[cache_key] = {
                    'edges': edges,
                    'compositions': compositions,
                    'node_count': len(CHECKPOINT_DATA[f'{pattern_type}_patterns']),
                    'edge_count': len(edges),
                    'source': 'checkpoint'
                }
            else:
                # Fallback: compute on the fly for older checkpoints
                print(f"Computing {pattern_type} transform relations (no checkpoint data)...")
                patterns = CHECKPOINT_DATA['rhythm_patterns'] if pattern_type == 'rhythm' else CHECKPOINT_DATA['contour_patterns']
                TRANSFORM_RELATIONS[cache_key] = compute_transform_relations(patterns, pattern_type, limit=200)
                TRANSFORM_RELATIONS[cache_key]['source'] = 'computed'
                print(f"  Found {len(TRANSFORM_RELATIONS[cache_key]['edges'])} edges")

        self.send_json(TRANSFORM_RELATIONS[cache_key])


def main():
    if len(sys.argv) < 2:
        print("Usage: python serve_checkpoint.py /path/to/checkpoint.npz [port]")
        print("\nExample:")
        print("  python serve_checkpoint.py /home/arlo/do-repo/midi_generator/factored_v28/checkpoint.npz")
        sys.exit(1)

    global CHECKPOINT_DATA, CHECKPOINT_PATH

    checkpoint_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765

    # Load checkpoint
    CHECKPOINT_PATH = checkpoint_path
    CHECKPOINT_DATA = load_checkpoint(checkpoint_path)

    # Threaded server to handle concurrent requests
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    # Start server
    server = ThreadedHTTPServer(('0.0.0.0', port), CheckpointHandler)
    print(f"\nServer running at http://localhost:{port}")
    print(f"Open http://localhost:{port}/editor to view checkpoint")
    print("Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
