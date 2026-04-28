"""
Unit tests for the stemphonic_server task-dict janitor.

Imports `sweep_tasks_dict` directly to avoid loading the full server (and
its torch / model dependencies) during CI. The wrapper that mutates the
global `tasks` dict + the background thread are intentionally not under
test — they're thin glue around this pure function.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

# Load just the function we care about, without executing the model-loading
# top-level code in stemphonic_server.py. We do this by reading the source
# and exec-ing only the janitor block in an isolated namespace.
_SERVER = Path(__file__).resolve().parent.parent / "stemphonic_server.py"


def _load_sweep_fn():
    src = _SERVER.read_text()
    # Locate the function block by sentinel; copying bytes verbatim into
    # a fresh module avoids importing torch/onnx/etc. Reads until the
    # next column-0 `def ` (the start of the next top-level function).
    start = src.index("def sweep_tasks_dict(")
    after = src.index("\ndef ", start + 1)
    block = src[start:after]
    ns = {
        "_TERMINAL_STATUSES": {"completed", "failed"},
        "TASK_TTL_SEC": 3600,
        "TASK_MAX_ENTRIES": 5000,
    }
    exec(block, ns)
    return ns["sweep_tasks_dict"]


sweep_tasks_dict = _load_sweep_fn()


class TestTaskJanitor(unittest.TestCase):
    def test_evicts_terminal_past_ttl(self):
        d = {
            "old_done":   {"status": "completed", "created": 0},
            "old_failed": {"status": "failed",    "created": 100},
            "fresh_done": {"status": "completed", "created": 10_000},
            "running":    {"status": "processing", "created": 0},
        }
        n_ttl, n_cap = sweep_tasks_dict(d, now=10_000, ttl_sec=3600)
        self.assertEqual(n_ttl, 2)
        self.assertEqual(n_cap, 0)
        self.assertIn("fresh_done", d)
        self.assertIn("running", d)  # non-terminal preserved even if ancient
        self.assertNotIn("old_done", d)
        self.assertNotIn("old_failed", d)

    def test_hard_cap_drops_oldest(self):
        d = {f"t{i}": {"status": "processing", "created": i} for i in range(10)}
        n_ttl, n_cap = sweep_tasks_dict(d, now=100, ttl_sec=3600, max_entries=5)
        self.assertEqual(n_ttl, 0)
        self.assertEqual(n_cap, 5)
        self.assertEqual(len(d), 5)
        # Oldest five (t0..t4) dropped, newest five (t5..t9) kept.
        for i in range(5):
            self.assertNotIn(f"t{i}", d)
        for i in range(5, 10):
            self.assertIn(f"t{i}", d)

    def test_missing_created_treated_as_ancient(self):
        d = {
            "no_ts_done": {"status": "completed"},  # no `created`
            "fresh":      {"status": "completed", "created": 9_999},
        }
        n_ttl, _ = sweep_tasks_dict(d, now=10_000, ttl_sec=3600)
        self.assertEqual(n_ttl, 1)
        self.assertNotIn("no_ts_done", d)
        self.assertIn("fresh", d)

    def test_empty_dict_noop(self):
        d = {}
        n_ttl, n_cap = sweep_tasks_dict(d, now=0)
        self.assertEqual((n_ttl, n_cap), (0, 0))


if __name__ == "__main__":
    unittest.main()
