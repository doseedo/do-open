"""
Example Database for Adaptive Corpus Learning

Stores analyzed MIDI examples with their predicted parameters and quality scores.
Enables retrieval of similar examples and high-quality references.

This database supports the adaptive learning loop by maintaining a growing
corpus of analyzed examples that can be used for:
1. Training data generation
2. Quality benchmarking
3. Similar example retrieval
4. Progress tracking

Author: Musical Program Synthesis Team
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class MIDIExample:
    """Analyzed MIDI example with predictions and quality."""
    id: Optional[int] = None
    midi_path: str = ""
    predicted_params: Dict[str, Any] = None
    quality: float = 0.0
    iteration: int = 0
    timestamp: str = ""
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.predicted_params is None:
            self.predicted_params = {}
        if self.metadata is None:
            self.metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ExampleDatabase:
    """
    SQLite database for storing analyzed MIDI examples.

    Schema:
        examples: Core table with MIDI analysis results
        parameters: Flattened parameter values for fast querying
        features: High-dimensional feature vectors
    """

    def __init__(self, db_path: str = "data/examples.db"):
        """
        Initialize example database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access

        self._create_tables()

    def _create_tables(self):
        """Create database schema."""

        cursor = self.conn.cursor()

        # Main examples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                midi_path TEXT NOT NULL,
                params TEXT NOT NULL,
                quality REAL NOT NULL,
                iteration INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                UNIQUE(midi_path, iteration)
            )
        """)

        # Index for fast quality queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality
            ON examples(quality DESC)
        """)

        # Index for iteration tracking
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_iteration
            ON examples(iteration DESC)
        """)

        # Features table (separate for performance)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                example_id INTEGER PRIMARY KEY,
                feature_vector BLOB,
                FOREIGN KEY (example_id) REFERENCES examples(id)
            )
        """)

        # Statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER NOT NULL,
                avg_quality REAL,
                num_examples INTEGER,
                num_improvements INTEGER,
                timestamp TEXT NOT NULL
            )
        """)

        self.conn.commit()

    def add(self,
            midi_file: str,
            predicted_params: Dict[str, Any],
            quality: float,
            iteration: int = 0,
            features: Optional[np.ndarray] = None,
            metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add analyzed example to database.

        Args:
            midi_file: Path to MIDI file
            predicted_params: Predicted parameter dictionary
            quality: Quality score (0-1)
            iteration: Learning iteration number
            features: Feature vector (optional)
            metadata: Additional metadata (optional)

        Returns:
            Database ID of inserted example
        """
        cursor = self.conn.cursor()

        # Prepare data
        params_json = json.dumps(predicted_params)
        metadata_json = json.dumps(metadata or {})
        timestamp = datetime.now().isoformat()

        # Insert example
        cursor.execute("""
            INSERT OR REPLACE INTO examples
            (midi_path, params, quality, iteration, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(midi_file), params_json, quality, iteration, timestamp, metadata_json))

        example_id = cursor.lastrowid

        # Store features if provided
        if features is not None:
            feature_blob = features.tobytes()
            cursor.execute("""
                INSERT OR REPLACE INTO features (example_id, feature_vector)
                VALUES (?, ?)
            """, (example_id, feature_blob))

        self.conn.commit()
        return example_id

    def get_by_id(self, example_id: int) -> Optional[MIDIExample]:
        """Get example by database ID."""
        cursor = self.conn.cursor()

        row = cursor.execute("""
            SELECT * FROM examples WHERE id = ?
        """, (example_id,)).fetchone()

        if row is None:
            return None

        return self._row_to_example(row)

    def get_by_quality(self,
                      min_quality: float = 0.8,
                      limit: int = 100) -> List[MIDIExample]:
        """
        Get high-quality examples.

        Args:
            min_quality: Minimum quality threshold
            limit: Maximum number of examples

        Returns:
            List of high-quality examples
        """
        cursor = self.conn.cursor()

        rows = cursor.execute("""
            SELECT * FROM examples
            WHERE quality >= ?
            ORDER BY quality DESC
            LIMIT ?
        """, (min_quality, limit)).fetchall()

        return [self._row_to_example(row) for row in rows]

    def get_by_iteration(self, iteration: int) -> List[MIDIExample]:
        """Get all examples from specific iteration."""
        cursor = self.conn.cursor()

        rows = cursor.execute("""
            SELECT * FROM examples WHERE iteration = ?
        """, (iteration,)).fetchall()

        return [self._row_to_example(row) for row in rows]

    def get_latest_iteration(self) -> int:
        """Get the latest iteration number."""
        cursor = self.conn.cursor()

        result = cursor.execute("""
            SELECT MAX(iteration) FROM examples
        """).fetchone()

        return result[0] if result[0] is not None else 0

    def query_similar(self,
                     target_params: Dict[str, Any],
                     k: int = 10,
                     min_quality: float = 0.0) -> List[MIDIExample]:
        """
        Find similar examples based on parameter similarity.

        Uses cosine similarity on parameter vectors.

        Args:
            target_params: Target parameter dictionary
            k: Number of similar examples to return
            min_quality: Minimum quality threshold

        Returns:
            List of similar examples ordered by similarity
        """
        cursor = self.conn.cursor()

        # Get all examples above quality threshold
        rows = cursor.execute("""
            SELECT * FROM examples WHERE quality >= ?
        """, (min_quality,)).fetchall()

        if not rows:
            return []

        # Convert target params to vector
        target_vector = self._params_to_vector(target_params)

        # Compute similarities
        similarities = []
        for row in rows:
            example = self._row_to_example(row)
            example_vector = self._params_to_vector(example.predicted_params)

            # Cosine similarity
            similarity = self._cosine_similarity(target_vector, example_vector)
            similarities.append((similarity, example))

        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [example for _, example in similarities[:k]]

    def get_statistics(self, iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Get database statistics.

        Args:
            iteration: Specific iteration (None for overall)

        Returns:
            Statistics dictionary
        """
        cursor = self.conn.cursor()

        if iteration is not None:
            query = """
                SELECT
                    COUNT(*) as num_examples,
                    AVG(quality) as avg_quality,
                    MIN(quality) as min_quality,
                    MAX(quality) as max_quality
                FROM examples
                WHERE iteration = ?
            """
            result = cursor.execute(query, (iteration,)).fetchone()
        else:
            query = """
                SELECT
                    COUNT(*) as num_examples,
                    AVG(quality) as avg_quality,
                    MIN(quality) as min_quality,
                    MAX(quality) as max_quality,
                    MAX(iteration) as latest_iteration
                FROM examples
            """
            result = cursor.execute(query).fetchone()

        return dict(result)

    def record_iteration_stats(self,
                              iteration: int,
                              avg_quality: float,
                              num_examples: int,
                              num_improvements: int):
        """Record statistics for an iteration."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO statistics
            (iteration, avg_quality, num_examples, num_improvements, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (iteration, avg_quality, num_examples, num_improvements,
              datetime.now().isoformat()))

        self.conn.commit()

    def get_improvement_history(self) -> List[Dict[str, Any]]:
        """Get history of quality improvements across iterations."""
        cursor = self.conn.cursor()

        rows = cursor.execute("""
            SELECT * FROM statistics ORDER BY iteration ASC
        """).fetchall()

        return [dict(row) for row in rows]

    def export_high_quality_examples(self,
                                    output_file: str,
                                    min_quality: float = 0.9):
        """
        Export high-quality examples to JSON file.

        Args:
            output_file: Output file path
            min_quality: Minimum quality threshold
        """
        examples = self.get_by_quality(min_quality, limit=1000)

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'min_quality': min_quality,
            'num_examples': len(examples),
            'examples': [asdict(ex) for ex in examples]
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported {len(examples)} high-quality examples to {output_file}")

    def _row_to_example(self, row: sqlite3.Row) -> MIDIExample:
        """Convert database row to MIDIExample."""
        return MIDIExample(
            id=row['id'],
            midi_path=row['midi_path'],
            predicted_params=json.loads(row['params']),
            quality=row['quality'],
            iteration=row['iteration'],
            timestamp=row['timestamp'],
            metadata=json.loads(row.get('metadata', '{}'))
        )

    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Convert parameter dictionary to vector for similarity computation.

        Handles categorical, numeric, and boolean parameters.
        """
        # Get all unique parameter names from database
        # For now, use a simple approach: sort keys and vectorize

        sorted_keys = sorted(params.keys())
        vector = []

        for key in sorted_keys:
            value = params[key]

            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple hash-based encoding for categorical
                vector.append(hash(value) % 1000 / 1000.0)
            else:
                vector.append(0.0)

        return np.array(vector)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if len(v1) != len(v2):
            # Pad shorter vector
            max_len = max(len(v1), len(v2))
            v1 = np.pad(v1, (0, max_len - len(v1)))
            v2 = np.pad(v2, (0, max_len - len(v2)))

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(v1, v2) / (norm1 * norm2)

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function
def create_example_database(db_path: str = "data/examples.db") -> ExampleDatabase:
    """Create and initialize example database."""
    return ExampleDatabase(db_path)


if __name__ == "__main__":
    # Test database
    print("Testing ExampleDatabase...")

    db = ExampleDatabase("test_examples.db")

    # Add test example
    test_params = {
        'harmony.chord_density': 0.75,
        'melody.note_density': 0.65,
        'rhythm.swing.amount': 0.5
    }

    example_id = db.add(
        "test.mid",
        test_params,
        quality=0.85,
        iteration=1
    )

    print(f"Added example with ID: {example_id}")

    # Query
    high_quality = db.get_by_quality(min_quality=0.8)
    print(f"Found {len(high_quality)} high-quality examples")

    # Statistics
    stats = db.get_statistics()
    print(f"Database statistics: {stats}")

    db.close()
    print("Test complete!")
