"""
Causal Structure for Musical Parameters

Documents causal dependencies between parameters to enable:
1. More accurate predictions (respecting music theory)
2. Better interpretability (understand why parameters were chosen)
3. Efficient inference (compute in causal order)

Author: Musical Program Synthesis Team
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import networkx as nx


class CausalRelationType(Enum):
    """Type of causal relationship."""
    CAUSES = "causes"           # A → B (A directly causes B)
    CAUSED_BY = "caused_by"     # A ← B (A is caused by B)
    CORRELATES = "correlates"   # A ↔ B (mutual influence, no clear direction)


@dataclass
class CausalRelation:
    """A causal relationship between parameters."""
    source: str                 # Parameter that causes
    target: str                 # Parameter that is caused
    strength: float             # 0-1, how strong is the causal effect
    mechanism: str              # Music theory explanation
    bidirectional: bool = False # Whether influence goes both ways


# Causal Structure Definition
CAUSAL_STRUCTURE = {
    # ========================================
    # GENRE → Everything (Root Cause)
    # ========================================
    'style.genre': {
        'causes': [
            'harmony.complexity',
            'melody.complexity',
            'rhythm.complexity',
            'instrumentation.ensemble_type',
            'tempo.bpm',
            'structure.form_type'
        ],
        'caused_by': [],
        'mechanism': 'Genre determines the stylistic framework for all other parameters',
        'examples': {
            'jazz': 'High harmony complexity, swing rhythm, extended chords',
            'classical': 'Complex structure, wider dynamic range, specific forms',
            'electronic': 'High rhythm complexity, synthesized timbres, repetitive structure'
        }
    },

    # ========================================
    # HARMONY (Causal Chain)
    # ========================================
    'harmony.chord_density': {
        'causes': [
            'harmony.voicing.spread',      # High density → tight voicings
            'melody.note_density',         # High density → simpler melody
            'texture.voice_count'          # High density → fuller texture
        ],
        'caused_by': [
            'style.genre',                 # Genre determines typical density
            'harmony.complexity',          # Complexity allows higher density
            'tempo.bpm'                    # Faster tempo → lower density
        ],
        'mechanism': 'Chord density constrains voicing options and melodic complexity',
        'strength': 0.8
    },

    'harmony.voicing.spread': {
        'causes': [
            'instrumentation.register_distribution',  # Wide spread → distributed registers
            'texture.fullness'                        # Spread voicing → fuller sound
        ],
        'caused_by': [
            'harmony.chord_density',                  # Low density allows wide spread
            'instrumentation.ensemble_type'           # Orchestra → wide spread, piano → close
        ],
        'mechanism': 'Voicing spread affects register usage and texture',
        'strength': 0.7
    },

    'harmony.chromaticism': {
        'causes': [
            'melody.chromaticism',         # Harmonic chromaticism → melodic chromaticism
            'harmony.tension'              # Chromaticism creates tension
        ],
        'caused_by': [
            'style.genre',                 # Jazz → high chromaticism
            'harmony.complexity',          # Complex harmony allows chromaticism
            'structure.section'            # Bridge/development → more chromatic
        ],
        'mechanism': 'Chromatic harmony enables chromatic melody and creates tension',
        'strength': 0.9
    },

    'harmony.modal_mixture': {
        'causes': [
            'harmony.color',               # Modal mixture creates color
            'melody.scale_usage'           # Determines available scale tones
        ],
        'caused_by': [
            'style.genre',                 # Blues/rock → heavy mixture
            'harmony.complexity'           # Requires harmonic sophistication
        ],
        'mechanism': 'Borrowing from parallel modes adds harmonic color',
        'strength': 0.6
    },

    # ========================================
    # MELODY (Follows Harmony)
    # ========================================
    'melody.note_density': {
        'causes': [
            'melody.articulation',         # High density → legato articulation
            'rhythm.subdivision'           # High density → finer subdivisions
        ],
        'caused_by': [
            'harmony.chord_density',       # Low harmonic density allows high melodic density
            'tempo.bpm',                   # Faster tempo → lower note density
            'melody.complexity'            # Complexity allows higher density
        ],
        'mechanism': 'Melodic density must balance with harmonic activity',
        'strength': 0.8
    },

    'melody.contour.shape': {
        'causes': [
            'melody.climax_position',      # Contour determines climax location
            'dynamics.phrase_shape'        # Contour affects dynamic shaping
        ],
        'caused_by': [
            'structure.phrase_position',   # Phrase position affects contour
            'melody.complexity'            # Complexity allows varied contours
        ],
        'mechanism': 'Melodic contour creates phrase shape and affects dynamics',
        'strength': 0.6
    },

    'melody.chromaticism': {
        'causes': [
            'melody.tension'               # Chromaticism creates melodic tension
        ],
        'caused_by': [
            'harmony.chromaticism',        # Follows harmonic chromaticism
            'melody.complexity'            # Requires melodic sophistication
        ],
        'mechanism': 'Melodic chromaticism follows and reinforces harmonic chromaticism',
        'strength': 0.85
    },

    # ========================================
    # RHYTHM (Independent but Influenced)
    # ========================================
    'tempo.bpm': {
        'causes': [
            'rhythm.subdivision',          # Slower tempo → finer subdivisions
            'melody.note_density',         # Slower tempo → higher note density
            'harmony.chord_density',       # Slower tempo → higher chord density
            'dynamics.articulation'        # Slower tempo → more articulation detail
        ],
        'caused_by': [
            'style.genre',                 # Genre determines typical tempo
            'structure.section'            # Intro/outro often slower
        ],
        'mechanism': 'Tempo constrains density and subdivision across all domains',
        'strength': 0.9
    },

    'rhythm.swing.amount': {
        'causes': [
            'rhythm.feel',                 # Swing creates laid-back feel
            'melody.articulation'          # Swing affects note articulation
        ],
        'caused_by': [
            'style.genre',                 # Jazz → swing, classical → straight
            'tempo.bpm'                    # Slower tempo → more swing
        ],
        'mechanism': 'Swing feel affects rhythmic interpretation',
        'strength': 0.7
    },

    'rhythm.syncopation': {
        'causes': [
            'rhythm.complexity',           # Syncopation increases complexity
            'dynamics.accent_placement'    # Syncopation affects accents
        ],
        'caused_by': [
            'style.genre',                 # Jazz/funk → high syncopation
            'rhythm.complexity'            # Requires rhythmic sophistication
        ],
        'mechanism': 'Syncopation creates rhythmic interest and complexity',
        'strength': 0.75
    },

    # ========================================
    # DYNAMICS (Affects Expression)
    # ========================================
    'dynamics.overall_level': {
        'causes': [
            'dynamics.range',              # Overall level constrains range
            'instrumentation.balance'      # Level affects section balance
        ],
        'caused_by': [
            'style.genre',                 # Classical → wide range, rock → compressed
            'structure.section',           # Climax → louder
            'instrumentation.ensemble_type' # Orchestra → wide range, solo → compressed
        ],
        'mechanism': 'Overall dynamic level sets the expressive range',
        'strength': 0.7
    },

    'dynamics.variation': {
        'causes': [
            'dynamics.expression',         # Variation creates expression
            'texture.interest'             # Dynamic variation adds interest
        ],
        'caused_by': [
            'style.genre',                 # Classical → high variation
            'melody.contour.shape',        # Contour affects dynamic shape
            'structure.phrase_position'    # Phrases have dynamic arcs
        ],
        'mechanism': 'Dynamic variation creates expressive phrasing',
        'strength': 0.8
    },

    # ========================================
    # TEXTURE (Emergent from Other Domains)
    # ========================================
    'texture.voice_count': {
        'causes': [
            'texture.density',             # More voices → denser texture
            'harmony.voicing.options'      # Voice count constrains voicing
        ],
        'caused_by': [
            'instrumentation.ensemble_type', # Orchestra → many voices
            'harmony.chord_density',       # High density → more voices
            'style.genre'                  # Baroque → many voices, minimalism → few
        ],
        'mechanism': 'Number of voices determines texture density',
        'strength': 0.85
    },

    # ========================================
    # INSTRUMENTATION (Contextual Constraints)
    # ========================================
    'instrumentation.ensemble_type': {
        'causes': [
            'instrumentation.voicing.options',     # Ensemble determines voicing possibilities
            'harmony.voicing.spread',              # Ensemble affects spread
            'dynamics.range',                      # Ensemble determines dynamic range
            'texture.voice_count'                  # Ensemble sets voice count
        ],
        'caused_by': [
            'style.genre'                          # Genre suggests typical ensembles
        ],
        'mechanism': 'Ensemble type constrains all performance parameters',
        'strength': 0.95
    },

    # ========================================
    # STRUCTURE (Temporal Constraints)
    # ========================================
    'structure.section': {
        'causes': [
            'harmony.complexity',          # Development → more complex
            'melody.complexity',           # Development → more complex
            'dynamics.overall_level',      # Climax → louder
            'harmony.chromaticism'         # Bridge → more chromatic
        ],
        'caused_by': [
            'structure.form_type'          # Form determines section sequence
        ],
        'mechanism': 'Formal section determines local parameter values',
        'strength': 0.8
    }
}


class CausalParameterGraph:
    """
    Directed acyclic graph (DAG) of causal parameter relationships.

    Enables:
    1. Topological ordering for efficient inference
    2. Identification of root causes (genre)
    3. Understanding parameter dependencies
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Build causal graph from CAUSAL_STRUCTURE."""
        for param, relations in CAUSAL_STRUCTURE.items():
            # Add node
            self.graph.add_node(param)

            # Add causal edges
            for caused_param in relations.get('causes', []):
                strength = relations.get('strength', 0.5)
                mechanism = relations.get('mechanism', 'Unknown')

                self.graph.add_edge(
                    param,
                    caused_param,
                    weight=strength,
                    mechanism=mechanism
                )

    def get_causal_order(self) -> List[str]:
        """
        Get topological ordering of parameters for inference.

        Returns parameters in causal order: predict parents before children.
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # Cycle detected - return best-effort ordering
            return list(self.graph.nodes())

    def get_root_causes(self) -> List[str]:
        """Get parameters with no parents (root causes)."""
        return [
            node for node in self.graph.nodes()
            if self.graph.in_degree(node) == 0
        ]

    def get_children(self, param: str) -> List[str]:
        """Get direct children (parameters caused by this one)."""
        return list(self.graph.successors(param))

    def get_parents(self, param: str) -> List[str]:
        """Get direct parents (parameters that cause this one)."""
        return list(self.graph.predecessors(param))

    def get_ancestors(self, param: str) -> Set[str]:
        """Get all ancestors (all parameters that influence this one)."""
        return nx.ancestors(self.graph, param)

    def get_descendants(self, param: str) -> Set[str]:
        """Get all descendants (all parameters influenced by this one)."""
        return nx.descendants(self.graph, param)

    def get_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get causal path from source to target."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def explain_relationship(self, source: str, target: str) -> Optional[str]:
        """Explain causal relationship between two parameters."""
        path = self.get_path(source, target)

        if not path:
            return None

        if len(path) == 2:
            # Direct relationship
            edge_data = self.graph.get_edge_data(source, target)
            return edge_data.get('mechanism', 'Direct causal relationship')
        else:
            # Indirect relationship
            mechanisms = []
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                mechanisms.append(edge_data.get('mechanism', 'Unknown'))

            return " → ".join(mechanisms)

    def visualize(self, output_path: str = "causal_graph.png"):
        """Visualize causal graph (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph)
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color='lightblue',
                node_size=3000,
                arrowsize=20,
                font_size=8
            )

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Causal graph saved to {output_path}")

        except ImportError:
            print("Warning: matplotlib not available for visualization")


def predict_with_causal_order(features, models) -> Dict[str, any]:
    """
    Predict parameters in causal order.

    This is MORE ACCURATE than predicting all parameters independently
    because it respects causal dependencies.

    Args:
        features: 1000-dimensional feature vector
        models: Dict of parameter_name → trained model

    Returns:
        Dict of parameter predictions
    """
    causal_graph = CausalParameterGraph()
    causal_order = causal_graph.get_causal_order()

    predictions = {}

    # Predict in causal order
    for param_name in causal_order:
        if param_name not in models:
            continue

        # Get parent predictions (already computed)
        parents = causal_graph.get_parents(param_name)
        parent_values = {p: predictions.get(p) for p in parents}

        # Predict conditioned on parents
        model = models[param_name]

        if parent_values:
            # Conditional prediction
            pred = model.predict_conditional(features, parent_values)
        else:
            # Unconditional prediction (root cause)
            pred = model.predict(features)

        predictions[param_name] = pred

    return predictions


# Example usage
if __name__ == "__main__":
    # Build causal graph
    graph = CausalParameterGraph()

    print("ROOT CAUSES (no parents):")
    print(graph.get_root_causes())

    print("\nCAUSAL ORDER (for inference):")
    order = graph.get_causal_order()
    for i, param in enumerate(order[:20]):
        print(f"{i+1}. {param}")

    print("\nEXPLAIN: harmony.chord_density → melody.note_density")
    explanation = graph.explain_relationship(
        'harmony.chord_density',
        'melody.note_density'
    )
    print(explanation)

    print("\nANCESTORS of melody.chromaticism:")
    ancestors = graph.get_ancestors('melody.chromaticism')
    print(ancestors)
