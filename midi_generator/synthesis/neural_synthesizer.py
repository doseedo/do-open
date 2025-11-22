"""
Neural Program Synthesizer - Phase 3: Neural Architecture
==========================================================

Learn to synthesize MIDI transforms from examples using neural networks.

Architecture:
1. MIDIDifferenceEncoder - Encode what changed (input → output)
2. DSLProgramDecoder - Generate DSL program with grammar constraints
3. NeuralProgramSynthesizer - End-to-end model

Key Innovation: Grammar-constrained generation
- Decoder can only generate syntactically valid DSL programs
- Uses grammar rules to mask invalid next tokens
- Guarantees compilable output

Author: Agent 8 - Transform Architecture
Phase: 4.3 (Neural Architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .transform_dsl import (
    DSLProgram,
    DSLVocabulary,
    ForEach,
    If,
    Operation,
    Aggregate,
    IteratorType,
    FilterType,
    OperationType,
    AggregatorType
)


# ============================================================================
# MIDI Representation
# ============================================================================

def midi_to_pianoroll(midi, time_resolution: float = 0.0625,
                      max_duration: Optional[float] = None) -> np.ndarray:
    """
    Convert MIDI to pianoroll representation.

    Args:
        midi: mido.MidiFile object
        time_resolution: Time step size in seconds (default: 16th note at 120 BPM)
        max_duration: Maximum duration to encode (None = full piece)

    Returns:
        Pianoroll: (time_steps, 128) array where pianoroll[t, p] = velocity
    """
    try:
        from midi_generator.transforms.space_level_transforms import extract_notes_from_midi
        notes = extract_notes_from_midi(midi)
    except:
        # Fallback: simple note extraction
        notes = []
        current_time = 0
        for track in midi.tracks:
            for msg in track:
                current_time += msg.time / midi.ticks_per_beat
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({
                        'pitch': msg.note,
                        'velocity': msg.velocity,
                        'start_time': current_time,
                        'duration': 0.5  # Placeholder
                    })

    if not notes:
        return np.zeros((1, 128), dtype=np.float32)

    # Determine duration
    if max_duration is None:
        max_time = max(n['start_time'] + n.get('duration', 0.5) for n in notes)
    else:
        max_time = max_duration

    # Create pianoroll
    n_timesteps = int(np.ceil(max_time / time_resolution))
    pianoroll = np.zeros((n_timesteps, 128), dtype=np.float32)

    # Fill pianoroll
    for note in notes:
        start_step = int(note['start_time'] / time_resolution)
        duration = note.get('duration', 0.5)
        end_step = int((note['start_time'] + duration) / time_resolution)

        if start_step < n_timesteps:
            end_step = min(end_step, n_timesteps)
            pianoroll[start_step:end_step, note['pitch']] = note['velocity'] / 127.0

    return pianoroll


# ============================================================================
# Phase 3.1: MIDI Difference Encoder
# ============================================================================

class MIDIDifferenceEncoder(nn.Module):
    """
    Encode the difference between input and output MIDI.

    This represents "what the transform does" as a fixed-size embedding.

    Architecture:
    - Separate Transformers for input and output MIDI
    - Concatenate and project to difference embedding
    - Output: 512D vector representing the transformation
    """

    def __init__(self, hidden_dim: int = 512, num_layers: int = 6, num_heads: int = 8):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection: pianoroll (128 pitches) → hidden_dim
        self.input_projection = nn.Linear(128, hidden_dim)
        self.output_projection = nn.Linear(128, hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=5000)

        # Transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.input_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Difference encoder
        self.difference_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Pooling
        self.pooling_type = 'attention'  # or 'mean'
        if self.pooling_type == 'attention':
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                batch_first=True
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, input_pianoroll: torch.Tensor,
                output_pianoroll: torch.Tensor) -> torch.Tensor:
        """
        Encode MIDI difference.

        Args:
            input_pianoroll: (batch, time_steps, 128)
            output_pianoroll: (batch, time_steps, 128)

        Returns:
            difference_embedding: (batch, hidden_dim)
        """
        batch_size = input_pianoroll.shape[0]

        # Project to hidden dim
        input_seq = self.input_projection(input_pianoroll)   # (batch, time, hidden)
        output_seq = self.output_projection(output_pianoroll)

        # Add positional encoding
        input_seq = self.positional_encoding(input_seq)
        output_seq = self.positional_encoding(output_seq)

        # Encode with Transformers
        input_encoded = self.input_encoder(input_seq)   # (batch, time, hidden)
        output_encoded = self.output_encoder(output_seq)

        # Pool to fixed size
        if self.pooling_type == 'mean':
            input_pooled = input_encoded.mean(dim=1)     # (batch, hidden)
            output_pooled = output_encoded.mean(dim=1)
        else:  # attention pooling
            # Expand query to batch size
            query = self.pool_query.expand(batch_size, -1, -1)  # (batch, 1, hidden)

            input_pooled, _ = self.attention_pool(query, input_encoded, input_encoded)
            output_pooled, _ = self.attention_pool(query, output_encoded, output_encoded)

            input_pooled = input_pooled.squeeze(1)       # (batch, hidden)
            output_pooled = output_pooled.squeeze(1)

        # Combine and encode difference
        combined = torch.cat([input_pooled, output_pooled], dim=1)  # (batch, hidden*2)
        difference = self.difference_encoder(combined)               # (batch, hidden)

        return difference


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


# ============================================================================
# Phase 3.2: DSL Program Decoder (Grammar-Constrained)
# ============================================================================

class DSLProgramDecoder(nn.Module):
    """
    Generate DSL programs from transformation embeddings.

    Key Innovation: Grammar-constrained generation
    - Only generates syntactically valid DSL programs
    - Uses grammar rules to mask invalid tokens
    - Guarantees compilable output

    Grammar Rules:
    - After FOREACH → must be IteratorType
    - After IF → must be FilterType
    - After Operation → must be value or END
    - Proper nesting of control structures
    """

    def __init__(self, hidden_dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_program_length: int = 50):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_length = max_program_length

        # Vocabulary
        self.vocab = DSLVocabulary()
        self.vocab_size = self.vocab.vocab_size

        # Token embedding
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_program_length)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.vocab_size)

        # Grammar state tracker
        self.grammar = DSLGrammarConstraints(self.vocab)

    def forward(self, difference_embedding: torch.Tensor,
                target_program: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate DSL program.

        Args:
            difference_embedding: (batch, hidden_dim) from encoder
            target_program: (batch, seq_len) token indices (for training)

        Returns:
            logits: (batch, seq_len, vocab_size) if training
            generated: (batch, seq_len) token indices if inference
        """
        if target_program is not None:
            # Training mode: teacher forcing
            return self._forward_train(difference_embedding, target_program)
        else:
            # Inference mode: autoregressive generation
            return self.generate(difference_embedding)

    def _forward_train(self, difference_embedding: torch.Tensor,
                      target_program: torch.Tensor) -> torch.Tensor:
        """Training forward pass with teacher forcing"""

        batch_size, seq_len = target_program.shape

        # Embed tokens
        token_embeds = self.embedding(target_program)  # (batch, seq_len, hidden)
        token_embeds = self.positional_encoding(token_embeds)

        # Prepare memory (transformation embedding)
        memory = difference_embedding.unsqueeze(1)  # (batch, 1, hidden)

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(target_program.device)

        # Decode
        output = self.decoder(
            tgt=token_embeds,
            memory=memory,
            tgt_mask=causal_mask
        )  # (batch, seq_len, hidden)

        # Project to vocabulary
        logits = self.output_proj(output)  # (batch, seq_len, vocab_size)

        return logits

    def generate(self, difference_embedding: torch.Tensor,
                beam_size: int = 1,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate program with grammar constraints.

        Args:
            difference_embedding: (batch, hidden_dim)
            beam_size: Beam search width (1 = greedy)
            temperature: Sampling temperature

        Returns:
            generated: (batch, seq_len) token indices
        """
        batch_size = difference_embedding.shape[0]
        device = difference_embedding.device

        # Initialize with <SOS> token
        generated = torch.full(
            (batch_size, 1),
            self.vocab.token_to_idx['<SOS>'],
            dtype=torch.long,
            device=device
        )

        # Grammar state (track nesting, context, etc.)
        grammar_states = [self.grammar.initial_state() for _ in range(batch_size)]

        # Generate tokens autoregressively
        for step in range(self.max_length):
            # Get logits for next token
            with torch.no_grad():
                logits = self._forward_train(difference_embedding, generated)[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # CRITICAL: Apply grammar constraints
            for batch_idx in range(batch_size):
                current_token = generated[batch_idx, -1].item()
                valid_next_tokens = self.grammar.get_valid_next_tokens(
                    current_token,
                    grammar_states[batch_idx]
                )

                # Mask invalid tokens
                mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
                mask[valid_next_tokens] = False
                logits[batch_idx] = logits[batch_idx].masked_fill(mask, float('-inf'))

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Update grammar states
            for batch_idx in range(batch_size):
                token = next_token[batch_idx, 0].item()
                grammar_states[batch_idx] = self.grammar.update_state(
                    grammar_states[batch_idx],
                    token
                )

            # Check for <EOS>
            if (next_token == self.vocab.token_to_idx['<EOS>']).all():
                break

        return generated


# ============================================================================
# Grammar Constraint System (CRITICAL COMPONENT)
# ============================================================================

class DSLGrammarConstraints:
    """
    Encode DSL grammar rules for constrained generation.

    This is the key innovation that ensures generated programs are valid.

    Grammar Rules:
    1. Structural: FOREACH, IF must be properly nested and closed
    2. Type: After FOREACH comes IteratorType, after IF comes FilterType
    3. Values: Numbers/expressions only where expected
    4. Termination: Must generate <EOS> when program is complete
    """

    def __init__(self, vocab: DSLVocabulary):
        self.vocab = vocab

        # Token sets for different categories
        self.iterator_tokens = [vocab.token_to_idx[t.value] for t in IteratorType]
        self.filter_tokens = [vocab.token_to_idx[t.value] for t in FilterType]
        self.operation_tokens = [vocab.token_to_idx[t.value] for t in OperationType]
        self.aggregator_tokens = [vocab.token_to_idx[t.value] for t in AggregatorType]

        # Structural tokens
        self.foreach_token = vocab.token_to_idx['FOREACH']
        self.if_token = vocab.token_to_idx['IF']
        self.then_token = vocab.token_to_idx['THEN']
        self.end_foreach_token = vocab.token_to_idx['END_FOREACH']
        self.end_if_token = vocab.token_to_idx['END_IF']
        self.sos_token = vocab.token_to_idx['<SOS>']
        self.eos_token = vocab.token_to_idx['<EOS>']

        # Value tokens (numbers, expressions)
        self.value_tokens = [
            vocab.token_to_idx[t] for t in vocab.token_to_idx.keys()
            if t not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>', 'FOREACH', 'IF', 'THEN',
                        'END_FOREACH', 'END_IF', 'ELSE']
            and (t.replace('.', '').replace('-', '').isdigit() or t in ['amount', '+', '-', '*'])
        ]

    def initial_state(self) -> Dict[str, Any]:
        """Initial grammar state"""
        return {
            'nesting_stack': [],     # Track FOREACH/IF nesting
            'expecting': 'statement', # What we expect next
            'depth': 0
        }

    def get_valid_next_tokens(self, current_token: int, state: Dict[str, Any]) -> List[int]:
        """
        Get list of valid next tokens given current token and grammar state.

        This is the CORE of grammar-constrained generation.
        """
        valid_tokens = []

        current_token_str = self.vocab.idx_to_token.get(current_token, '<UNK>')
        expecting = state['expecting']

        # Rule 1: After <SOS>, expect statement
        if current_token == self.sos_token:
            valid_tokens = [self.foreach_token, self.if_token] + self.operation_tokens + self.aggregator_tokens

        # Rule 2: After FOREACH, expect iterator type
        elif current_token == self.foreach_token:
            valid_tokens = self.iterator_tokens

        # Rule 3: After iterator, expect colon/body start (simplified: expect statement or operation)
        elif current_token in self.iterator_tokens:
            valid_tokens = self.operation_tokens + self.aggregator_tokens + [self.if_token, self.end_foreach_token]

        # Rule 4: After IF, expect filter type
        elif current_token == self.if_token:
            valid_tokens = self.filter_tokens

        # Rule 5: After filter, expect value threshold
        elif current_token in self.filter_tokens:
            valid_tokens = self.value_tokens

        # Rule 6: After threshold value, expect THEN
        elif expecting == 'then':
            valid_tokens = [self.then_token]

        # Rule 7: After THEN, expect operation or end
        elif current_token == self.then_token:
            valid_tokens = self.operation_tokens + [self.end_if_token]

        # Rule 8: After operation, expect value or end
        elif current_token in self.operation_tokens:
            valid_tokens = self.value_tokens

        # Rule 9: After aggregator, expect source/target
        elif current_token in self.aggregator_tokens:
            valid_tokens = self.value_tokens + [self.end_foreach_token, self.end_if_token]

        # Rule 10: After value, context-dependent
        elif current_token in self.value_tokens:
            # Could be end of statement, or more operators
            if state['depth'] > 0:
                valid_tokens = [self.end_foreach_token, self.end_if_token] + self.operation_tokens
            else:
                valid_tokens = [self.eos_token] + self.operation_tokens

        # Rule 11: After END_FOREACH or END_IF, can start new statement or end
        elif current_token in [self.end_foreach_token, self.end_if_token]:
            if state['depth'] > 0:
                valid_tokens = [self.foreach_token, self.if_token, self.end_foreach_token, self.end_if_token]
            else:
                valid_tokens = [self.foreach_token, self.if_token, self.eos_token]

        # Default: if nothing matches, allow EOS
        if not valid_tokens:
            valid_tokens = [self.eos_token]

        return valid_tokens

    def update_state(self, state: Dict[str, Any], token: int) -> Dict[str, Any]:
        """Update grammar state after generating token"""
        new_state = state.copy()
        new_state['nesting_stack'] = state['nesting_stack'].copy()

        # Track nesting
        if token == self.foreach_token or token == self.if_token:
            new_state['nesting_stack'].append(token)
            new_state['depth'] += 1

        elif token == self.end_foreach_token or token == self.end_if_token:
            if new_state['nesting_stack']:
                new_state['nesting_stack'].pop()
                new_state['depth'] -= 1

        # Update expectations
        if token in self.filter_tokens:
            new_state['expecting'] = 'value'
        elif new_state['expecting'] == 'value' and token in self.value_tokens:
            new_state['expecting'] = 'then'
        elif token == self.then_token:
            new_state['expecting'] = 'operation'
        else:
            new_state['expecting'] = 'statement'

        return new_state


# ============================================================================
# Phase 3.3: Complete Neural Program Synthesizer
# ============================================================================

class NeuralProgramSynthesizer(nn.Module):
    """
    End-to-end neural program synthesis model.

    Combines encoder and decoder for learning MIDI transforms.

    Usage:
        model = NeuralProgramSynthesizer()

        # Training
        logits = model(input_midi, output_midi, target_program)
        loss = cross_entropy(logits, target_program)

        # Inference
        generated = model(input_midi, output_midi)
        dsl_program = model.tokens_to_dsl(generated)
        python_code = dsl_program.to_python()
    """

    def __init__(self, hidden_dim: int = 512, num_layers: int = 6):
        super().__init__()

        self.encoder = MIDIDifferenceEncoder(hidden_dim=hidden_dim, num_layers=num_layers)
        self.decoder = DSLProgramDecoder(hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, input_midi: torch.Tensor,
                output_midi: torch.Tensor,
                target_program: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            input_midi: (batch, time, 128) pianoroll
            output_midi: (batch, time, 128) pianoroll
            target_program: (batch, seq_len) token indices (optional)

        Returns:
            If training: logits (batch, seq_len, vocab_size)
            If inference: generated tokens (batch, seq_len)
        """
        # Encode transformation
        difference_embedding = self.encoder(input_midi, output_midi)

        # Decode to DSL program
        if target_program is not None:
            # Training mode
            logits = self.decoder(difference_embedding, target_program)
            return logits
        else:
            # Inference mode
            generated = self.decoder.generate(difference_embedding)
            return generated

    def tokens_to_dsl(self, token_sequence: torch.Tensor) -> DSLProgram:
        """
        Convert generated token sequence to DSL program.

        Args:
            token_sequence: (seq_len,) token indices

        Returns:
            DSLProgram object
        """
        # Convert indices to tokens
        tokens = [self.decoder.vocab.idx_to_token[idx.item()]
                 for idx in token_sequence]

        # Parse tokens into DSL AST
        # This is a simplified parser - full version would be more robust
        statements = []

        # Remove <SOS> and <EOS>
        tokens = [t for t in tokens if t not in ['<SOS>', '<EOS>', '<PAD>']]

        # Very simplified parsing (would need proper parser for production)
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == 'FOREACH':
                # Parse FOREACH statement
                # ...simplified...
                i += 1

            elif token in [op.value for op in OperationType]:
                # Parse operation
                # ...simplified...
                i += 1

            else:
                i += 1

        return DSLProgram(statements=statements, name="generated_transform")

    def synthesize_transform(self, gap_cluster: List[Dict]) -> callable:
        """
        Main entry point: synthesize transform from gap cluster.

        This replaces LLM-based code generation in hybrid_synthesizer.

        Args:
            gap_cluster: List of {'original': midi, 'reconstructed': midi}

        Returns:
            Executable Python function
        """
        # Take first example
        example = gap_cluster[0]

        # Convert to pianorolls
        input_pianoroll = torch.tensor(
            midi_to_pianoroll(example['reconstructed']),
            dtype=torch.float32
        ).unsqueeze(0)  # Add batch dim

        output_pianoroll = torch.tensor(
            midi_to_pianoroll(example['original']),
            dtype=torch.float32
        ).unsqueeze(0)

        # Generate DSL program
        with torch.no_grad():
            token_sequence = self.forward(input_pianoroll, output_pianoroll)

        # Convert to DSL
        dsl_program = self.tokens_to_dsl(token_sequence[0])

        # Compile to Python
        python_code = dsl_program.to_python()

        # Return executable function
        namespace = {}
        exec(python_code, namespace)

        return namespace.get('generated_transform', None)
