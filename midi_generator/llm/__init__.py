"""
LLM-Powered Music Generation Module - Agents 11 & 12
====================================================

This module contains LLM-powered agents for self-expanding music generation.

Agents:
- Agent 11: Parameter Proposal Agent (using Claude API)
- Agent 12: Code Generation Agent (automatic implementation)

Features:
---------
1. **Parameter Proposal** (Agent 11):
   - Analyzes musical gaps and proposes new parameters
   - Musical knowledge integration
   - Validates proposals against existing parameters
   - Tracks proposal history

2. **Code Generation** (Agent 12):
   - Generates implementation code for new parameters
   - Indexes and searches codebase
   - Pattern-based code generation
   - Code validation and integration

Authors: Agent 11 (Parameter Proposal) & Agent 12 (Code Generation)
License: MIT
"""

# Agent 11: Parameter Proposal
from .parameter_proposer import (
    LLMParameterProposalAgent,
    ParameterProposal,
    ProposalValidator,
    ProposalHistory
)

# Agent 12: Code Generation
from .code_generator import (
    LLMCodeGenerationAgent,
    CodebaseIndex,
    CodePatternLibrary,
    GeneratedImplementation,
    CodeSection
)

__all__ = [
    # Agent 11: Parameter proposal
    'LLMParameterProposalAgent',
    'ParameterProposal',
    'ProposalValidator',
    'ProposalHistory',

    # Agent 12: Code generation
    'LLMCodeGenerationAgent',
    'CodebaseIndex',
    'CodePatternLibrary',
    'GeneratedImplementation',
    'CodeSection'
]

__version__ = '1.0.0'
__authors__ = 'Agent 11 & Agent 12'
