"""
<<<<<<< HEAD
LLM-Powered Music Generation Components
========================================

This module contains LLM-based agents for self-expanding music generation:
- Parameter proposal using Claude API
- Gap analysis interpretation
- Musical knowledge integration
- Code generation suggestions

Author: Agent 11 - Parameter Proposal Agent
License: MIT
"""

from .parameter_proposer import (
    LLMParameterProposalAgent,
    ParameterProposal,
    ProposalValidator,
    ProposalHistory
)

__all__ = [
    'LLMParameterProposalAgent',
    'ParameterProposal',
    'ProposalValidator',
    'ProposalHistory'
]
=======
LLM Module - Agent 12
=====================

LLM-powered code generation for automatic parameter expansion.

This module contains:
- LLMCodeGenerationAgent: Generates implementation code for new parameters
- CodebaseIndex: Indexes and searches the codebase
- CodeValidator: Validates generated code

Author: Agent 12 - Code Generation Agent
License: MIT
"""

from .code_generator import LLMCodeGenerationAgent

__all__ = ['LLMCodeGenerationAgent']
>>>>>>> origin/claude/music-generation-agents-016iuqojwjedj9QM4JT8NZWY
