"""
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
