"""
LLM-Powered Code Generation Module
===================================

This module contains LLM-powered agents for code generation and parameter proposals.

Modules:
- code_generator: Agent 12 - LLM Code Generation Agent

Author: Musical Program Synthesis Team
License: MIT
"""

from .code_generator import (
    LLMCodeGenerationAgent,
    CodebaseIndex,
    CodePatternLibrary,
    GeneratedImplementation,
    CodeSection
)

__all__ = [
    'LLMCodeGenerationAgent',
    'CodebaseIndex',
    'CodePatternLibrary',
    'GeneratedImplementation',
    'CodeSection'
]
