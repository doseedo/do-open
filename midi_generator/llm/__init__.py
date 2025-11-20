"""
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
