"""
Topic Switch - A project for studying proactive interference in multi-turn dialogues.

This package provides tools for:
- Data processing and intervention
- Model loading and generation
- Graph construction for knowledge extraction
- Prompt management for GraphReader-based agents
"""

from . import config
from . import data_processing
from . import models
from . import prompts
from . import graph_construction

__all__ = [
    'config',
    'data_processing',
    'models',
    'prompts',
    'graph_construction',
]

