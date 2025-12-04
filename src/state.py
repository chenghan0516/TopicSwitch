"""
State schema for LangGraph-based graph exploration agent.
"""
from typing import TypedDict, List, Dict, Set, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AtomicFact:
    """Represents an atomic fact extracted from text."""
    id: int
    fact: str
    key_elements: List[str]
    chunk_id: int  # Which text chunk this fact came from


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    key_element: str  # The key element (noun, verb, adjective)
    atomic_facts: List[AtomicFact]  # Associated atomic facts
    neighbors: List[str] = field(default_factory=list)  # Neighbor node key elements


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure."""
    nodes: Dict[str, GraphNode]  # key_element -> GraphNode
    chunks: List[str]  # Original text chunks
    atomic_facts: List[AtomicFact]  # All atomic facts


class GraphExplorationState(TypedDict, total=False):
    """State for LangGraph exploration workflow."""
    # Input
    question: str
    knowledge_graph: KnowledgeGraph
    
    # Planning
    rational_plan: Optional[str]
    
    # Exploration
    initial_nodes: List[str]  # Selected starting nodes
    current_node: Optional[str]  # Current node being explored
    current_chunk_id: Optional[int]  # Current chunk being read
    
    # Memory
    notebooks: List[str]  # Notebook entries from different exploration paths
    visited_nodes: List[str]  # Nodes already visited (as list for TypedDict compatibility)
    visited_chunks: List[int]  # Chunks already read (as list for TypedDict compatibility)
    
    # Actions
    previous_actions: List[str]  # History of actions taken
    last_action: Optional[str]  # Most recent action
    
    # Output
    final_answer: Optional[str]
    
    # Unbinding (for PI mitigation)
    irrelevant_nodes: List[str]  # Nodes to unbind/ignore (as list for TypedDict compatibility)
    topic_context: Optional[str]  # Current topic context for unbinding


def create_initial_state(
    question: str,
    knowledge_graph: KnowledgeGraph
) -> GraphExplorationState:
    """Create initial state for graph exploration."""
    return {
        "question": question,
        "knowledge_graph": knowledge_graph,
        "rational_plan": None,
        "initial_nodes": [],
        "current_node": None,
        "current_chunk_id": None,
        "notebooks": [],
        "visited_nodes": [],
        "visited_chunks": [],
        "previous_actions": [],
        "last_action": None,
        "final_answer": None,
        "irrelevant_nodes": [],
        "topic_context": None
    }

