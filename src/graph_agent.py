"""
LangGraph-based graph exploration agent for answering questions.
Implements the GraphReader workflow with unbinding mechanism for PI mitigation.
"""
import re
from typing import Literal
from langgraph.graph import StateGraph, END

from .state import GraphExplorationState, create_initial_state
from .models import generate_response_llama
from .prompts import (
    RATIONAL_PLAN_PROMPT,
    INITIAL_NODE_PROMPT,
    EXPLORE_ATOMIC_PROMPT,
    EXPLORE_CHUNK_PROMPT,
    EXPLORE_NEIGHBOR_PROMPT,
    QA_PROMPT
)


def create_rational_plan(state: GraphExplorationState) -> GraphExplorationState:
    """Create a rational plan for answering the question."""
    question = state["question"]
    
    plan = generate_response_llama(
        RATIONAL_PLAN_PROMPT,
        question
    )
    
    state["rational_plan"] = plan
    state["previous_actions"].append("create_rational_plan")
    return state


def select_initial_nodes(state: GraphExplorationState) -> GraphExplorationState:
    """Select initial nodes from the knowledge graph."""
    question = state["question"]
    plan = state["rational_plan"]
    graph = state["knowledge_graph"]
    
    # Get all node key elements
    node_list = "\n".join([f"- {key}" for key in graph.nodes.keys()])
    
    # Format prompt
    prompt_input = f"""Question: {question}
Plan: {plan}
Nodes:
{node_list}"""
    
    response = generate_response_llama(INITIAL_NODE_PROMPT, prompt_input)
    
    # Parse selected nodes with scores
    selected_nodes = []
    for line in response.split('\n'):
        match = re.search(r'Node:\s*(.+?),\s*Score:\s*(\d+)', line)
        if match:
            node = match.group(1).strip()
            score = int(match.group(2))
            if node in graph.nodes and score >= 50:  # Threshold for relevance
                selected_nodes.append(node)
    
    # Ensure at least some nodes are selected
    if not selected_nodes and graph.nodes:
        # Fallback: select top nodes by number of atomic facts
        sorted_nodes = sorted(
            graph.nodes.items(),
            key=lambda x: len(x[1].atomic_facts),
            reverse=True
        )
        selected_nodes = [node.key_element for node, _ in sorted_nodes[:10]]
    
    state["initial_nodes"] = selected_nodes[:10]  # Limit to 10
    state["previous_actions"].append(f"select_initial_nodes: {len(selected_nodes)} nodes")
    return state


def explore_atomic_facts(state: GraphExplorationState) -> GraphExplorationState:
    """Explore atomic facts of current node."""
    question = state["question"]
    plan = state["rational_plan"]
    graph = state["knowledge_graph"]
    current_node_key = state["current_node"]
    
    if not current_node_key or current_node_key not in graph.nodes:
        return state
    
    node = graph.nodes[current_node_key]
    notebook = "\n".join(state["notebooks"]) if state["notebooks"] else "No previous notes."
    previous_actions = "\n".join(state["previous_actions"][-5:])  # Last 5 actions
    
    # Format atomic facts with chunk IDs
    atomic_facts_text = []
    for fact in node.atomic_facts:
        atomic_facts_text.append(
            f"Chunk {fact.chunk_id}: {fact.fact} | Key Elements: {', '.join(fact.key_elements)}"
        )
    
    prompt_input = f"""Question: {question}
Plan: {plan}
Previous Actions:
{previous_actions}
Notebook Content:
{notebook}
Current Node: {current_node_key}
Atomic Facts:
{chr(10).join(atomic_facts_text)}"""
    
    response = generate_response_llama(EXPLORE_ATOMIC_PROMPT, prompt_input)
    
    # Parse response
    updated_notebook = ""
    chosen_action = ""
    
    if "*Updated Notebook*" in response:
        notebook_match = re.search(
            r'\*Updated Notebook\*:\s*(.+?)(?=\*|$)', 
            response, 
            re.DOTALL
        )
        if notebook_match:
            updated_notebook = notebook_match.group(1).strip()
    
    if "*Chosen Action*" in response:
        action_match = re.search(
            r'\*Chosen Action\*:\s*(.+?)(?=\*|$)', 
            response, 
            re.DOTALL
        )
        if action_match:
            chosen_action = action_match.group(1).strip()
    
    # Update notebook
    if updated_notebook:
        state["notebooks"].append(updated_notebook)
    
    state["last_action"] = chosen_action
    state["previous_actions"].append(f"explore_atomic: {current_node_key}")
    
    return state


def explore_chunk(state: GraphExplorationState) -> GraphExplorationState:
    """Explore a text chunk."""
    question = state["question"]
    plan = state["rational_plan"]
    graph = state["knowledge_graph"]
    chunk_id = state["current_chunk_id"]
    
    if chunk_id is None or chunk_id >= len(graph.chunks):
        return state
    
    chunk_text = graph.chunks[chunk_id]
    notebook = "\n".join(state["notebooks"]) if state["notebooks"] else "No previous notes."
    previous_actions = "\n".join(state["previous_actions"][-5:])
    
    prompt_input = f"""Question: {question}
Plan: {plan}
Previous Actions:
{previous_actions}
Notebook Content:
{notebook}
Current Text Chunk (ID: {chunk_id}):
{chunk_text}"""
    
    response = generate_response_llama(EXPLORE_CHUNK_PROMPT, prompt_input)
    
    # Parse response
    updated_notebook = ""
    chosen_action = ""
    
    if "*Updated Notebook*" in response:
        notebook_match = re.search(
            r'\*Updated Notebook\*:\s*(.+?)(?=\*|$)', 
            response, 
            re.DOTALL
        )
        if notebook_match:
            updated_notebook = notebook_match.group(1).strip()
    
    if "*Chosen Action*" in response:
        action_match = re.search(
            r'\*Chosen Action\*:\s*(.+?)(?=\*|$)', 
            response, 
            re.DOTALL
        )
        if action_match:
            chosen_action = action_match.group(1).strip()
    
    # Update notebook and mark chunk as visited
    if updated_notebook:
        state["notebooks"].append(updated_notebook)
    
    visited_chunks = state.get("visited_chunks", [])
    if chunk_id not in visited_chunks:
        visited_chunks.append(chunk_id)
    state["visited_chunks"] = visited_chunks
    state["last_action"] = chosen_action
    state["previous_actions"].append(f"explore_chunk: {chunk_id}")
    
    return state


def explore_neighbor(state: GraphExplorationState) -> GraphExplorationState:
    """Explore neighboring nodes."""
    question = state["question"]
    plan = state["rational_plan"]
    graph = state["knowledge_graph"]
    current_node_key = state["current_node"]
    
    if not current_node_key or current_node_key not in graph.nodes:
        return state
    
    node = graph.nodes[current_node_key]
    notebook = "\n".join(state["notebooks"]) if state["notebooks"] else "No previous notes."
    previous_actions = "\n".join(state["previous_actions"][-5:])
    
    # Filter out visited and irrelevant neighbors (convert lists to sets for comparison)
    visited_set = set(state.get("visited_nodes", []))
    irrelevant_set = set(state.get("irrelevant_nodes", []))
    available_neighbors = [
        n for n in node.neighbors 
        if n not in visited_set and n not in irrelevant_set
    ]
    
    if not available_neighbors:
        state["last_action"] = "termination()"
        return state
    
    neighbors_text = "\n".join([f"- {n}" for n in available_neighbors])
    
    prompt_input = f"""Question: {question}
Plan: {plan}
Previous Actions:
{previous_actions}
Notebook Content:
{notebook}
Current Node: {current_node_key}
Neighbor Nodes:
{neighbors_text}"""
    
    response = generate_response_llama(EXPLORE_NEIGHBOR_PROMPT, prompt_input)
    
    # Parse response
    chosen_action = ""
    if "*Chosen Action*" in response:
        action_match = re.search(
            r'\*Chosen Action\*:\s*(.+?)(?=\*|$)', 
            response, 
            re.DOTALL
        )
        if action_match:
            chosen_action = action_match.group(1).strip()
    
    state["last_action"] = chosen_action
    state["previous_actions"].append(f"explore_neighbor: {current_node_key}")
    
    return state


def generate_final_answer(state: GraphExplorationState) -> GraphExplorationState:
    """Generate final answer from notebooks."""
    question = state["question"]
    notebooks = state["notebooks"]
    
    notebooks_text = "\n".join([
        f"{i+1}. {notebook}" 
        for i, notebook in enumerate(notebooks)
    ])
    
    prompt_input = f"""Question: {question}
Notebook of different exploration paths:
{notebooks_text}"""
    
    response = generate_response_llama(QA_PROMPT, prompt_input)
    
    # Extract final answer
    if "Final answer:" in response:
        answer_match = re.search(
            r'Final answer:\s*(.+?)(?=\n\n|$)', 
            response, 
            re.DOTALL
        )
        if answer_match:
            state["final_answer"] = answer_match.group(1).strip()
        else:
            state["final_answer"] = response
    else:
        state["final_answer"] = response
    
    return state


def route_after_atomic(state: GraphExplorationState) -> Literal["read_chunk", "read_neighbor", "terminate"]:
    """Route after exploring atomic facts."""
    action = state.get("last_action", "")
    
    if "read_chunk" in action.lower():
        # Parse chunk IDs and set current_chunk_id
        chunk_ids = parse_chunk_ids(action)
        if chunk_ids:
            state["current_chunk_id"] = chunk_ids[0]  # Use first chunk
        return "read_chunk"
    elif "stop_and_read_neighbor" in action.lower() or "read_neighbor" in action.lower():
        return "read_neighbor"
    else:
        return "terminate"


def route_after_chunk(state: GraphExplorationState) -> Literal["search_more", "read_previous", "read_subsequent", "terminate"]:
    """Route after exploring chunk."""
    action = state.get("last_action", "")
    current_chunk_id = state.get("current_chunk_id")
    graph = state.get("knowledge_graph")
    
    if "read_previous_chunk" in action.lower():
        if current_chunk_id is not None and current_chunk_id > 0:
            state["current_chunk_id"] = current_chunk_id - 1
        return "read_previous"
    elif "read_subsequent_chunk" in action.lower():
        if current_chunk_id is not None and graph and current_chunk_id < len(graph.chunks) - 1:
            state["current_chunk_id"] = current_chunk_id + 1
        return "read_subsequent"
    elif "termination" in action.lower() or "terminate" in action.lower():
        return "terminate"
    else:
        return "search_more"


def route_after_neighbor(state: GraphExplorationState) -> Literal["read_neighbor_node", "terminate"]:
    """Route after exploring neighbors."""
    action = state.get("last_action", "")
    
    if "read_neighbor_node" in action.lower():
        # Parse neighbor node and set as current
        neighbor = parse_neighbor_node(action)
        if neighbor:
            state["current_node"] = neighbor
            visited_nodes = state.get("visited_nodes", [])
            if neighbor not in visited_nodes:
                visited_nodes.append(neighbor)
            state["visited_nodes"] = visited_nodes
        return "read_neighbor_node"
    else:
        return "terminate"


def parse_chunk_ids(action: str) -> list[int]:
    """Parse chunk IDs from action string."""
    # Look for read_chunk([1, 2, 3]) or read_chunk(1, 2, 3)
    match = re.search(r'read_chunk\((.+?)\)', action)
    if match:
        ids_str = match.group(1)
        # Try to parse as list or comma-separated
        try:
            ids = eval(ids_str)  # Safe for simple lists
            if isinstance(ids, list):
                return [int(id) for id in ids]
            elif isinstance(ids, int):
                return [ids]
        except:
            # Fallback: extract numbers
            ids = re.findall(r'\d+', ids_str)
            return [int(id) for id in ids]
    return []


def parse_neighbor_node(action: str) -> str:
    """Parse neighbor node from action string."""
    match = re.search(r'read_neighbor_node\((.+?)\)', action)
    if match:
        return match.group(1).strip().strip('"').strip("'")
    return ""


def apply_unbinding(state: GraphExplorationState) -> GraphExplorationState:
    """
    Apply unbinding mechanism to mitigate proactive interference.
    Marks nodes from previous topics as irrelevant.
    """
    # Simple heuristic: if we've switched topics significantly, mark old nodes as irrelevant
    # This is a placeholder - can be enhanced with topic detection
    current_node = state.get("current_node")
    visited_nodes = state.get("visited_nodes", set())
    
    # If we've visited many nodes but current exploration is on a different path,
    # mark some early nodes as potentially irrelevant
    if len(visited_nodes) > 5 and current_node:
        # Mark nodes that are far from current exploration as potentially irrelevant
        # This is a simplified version - can be enhanced with topic similarity
        pass
    
    return state


def create_graph_agent() -> StateGraph:
    """Create the LangGraph workflow for graph exploration."""
    
    workflow = StateGraph(GraphExplorationState)
    
    # Add nodes
    workflow.add_node("create_plan", create_rational_plan)
    workflow.add_node("select_nodes", select_initial_nodes)
    workflow.add_node("explore_atomic", explore_atomic_facts)
    workflow.add_node("explore_chunk", explore_chunk)
    workflow.add_node("explore_neighbor", explore_neighbor)
    workflow.add_node("generate_answer", generate_final_answer)
    workflow.add_node("apply_unbinding", apply_unbinding)
    
    # Set entry point
    workflow.set_entry_point("create_plan")
    
    # Add edges
    workflow.add_edge("create_plan", "select_nodes")
    workflow.add_edge("select_nodes", "explore_atomic")
    
    # Conditional routing after atomic facts
    workflow.add_conditional_edges(
        "explore_atomic",
        route_after_atomic,
        {
            "read_chunk": "explore_chunk",
            "read_neighbor": "explore_neighbor",
            "terminate": "apply_unbinding"
        }
    )
    
    # Conditional routing after chunk
    workflow.add_conditional_edges(
        "explore_chunk",
        route_after_chunk,
        {
            "search_more": "explore_atomic",
            "read_previous": "explore_chunk",
            "read_subsequent": "explore_chunk",
            "terminate": "apply_unbinding"
        }
    )
    
    # Conditional routing after neighbor
    workflow.add_conditional_edges(
        "explore_neighbor",
        route_after_neighbor,
        {
            "read_neighbor_node": "explore_atomic",
            "terminate": "apply_unbinding"
        }
    )
    
    # Apply unbinding before generating answer
    workflow.add_edge("apply_unbinding", "generate_answer")
    
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()


def run_graph_agent(
    question: str,
    knowledge_graph,
    max_iterations: int = 20
) -> GraphExplorationState:
    """
    Run the graph exploration agent.
    
    Args:
        question: Question to answer
        knowledge_graph: KnowledgeGraph object
        max_iterations: Maximum number of iterations
        
    Returns:
        Final state with answer
    """
    from .state import create_initial_state
    
    state = create_initial_state(question, knowledge_graph)
    agent = create_graph_agent()
    
    # Run agent with iteration limit
    config = {"recursion_limit": max_iterations}
    
    try:
        final_state = agent.invoke(state, config)
        
        # Set first node to explore if we have initial nodes but haven't started
        if final_state.get("initial_nodes") and not final_state.get("current_node"):
            if final_state["initial_nodes"]:
                final_state["current_node"] = final_state["initial_nodes"][0]
                final_state["visited_nodes"].add(final_state["initial_nodes"][0])
    except Exception as e:
        print(f"Error during agent execution: {e}")
        # Return state even if there was an error
        final_state = state
    
    return final_state

