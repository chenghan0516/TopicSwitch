"""
Graph construction functions for extracting key elements and atomic facts.
Refactored to return structured data for LangGraph workflow.
"""
import re
import tiktoken
from typing import List, Dict, Optional
from tqdm import tqdm

from .models import generate_response_llama
from .prompts import EXTRACTION_PROMPT
from .config import MAX_TOKENS_PER_CHUNK
from .state import AtomicFact, GraphNode, KnowledgeGraph


def parse_extraction_output(output_text: str, chunk_id: int) -> List[AtomicFact]:
    """
    Parse LLM output to extract atomic facts and key elements.
    
    Expected format: [Serial Number], [Atomic Facts], [List of Key Elements, separated with '|']
    Example: "1. One day, a father and his little son were going home. | father | little son | going home"
    
    Args:
        output_text: LLM output text
        chunk_id: ID of the chunk these facts came from
        
    Returns:
        List of AtomicFact objects
    """
    atomic_facts = []
    lines = output_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to match format: "1. [fact] | element1 | element2 | ..."
        # Or: "1. [fact]"
        match = re.match(r'^\d+\.\s*(.+?)(?:\s*\|\s*(.+))?$', line)
        if match:
            fact_text = match.group(1).strip()
            elements_text = match.group(2).strip() if match.group(2) else ""
            
            # Parse key elements
            key_elements = [e.strip() for e in elements_text.split('|') if e.strip()]
            
            # If no elements in format, try to extract from fact
            if not key_elements and fact_text:
                # Simple heuristic: extract capitalized words and common nouns
                words = re.findall(r'\b[A-Z][a-z]+\b|\b\w+\b', fact_text)
                key_elements = [w for w in words if len(w) > 3][:5]  # Limit to 5
            
            if fact_text:
                atomic_facts.append(AtomicFact(
                    id=len(atomic_facts),
                    fact=fact_text,
                    key_elements=key_elements,
                    chunk_id=chunk_id
                ))
    
    return atomic_facts


def build_knowledge_graph(
    atomic_facts: List[AtomicFact],
    chunks: List[str]
) -> KnowledgeGraph:
    """
    Build knowledge graph from atomic facts.
    
    Args:
        atomic_facts: List of extracted atomic facts
        chunks: Original text chunks
        
    Returns:
        KnowledgeGraph object
    """
    # Group atomic facts by key elements to create nodes
    element_to_facts: Dict[str, List[AtomicFact]] = {}
    
    for fact in atomic_facts:
        for element in fact.key_elements:
            if element not in element_to_facts:
                element_to_facts[element] = []
            element_to_facts[element].append(fact)
    
    # Create nodes
    nodes: Dict[str, GraphNode] = {}
    for element, facts in element_to_facts.items():
        nodes[element] = GraphNode(
            key_element=element,
            atomic_facts=facts,
            neighbors=[]
        )
    
    # Build neighbor relationships (nodes that share atomic facts)
    node_keys = list(nodes.keys())
    for i, key1 in enumerate(node_keys):
        for key2 in node_keys[i+1:]:
            facts1 = {f.id for f in nodes[key1].atomic_facts}
            facts2 = {f.id for f in nodes[key2].atomic_facts}
            if facts1 & facts2:  # Share at least one atomic fact
                nodes[key1].neighbors.append(key2)
                nodes[key2].neighbors.append(key1)
    
    return KnowledgeGraph(
        nodes=nodes,
        chunks=chunks,
        atomic_facts=atomic_facts
    )


def extract_key_elements_and_atomic_facts(
    text: str,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    extraction_prompt: str = EXTRACTION_PROMPT,
    max_chunks: Optional[int] = None
) -> KnowledgeGraph:
    """
    Extract key elements and atomic facts, then build knowledge graph.
    
    Args:
        text: Input text to extract from
        max_tokens_per_chunk: Maximum tokens per chunk
        extraction_prompt: Prompt for extraction
        max_chunks: Maximum number of chunks to process (None = all)
        
    Returns:
        KnowledgeGraph object
    """
    encoding = tiktoken.encoding_for_model("gpt-4")
    prompt_tokens = encoding.encode(extraction_prompt)
    tokens = encoding.encode(text)
    
    # Break text into chunks
    max_context_tokens_per_chunk = max_tokens_per_chunk - len(prompt_tokens)
    context_chunks = []
    
    for i in range(0, len(tokens), max_context_tokens_per_chunk):
        chunk = tokens[i:i + max_context_tokens_per_chunk]
        if len(chunk) > 0:
            context_chunks.append(encoding.decode(chunk))
    
    # Limit chunks if specified
    if max_chunks:
        context_chunks = context_chunks[:max_chunks]
    
    # Extract atomic facts from each chunk
    all_atomic_facts = []
    print(f"Processing {len(context_chunks)} chunks...")
    
    for chunk_id, context in enumerate(tqdm(context_chunks, desc="Extracting facts")):
        try:
            output_text = generate_response_llama(extraction_prompt, context)
            facts = parse_extraction_output(output_text, chunk_id)
            all_atomic_facts.extend(facts)
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            continue
    
    # Build knowledge graph
    print(f"Building knowledge graph from {len(all_atomic_facts)} atomic facts...")
    graph = build_knowledge_graph(all_atomic_facts, context_chunks)
    print(f"Created graph with {len(graph.nodes)} nodes")
    
    return graph


def extract_key_elements_and_atomic_facts_T5(
    text: str,
    tokenizer,
    model,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    extraction_prompt: str = EXTRACTION_PROMPT
) -> KnowledgeGraph:
    """
    Extract key elements and atomic facts using flan-T5 model (for prototyping).
    
    Args:
        text: Input text to extract from
        tokenizer: Tokenizer for the model
        model: flan-T5 model
        max_tokens_per_chunk: Maximum tokens per chunk
        extraction_prompt: Prompt for extraction
        
    Returns:
        KnowledgeGraph object
    """
    import torch
    
    prompt_tokens = tokenizer.encode(extraction_prompt, return_tensors='pt')
    tokens = tokenizer.encode(text, return_tensors='pt')
    
    # Break text into chunks
    max_context_tokens_per_chunk = max_tokens_per_chunk - len(prompt_tokens[0])
    token_chunks = []
    
    for i in range(0, len(tokens[0]), max_context_tokens_per_chunk):
        chunk = tokens[0][i:i + max_context_tokens_per_chunk].view(1, -1)
        if len(chunk[0]) > 0:
            token_chunks.append(chunk)
    
    # Extract from first 2 chunks as example
    all_atomic_facts = []
    chunks_text = []
    
    for chunk_id, context_tokens in enumerate(token_chunks[:2]):
        input_tokens = torch.cat((prompt_tokens, context_tokens), dim=1)
        outputs = model.generate(input_tokens, max_length=max_tokens_per_chunk)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        chunk_text = tokenizer.decode(context_tokens[0], skip_special_tokens=True)
        chunks_text.append(chunk_text)
        
        facts = parse_extraction_output(output_text, chunk_id)
        all_atomic_facts.extend(facts)
    
    # Build knowledge graph
    graph = build_knowledge_graph(all_atomic_facts, chunks_text)
    return graph
