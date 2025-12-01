"""
Graph construction functions for extracting key elements and atomic facts.
"""
import torch
import tiktoken
from typing import Optional
from tqdm import trange

from .models import load_flan_t5_model, generate_response_llama
from .prompts import EXTRACTION_PROMPT
from .config import MAX_TOKENS_PER_CHUNK


def extract_key_elements_and_atomic_facts_T5(
    text: str,
    tokenizer,
    model,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    extraction_prompt: str = EXTRACTION_PROMPT
) -> str:
    """
    Extract key elements and atomic facts using flan-T5 model.
    
    Args:
        text: Input text to extract from
        tokenizer: Tokenizer for the model
        model: flan-T5 model
        max_tokens_per_chunk: Maximum tokens per chunk
        extraction_prompt: Prompt for extraction
        
    Returns:
        Extracted key elements and atomic facts as string
    """
    prompt_tokens = tokenizer.encode(extraction_prompt, return_tensors='pt')
    tokens = tokenizer.encode(text, return_tensors='pt')
    
    # Break text into chunks with max_tokens_per_chunk
    max_context_tokens_per_chunk = max_tokens_per_chunk - len(prompt_tokens[0])
    token_chunks = []
    
    for i in range(0, len(tokens[0]), max_context_tokens_per_chunk):
        chunk = tokens[0][i:i + max_context_tokens_per_chunk].view(1, -1)
        if len(chunk[0]) > 0:
            token_chunks.append(chunk)
    
    results = []
    for context_tokens in token_chunks[:2]:  # Process first 2 chunks as example
        print(f"Processing chunk with {context_tokens.shape}")
        print(f"Prompt tokens: {prompt_tokens.shape}")
        input_tokens = torch.cat((prompt_tokens, context_tokens), dim=1)
        print(f"Input tokens: {input_tokens.shape}")
        
        outputs = model.generate(input_tokens, max_length=max_tokens_per_chunk)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(output_text)
        results.append(output_text)
    
    # Placeholder for actual implementation
    return "Extracted Key Elements and Atomic Facts"


def extract_key_elements_and_atomic_facts(
    text: str,
    tokenizer=None,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    extraction_prompt: str = EXTRACTION_PROMPT
) -> str:
    """
    Extract key elements and atomic facts using tiktoken and LLM generation.
    
    Args:
        text: Input text to extract from
        tokenizer: Optional tokenizer (not used, kept for compatibility)
        max_tokens_per_chunk: Maximum tokens per chunk
        extraction_prompt: Prompt for extraction
        
    Returns:
        Extracted key elements and atomic facts as string
    """
    encoding = tiktoken.encoding_for_model("gpt-4")
    prompt_tokens = encoding.encode(extraction_prompt)
    tokens = encoding.encode(text)
    
    # Break text into chunks with max_tokens_per_chunk
    max_context_tokens_per_chunk = max_tokens_per_chunk - len(prompt_tokens)
    context_chunks = []
    
    for i in range(0, len(tokens), max_context_tokens_per_chunk):
        chunk = tokens[i:i + max_context_tokens_per_chunk]
        if len(chunk) > 0:
            context_chunks.append(encoding.decode(chunk))
    
    results = []
    for context in context_chunks[:2]:  # Process first 2 chunks as example
        output_text = generate_response_llama(extraction_prompt, context)
        print(output_text)
        results.append(output_text)
    
    # Placeholder for actual implementation
    return "Extracted Key Elements and Atomic Facts"

