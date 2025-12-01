"""
Model loading and generation functions.
"""
import torch
import transformers
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .config import (
    FLAN_T5_MODEL_NAME,
    LLAMA_MODEL_ID,
    LLAMA_SAVE_PATH,
    MAX_NEW_TOKENS
)


def load_flan_t5_model(
    model_name: str = FLAN_T5_MODEL_NAME
) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Load flan-T5 model and tokenizer for prototyping.
    
    Args:
        model_name: Name of the flan-T5 model to load
        
    Returns:
        Tuple of (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def load_llama_model(
    model_id: str = LLAMA_MODEL_ID,
    save_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto"
) -> transformers.AutoModelForCausalLM:
    """
    Load Llama model from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID
        save_path: Optional path to save the model after loading
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        
    Returns:
        Loaded Llama model
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    if save_path:
        model.save_pretrained(save_path)
    
    return model


def generate_response_llama(
    prompt: str,
    input_text: str,
    model_id: str = LLAMA_MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto"
) -> str:
    """
    Generate a response using Llama model.
    
    Args:
        prompt: System prompt
        input_text: User input text
        model_id: HuggingFace model ID
        max_new_tokens: Maximum number of new tokens to generate
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        
    Returns:
        Generated response text
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch_dtype},
        device_map=device_map,
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_text},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens,
    )
    
    return outputs[0]["generated_text"][-1]

