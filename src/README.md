# Topic Switch Source Code

This directory contains the organized Python modules for the Topic Switch project, which studies proactive interference in multi-turn dialogues using GraphReader-based agents.

## Structure

### `config.py`
Configuration constants and default parameters:
- Random seed settings
- Data processing parameters (sample numbers, intervention settings)
- Model parameters (model names, paths)
- Generation parameters (token limits)

### `data_processing.py`
Functions for processing and creating intervened datasets:
- `load_data()`: Load data from JSON files
- `sample_intervene_data()`: Sample sets of data indices to intervene
- `sample_session()`: Sample sessions to intervene for each data group
- `retrieve_session_data()`: Concatenate session data for each data pair
- `get_string_from_session()`: Convert session messages to formatted strings

### `models.py`
Model loading and generation functions:
- `load_flan_t5_model()`: Load flan-T5 model and tokenizer
- `load_llama_model()`: Load Llama model from HuggingFace
- `generate_response_llama()`: Generate responses using Llama model

### `prompts.py`
All prompt definitions for the GraphReader-based agent system:
- `EXTRACTION_PROMPT`: For extracting key elements and atomic facts
- `RATIONAL_PLAN_PROMPT`: For creating rational plans
- `INITIAL_NODE_PROMPT`: For selecting initial nodes
- `EXPLORE_ATOMIC_PROMPT`: For exploring atomic facts
- `EXPLORE_CHUNK_PROMPT`: For exploring text chunks
- `EXPLORE_NEIGHBOR_PROMPT`: For exploring neighbor nodes
- `QA_PROMPT`: For final question answering

### `graph_construction.py`
Graph construction and knowledge extraction functions:
- `extract_key_elements_and_atomic_facts_T5()`: Extract using flan-T5
- `extract_key_elements_and_atomic_facts()`: Extract using tiktoken and LLM

### `example_usage.py`
Example script demonstrating how to use the modules together.

## Usage

### Basic Usage

```python
from src import data_processing, models, prompts, graph_construction
from src.config import DEFAULT_DATA_PATH

# Load data
data = data_processing.load_data(DEFAULT_DATA_PATH)

# Sample intervention data
intervene_haystack_idx = data_processing.sample_intervene_data(
    len(data), sample_num=1, intervene_data_num=2
)

# Load models
tokenizer, model = models.load_flan_t5_model()

# Use prompts
extraction_prompt = prompts.EXTRACTION_PROMPT
```

### Running the Example

```bash
cd TopicSwitch/src
python example_usage.py
```

## Dependencies

- `torch`
- `transformers`
- `tiktoken`
- `tqdm`
- `json` (standard library)
- `random` (standard library)

## Notes

- The code is organized to be modular and reusable
- Configuration is centralized in `config.py`
- All prompts are defined in `prompts.py` for easy modification
- Model loading functions support both flan-T5 and Llama models

