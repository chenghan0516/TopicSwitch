# Topic Switch: Proactive Interference in Multi-Turn Dialogues

A research project studying **Proactive Interference (PI)** in multi-turn dialogues using GraphReader-based agents with LangGraph. This project adapts the Interference Endurance Score (IES) for natural dialogue scenarios and implements unbinding mechanisms to mitigate PI.

## 📋 Overview

Users in multi-turn dialogues frequently change subjects or return to previous ones, particularly when explaining complex topics. Current LLM-based agents tend to get sidetracked by irrelevant details, creating challenges in tracking conversational context and maintaining coherent interactions.

**Proactive Interference** refers to the difficulty in recalling new information due to interference from previously learned information. This project investigates how graph-based agents can mitigate PI by:
1. Structuring dialogue context into knowledge graphs
2. Implementing unbinding mechanisms to filter irrelevant information
3. Adapting IES metrics to measure PI in natural dialogues

## 🎯 Key Features

- **Intervened Dataset Creation**: Synthesizes multi-topic dialogues from LongMemEval
- **Knowledge Graph Construction**: Extracts atomic facts and key elements from text
- **Graph-Based Agent**: LangGraph-powered agent for intelligent graph exploration
- **Unbinding Mechanism**: Topic-aware filtering to reduce proactive interference
- **IES Evaluation**: Adapted Interference Endurance Score for dialogue evaluation

## 📁 Project Structure

```
TopicSwitch/
├── main.py                          # Main entry point with CLI commands
├── src/                             # Source code modules
│   ├── config.py                    # Configuration constants
│   ├── data_processing.py           # Dataset creation and intervention
│   ├── graph_construction.py       # Knowledge graph building
│   ├── graph_agent.py              # LangGraph agent workflow
│   ├── state.py                     # State management for agent
│   ├── models.py                    # Model loading (Llama, flan-T5)
│   ├── prompts.py                   # All prompt definitions
│   └── README.md                    # Detailed module documentation
├── ref/                             # Reference materials
│   ├── papers/                      # Research papers
│   └── sample_intervened_data.txt   # Example data
├── IMPLEMENTATION_PLAN.md           # Detailed implementation roadmap
├── MIGRATION_REPORT.md              # Migration notes
└── Topic_Shift_Proposal__revised_ver__.pdf  # Original proposal
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for model inference)
- HuggingFace account and access token (for Llama models)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd TopicSwitch
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch transformers langgraph langchain tiktoken tqdm
   ```

4. **Configure model access** (for Llama models):
   - Set up HuggingFace token: `huggingface-cli login`
   - Or set environment variable: `export HF_TOKEN=your_token`

5. **Download LongMemEval dataset**:
   - Place the dataset JSON file in the data directory
   - Update `DEFAULT_DATA_PATH` in `src/config.py` if needed

## 💻 Usage

The project provides a command-line interface through `main.py` with five main commands:

### 1. Prepare Data (`prepare_data`)

Creates intervened datasets by concatenating multiple conversation threads to simulate topic switching.

```bash
python main.py prepare_data \
    --data_path data/longmemeval/longmemeval_s_cleaned.json \
    --output_dir output/ \
    --sample_num 1 \
    --intervene_num 2 \
    --session_num 2 \
    --limit_groups 10
```

**Outputs:**
- `intervened_texts.txt`: Concatenated dialogue texts
- `metadata.json`: Intervention metadata
- `qa_pairs.json`: Question-answer pairs

### 2. Build Graph (`build_graph`)

Extracts atomic facts and key elements, then constructs a knowledge graph.

```bash
python main.py build_graph \
    --text_file output/intervened_texts.txt \
    --output_dir output/ \
    --max_chunks 10 \
    --graph_id 0
```

**Alternative** (with direct text):
```bash
python main.py build_graph \
    --text_content "Your dialogue text here..." \
    --output_dir output/
```

**Outputs:**
- `graph.pkl` or `graph_{id}.pkl`: Pickled knowledge graph

### 3. Run Agent (`run_agent`)

Executes the graph-based agent to answer a question using the knowledge graph.

```bash
python main.py run_agent \
    --question "What is the main topic discussed?" \
    --graph_file output/graph.pkl \
    --output_dir output/ \
    --max_iterations 20
```

**Outputs:**
- `result.json` or `result_{id}.json`: Agent exploration results and answer

### 4. Evaluate (`evaluate`)

Evaluates agent performance and calculates metrics (IES calculation in development).

```bash
python main.py evaluate \
    --results_file output/all_results.json \
    --qa_pairs_file output/qa_pairs.json \
    --output_dir output/
```

**Outputs:**
- `evaluation.json`: Evaluation metrics and scores

### 5. Full Pipeline (`full_pipeline`)

Runs the complete experimental workflow end-to-end.

```bash
python main.py full_pipeline \
    --data_path data/longmemeval/longmemeval_s_cleaned.json \
    --output_dir output/ \
    --limit_groups 5 \
    --max_chunks 10 \
    --max_iterations 20
```

**Skip specific steps:**
```bash
python main.py full_pipeline \
    --data_path data.json \
    --output_dir output/ \
    --skip_data \      # Use existing prepared data
    --skip_graph \     # Use existing graphs
    --skip_agent       # Skip agent execution
```

## 🔬 Experimental Workflow

The experimental workflow follows these steps:

1. **Data Preparation**
   - Load LongMemEval dataset
   - Sample intervention haystacks and sessions
   - Concatenate sessions from different topics
   - Create synthetic multi-topic dialogues

2. **Graph Construction**
   - Split dialogue into chunks
   - Extract atomic facts and key elements using LLM
   - Build knowledge graph with nodes (key elements) and edges (shared facts)
   - Store graph structure for agent exploration

3. **Agent Execution**
   - Create rational plan for answering question
   - Select initial nodes from graph
   - Explore atomic facts, chunks, and neighbors
   - Apply unbinding mechanism to filter irrelevant nodes
   - Generate final answer from exploration notebook

4. **Evaluation**
   - Compare agent answers with ground truth
   - Calculate retrieval accuracy across dialogue turns
   - Compute Interference Endurance Score (IES)
   - Analyze task complexity effects

## 📊 Configuration

Edit `src/config.py` to customize:

- **Data Processing**: `SAMPLE_NUM_FOR_EACH_DATA`, `INTERVENE_DATA_NUM`, `SAMPLE_SESSION_NUM`
- **Model Settings**: `LLAMA_MODEL_ID`, `FLAN_T5_MODEL_NAME`, `LLAMA_SAVE_PATH`
- **Generation**: `MAX_NEW_TOKENS`, `MAX_TOKENS_PER_CHUNK`
- **Data Paths**: `DEFAULT_DATA_PATH`

## 📚 References

### Key Papers

1. **GraphReader**: Li et al. (2024) - "GraphReader: Building graph-based agent to enhance long-context abilities of large language models"
   - Location: `ref/papers/GraphReader.pdf`

2. **Proactive Interference**: Wang & Sun (2025) - "Unable to forget: Proactive Interference reveals working memory limits in LLMs"
   - Introduces IES (Interference Endurance Score) metric

3. **LongMemEval**: Dataset for long-term memory evaluation in dialogues

### Proposal

- Original proposal: `Topic_Shift_Proposal__revised_ver__.pdf`

## 🛠️ Current Status

### ✅ Completed

- Data processing pipeline for creating intervened datasets
- Knowledge graph construction from text
- LangGraph-based agent workflow
- Basic unbinding mechanism (placeholder)
- Command-line interface for workflow control

### 🚧 In Progress / TODO

- Enhanced topic-aware unbinding mechanism
- Full IES (Interference Endurance Score) calculation
- Task complexity analysis
- Topic shift detection
- Multi-path exploration support
- Baseline comparisons

See `IMPLEMENTATION_PLAN.md` for detailed roadmap.

## 🧪 Example Workflow

```bash
# 1. Prepare a small test dataset
python main.py prepare_data \
    --data_path data/longmemeval/longmemeval_s_cleaned.json \
    --output_dir test_output/ \
    --limit_groups 2

# 2. Build graph for first group
python main.py build_graph \
    --text_file test_output/intervened_texts.txt \
    --output_dir test_output/ \
    --graph_id 0 \
    --max_chunks 5

# 3. Run agent on a question
python main.py run_agent \
    --question "What was discussed in the conversation?" \
    --graph_file test_output/graph_0.pkl \
    --output_dir test_output/

# 4. Check results
cat test_output/result.json
```

## 📝 Notes

- The project uses **LangGraph** for agent workflow orchestration
- **Llama 3.1 8B** is the default model (configurable)
- Graph exploration is limited by `max_iterations` to prevent infinite loops
- Unbinding mechanism is currently a placeholder and needs enhancement
- IES calculation is adapted from Wang & Sun (2025) for dialogue scenarios

## 🤝 Contributing

This is a research project. For questions or contributions, please refer to the implementation plan and maintain consistency with the existing code structure.

## 📄 License

[Specify license if applicable]

## 👤 Author

Cheng-Han Wu  
University of Southern California  
{wuchengh}@usc.edu

---

For detailed module documentation, see `src/README.md`.  
For implementation roadmap, see `IMPLEMENTATION_PLAN.md`.
