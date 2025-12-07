# Implementation Plan: Topic Switch with GraphReader & LangGraph

## Overview
This document outlines the implementation plan for studying **Proactive Interference (PI)** in multi-turn dialogues using GraphReader-based agents with LangGraph. The project adapts the Interference Endurance Score (IES) for natural dialogue scenarios and implements unbinding mechanisms to mitigate PI.

**References:**
- Proposal: `Topic_Shift_Proposal__revised_ver__.pdf`
- GraphReader: Li et al. (2024) - "GraphReader: Building graph-based agent to enhance long-context abilities of large language models"
- IES Metric: Wang & Sun (2025) - "Unable to forget: Proactive Interference reveals working memory limits in LLMs"

---

## Current Status

### ✅ Completed Components
1. **Data Processing Module** (`src/data_processing.py`)
   - ✅ Data loading from LongMemEval
   - ✅ Intervention sampling (haystacks and sessions)
   - ✅ Session concatenation for topic switching

2. **Graph Construction** (`src/graph_construction.py`)
   - ✅ Atomic fact extraction
   - ✅ Knowledge graph building
   - ✅ Node and edge creation

3. **LangGraph Agent Framework** (`src/graph_agent.py`)
   - ✅ Basic workflow structure
   - ✅ Graph exploration nodes (plan, select, explore)
   - ⚠️ Unbinding mechanism (placeholder only)

4. **Supporting Modules**
   - ✅ State management (`src/state.py`)
   - ✅ Prompts (`src/prompts.py`)
   - ✅ Model loading (`src/models.py`)
   - ✅ Configuration (`src/config.py`)

### ❌ Missing Components
1. Enhanced unbinding mechanism with topic detection
2. IES (Interference Endurance Score) calculation
3. Task complexity analysis
4. End-to-end evaluation pipeline
5. Multi-path exploration support
6. Topic shift detection
7. Baselines comparison

---

## Implementation Phases

## Phase 1: Enhanced Unbinding Mechanism 🔴 **PRIORITY**

### Goal
Implement sophisticated topic-aware unbinding to mitigate proactive interference by identifying and marking irrelevant nodes from previous topics.

### Tasks

#### 1.1 Topic Detection Module
**File:** `src/topic_detection.py` (NEW)

```python
# Functions to implement:
- detect_topic_shifts(dialogue: List[str]) -> List[TopicSegment]
- extract_topic_entities(text: str) -> List[str]
- compute_topic_similarity(topic1: List[str], topic2: List[str]) -> float
- identify_current_topic(state: GraphExplorationState) -> TopicContext
```

**Key Components:**
- Use entity extraction to identify topic-relevant entities
- Track entity mentions across dialogue turns
- Detect topic boundaries using entity overlap analysis
- Maintain topic context history

#### 1.2 Enhanced Unbinding in Graph Agent
**File:** `src/graph_agent.py` (MODIFY `apply_unbinding()`)

```python
def apply_unbinding(state: GraphExplorationState) -> GraphExplorationState:
    """
    Enhanced unbinding with topic-aware node relevance scoring.
    """
    # 1. Identify current topic from question and recent exploration
    # 2. Score all visited nodes by topic relevance
    # 3. Mark low-relevance nodes as irrelevant
    # 4. Update topic context in state
    # 5. Filter neighbor exploration using irrelevant_nodes
```

**Key Features:**
- Topic-based relevance scoring
- Temporal decay (older topics get lower relevance)
- Entity overlap analysis between current topic and node content
- Configurable threshold for marking nodes as irrelevant

#### 1.3 Unbinding Prompts
**File:** `src/prompts.py` (ADD)

```python
UNBINDING_PROMPT = """...
Instructions for LLM to identify irrelevant nodes based on topic context.
"""
```

**Implementation Notes:**
- Add prompt for LLM-assisted topic relevance judgment
- Include current topic context in unbinding decisions
- Allow model to reason about node relevance

### Dependencies
- Entity extraction (can use LLM or NER model)
- Topic modeling or simple entity tracking

### Estimated Effort
- **Time:** 3-5 days
- **Complexity:** Medium
- **Testing:** Unit tests for topic detection, integration tests for unbinding

---

## Phase 2: IES Evaluation Framework 🔴 **PRIORITY**

### Goal
Adapt the Interference Endurance Score (IES) from Wang & Sun (2025) for multi-turn dialogue evaluation. IES measures a model's ability to resist proactive interference.

### Tasks

#### 2.1 IES Calculation Module
**File:** `src/evaluation.py` (NEW)

```python
# Functions to implement:
- calculate_retrieval_accuracy(predictions: List[str], ground_truth: List[str]) -> float
- compute_ies(accuracies: List[float], dialogue_turns: List[int], 
              use_log_scale: bool = True) -> float
- plot_ies_curve(accuracies: List[float], dialogue_turns: List[int]) -> Figure
- compare_ies_scores(model_scores: Dict[str, float]) -> ComparisonReport
```

**IES Formula (Adapted):**
```
IES = AUC(retrieval_accuracy, f(dialogue_turns))
where f(x) = log(x) for log-scaled or x for linear
```

**Adaptations from Proposal:**
- Replace "log-scaled update counts" with "dialogue turns"
- Option to normalize by entity count (task complexity proxy)
- Support multiple difficulty metrics (answer length, entity count, etc.)

#### 2.2 Evaluation Data Structure
**File:** `src/evaluation.py` (NEW)

```python
@dataclass
class EvaluationResult:
    question_id: str
    question: str
    ground_truth: str
    prediction: str
    dialogue_turn: int
    topic_context: Optional[str]
    accuracy: float
    task_complexity: Dict[str, float]  # answer_length, entity_count, etc.

@dataclass
class IESReport:
    model_name: str
    ies_score: float
    accuracy_by_turn: List[Tuple[int, float]]
    complexity_analysis: Dict[str, float]
    total_samples: int
```

#### 2.3 Batch Evaluation Pipeline
**File:** `src/evaluation.py` (NEW)

```python
def evaluate_model_on_dataset(
    model,
    dataset: List[Dict],
    intervened_data: List[Tuple],
    max_samples: Optional[int] = None
) -> List[EvaluationResult]:
    """
    Run evaluation on all QA pairs in intervened dataset.
    """
    # 1. Load intervened texts
    # 2. Build knowledge graphs for each
    # 3. Run graph agent for each question
    # 4. Compare predictions to ground truth
    # 5. Track dialogue turns and topic shifts
    # 6. Calculate accuracy per turn
```

#### 2.4 Integration with LongMemEval
**File:** `src/evaluation.py` (NEW)

```python
def extract_qa_from_longmemeval(
    data: List[Dict],
    intervened_sessions: List[Tuple]
) -> List[QAPair]:
    """
    Extract QA pairs from LongMemEval data with topic shift annotations.
    """
    # Extract questions and answers
    # Identify answer session IDs
    # Map to dialogue turns
    # Track long-term memory ability labels
```

### Dependencies
- Matplotlib for plotting
- NumPy for AUC calculation
- Integration with existing graph agent

### Estimated Effort
- **Time:** 4-6 days
- **Complexity:** Medium-High
- **Testing:** Unit tests for IES calculation, validation against paper examples

---

## Phase 3: Task Complexity Analysis

### Goal
Examine how task complexity affects proactive interference, as hypothesized in the proposal (Section 6).

### Tasks

#### 3.1 Complexity Metrics
**File:** `src/complexity.py` (NEW)

```python
# Functions to implement:
- calculate_answer_length(answer: str) -> int
- count_entities_in_question(question: str, graph: KnowledgeGraph) -> int
- estimate_retrieval_difficulty(question: str, graph: KnowledgeGraph) -> float
- compute_complexity_score(qa_pair: QAPair, graph: KnowledgeGraph) -> Dict[str, float]
```

**Complexity Dimensions:**
- Answer length (proxy for information density)
- Number of entities required
- Graph traversal depth needed
- Question type (factual, reasoning, comparison)

#### 3.2 Complexity-Aware Analysis
**File:** `src/evaluation.py` (MODIFY)

```python
def analyze_ies_by_complexity(
    results: List[EvaluationResult],
    complexity_metric: str = "answer_length"
) -> Dict[str, IESReport]:
    """
    Compute IES scores stratified by task complexity.
    """
    # Group results by complexity buckets
    # Calculate IES for each group
    # Compare performance across complexity levels
```

#### 3.3 Ablation Studies
**File:** `src/experiments.py` (NEW)

```python
def run_complexity_ablation(
    dataset: List[Dict],
    complexity_metrics: List[str]
) -> AblationReport:
    """
    Study effect of task complexity on PI.
    """
    # Vary complexity dimensions
    # Measure IES at each level
    # Analyze ratio of complexity effect
```

### Dependencies
- Entity extraction
- Graph analysis tools
- Statistical analysis (pandas, scipy)

### Estimated Effort
- **Time:** 3-4 days
- **Complexity:** Medium
- **Testing:** Statistical validation of complexity effects

---

## Phase 4: Multi-Path Exploration Enhancement

### Goal
Implement true multi-path exploration as described in GraphReader, allowing the agent to explore multiple starting nodes in parallel and aggregate findings.

### Tasks

#### 4.1 Parallel Path Exploration
**File:** `src/graph_agent.py` (MODIFY)

```python
def explore_multiple_paths(
    state: GraphExplorationState,
    num_paths: int = 5
) -> List[GraphExplorationState]:
    """
    Explore multiple independent paths from different initial nodes.
    Each path maintains its own notebook.
    """
    # Initialize multiple states with different initial nodes
    # Run exploration in parallel (or sequentially)
    # Aggregate notebooks at the end
```

#### 4.2 Path Aggregation
**File:** `src/graph_agent.py` (MODIFY `generate_final_answer()`)

```python
def aggregate_path_findings(
    path_states: List[GraphExplorationState]
) -> str:
    """
    Combine notebooks from multiple exploration paths using majority voting.
    """
    # Extract notebooks from all paths
    # Resolve conflicts using voting strategy
    # Generate final answer
```

#### 4.3 LangGraph Multi-Agent Setup
**File:** `src/graph_agent.py` (NEW)

```python
def create_multi_path_agent() -> StateGraph:
    """
    Create LangGraph workflow supporting parallel path exploration.
    """
    # Split after initial node selection
    # Run multiple exploration paths
    # Merge before answer generation
```

**LangGraph Pattern:**
- Use `add_node` for each path
- Use conditional edges to spawn paths
- Aggregate before final answer

### Dependencies
- LangGraph multi-agent patterns
- State merging logic

### Estimated Effort
- **Time:** 4-5 days
- **Complexity:** Medium-High
- **Testing:** Verify path independence and aggregation quality

---

## Phase 5: Topic Shift Detection & Tracking

### Goal
Implement explicit topic shift detection to support evaluation and unbinding mechanisms.

### Tasks

#### 5.1 Topic Boundary Detection
**File:** `src/topic_detection.py` (EXTEND)

```python
def detect_topic_boundaries(
    dialogue: List[str],
    method: str = "entity_overlap"
) -> List[TopicBoundary]:
    """
    Identify points where topics shift in concatenated dialogue.
    """
    # Method 1: Entity overlap analysis
    # Method 2: LLM-based topic classification
    # Method 3: Semantic similarity clustering
```

#### 5.2 Topic Annotation
**File:** `src/data_processing.py` (MODIFY)

```python
def annotate_topic_shifts(
    intervened_text: str,
    haystack_groups: List[Dict]
) -> AnnotatedDialogue:
    """
    Add topic shift annotations to intervened dialogue.
    """
    # Detect boundaries
    # Label topics
    # Map to original haystack sources
```

#### 5.3 Topic-Aware Evaluation
**File:** `src/evaluation.py` (MODIFY)

```python
def evaluate_by_topic_distance(
    results: List[EvaluationResult],
    topic_boundaries: List[TopicBoundary]
) -> Dict[str, float]:
    """
    Analyze PI effects based on distance from topic shift.
    """
    # Measure accuracy at different distances from topic boundary
    # Identify where PI effects are strongest
```

### Dependencies
- NLP libraries (spaCy, transformers)
- Clustering algorithms (scikit-learn)

### Estimated Effort
- **Time:** 3-4 days
- **Complexity:** Medium
- **Testing:** Validate topic boundaries against ground truth annotations

---

## Phase 6: End-to-End Pipeline & Scripts

### Goal
Create complete, runnable pipeline from data processing through evaluation.

### Tasks

#### 6.1 Main Pipeline Script
**File:** `scripts/run_pipeline.py` (NEW)

```python
def main():
    """
    Complete pipeline:
    1. Load LongMemEval data
    2. Create intervened dataset
    3. Build knowledge graphs
    4. Run graph agent evaluations
    5. Calculate IES scores
    6. Generate reports
    """
```

**CLI Interface:**
```bash
python scripts/run_pipeline.py \
    --data_path data/longmemeval_s.json \
    --model_name llama-3.1-8B \
    --output_dir results/ \
    --max_samples 100
```

#### 6.2 Experiment Configuration
**File:** `src/experiments.py` (NEW)

```python
@dataclass
class ExperimentConfig:
    dataset_path: str
    model_name: str
    intervention_params: Dict[str, Any]
    evaluation_params: Dict[str, Any]
    unbinding_enabled: bool
    multi_path: bool
    num_paths: int

def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run complete experiment with given configuration."""
```

#### 6.3 Results Visualization
**File:** `src/visualization.py` (NEW)

```python
def plot_ies_comparison(models: Dict[str, IESReport]) -> Figure
def plot_accuracy_by_turn(results: List[EvaluationResult]) -> Figure
def plot_complexity_analysis(complexity_report: Dict) -> Figure
def generate_evaluation_report(results: ExperimentResults) -> str
```

#### 6.4 Example Scripts
**File:** `examples/` (NEW)

```
examples/
├── basic_usage.py          # Simple graph agent example
├── evaluation_example.py    # IES calculation example
├── unbinding_demo.py        # Unbinding mechanism demo
└── full_pipeline.py         # Complete pipeline example
```

### Dependencies
- Click or argparse for CLI
- Report generation (markdown, LaTeX)
- Visualization (matplotlib, seaborn)

### Estimated Effort
- **Time:** 3-4 days
- **Complexity:** Low-Medium
- **Testing:** End-to-end integration tests

---

## Phase 7: Baseline Comparisons

### Goal
Compare graph-based agent with baseline methods to demonstrate effectiveness.

### Tasks

#### 7.1 Baseline Implementations
**File:** `src/baselines.py` (NEW)

```python
def baseline_naive_retrieval(question: str, text: str) -> str:
    """Simple retrieval baseline (first match)."""

def baseline_llm_direct(question: str, text: str) -> str:
    """Direct LLM without graph structure."""

def baseline_with_unbinding(question: str, text: str, 
                            topic_context: str) -> str:
    """LLM with simple prompt-based unbinding."""

def baseline_graphreader_original(question: str, graph: KnowledgeGraph) -> str:
    """Original GraphReader without unbinding."""
```

#### 7.2 Comparative Evaluation
**File:** `src/evaluation.py` (MODIFY)

```python
def compare_with_baselines(
    dataset: List[Dict],
    models: Dict[str, Callable]
) -> ComparisonReport:
    """
    Run evaluation with multiple baselines and graph agent.
    """
```

### Dependencies
- Baseline implementations
- Consistent evaluation framework

### Estimated Effort
- **Time:** 2-3 days
- **Complexity:** Low
- **Testing:** Ensure fair comparison

---

## Phase 8: Testing & Validation

### Goal
Ensure code quality and correctness through comprehensive testing.

### Tasks

#### 8.1 Unit Tests
**Files:** `tests/unit/` (NEW)

```
tests/unit/
├── test_data_processing.py
├── test_graph_construction.py
├── test_graph_agent.py
├── test_topic_detection.py
├── test_evaluation.py
└── test_unbinding.py
```

#### 8.2 Integration Tests
**Files:** `tests/integration/` (NEW)

```
tests/integration/
├── test_end_to_end.py
├── test_ies_calculation.py
└── test_multi_path_exploration.py
```

#### 8.3 Validation Tests
**Files:** `tests/validation/` (NEW)

```
tests/validation/
├── validate_ies_implementation.py  # Compare with paper
├── validate_graphreader.py         # Compare with original
└── validate_data_format.py         # LongMemEval compatibility
```

### Testing Framework
- pytest
- Coverage target: >80%

### Estimated Effort
- **Time:** 5-7 days
- **Complexity:** Medium
- **Testing:** Continuous integration

---

## Implementation Priority & Timeline

### Critical Path (Must Have)
1. **Phase 1: Enhanced Unbinding** (Week 1-2)
2. **Phase 2: IES Evaluation** (Week 2-3)
3. **Phase 6: End-to-End Pipeline** (Week 3-4)

### Important (Should Have)
4. **Phase 3: Task Complexity** (Week 4-5)
5. **Phase 5: Topic Detection** (Week 5-6)

### Nice to Have (Could Have)
6. **Phase 4: Multi-Path Exploration** (Week 6-7)
7. **Phase 7: Baselines** (Week 7-8)
8. **Phase 8: Testing** (Ongoing)

### Estimated Total Timeline
- **Minimum Viable Product:** 4 weeks (Phases 1, 2, 6)
- **Complete Implementation:** 8 weeks (All phases)

---

## Technical Considerations

### LangGraph Architecture
- Use `StateGraph` for sequential workflow
- Use conditional edges for decision points
- Support checkpointing for long-running evaluations
- Implement state persistence for debugging

### GraphReader Alignment
- Follow original GraphReader prompts (from paper appendix)
- Maintain compatibility with original workflow
- Add unbinding as enhancement, not replacement

### Scalability
- Support batch processing for large datasets
- Cache knowledge graphs to avoid recomputation
- Parallel path exploration (if multi-path implemented)
- Memory-efficient state management

### Model Integration
- Support multiple LLM backends (OpenAI, HuggingFace, etc.)
- Configurable model parameters
- Cost tracking for API calls

---

## Dependencies & Setup

### Python Dependencies
```
langgraph>=0.0.40
langchain>=0.1.0
transformers>=4.30.0
torch>=2.0.0
tiktoken>=0.5.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pytest>=7.4.0
```

### Data Dependencies
- LongMemEval dataset (GitHub)
- Access to LLM API or local model

### External Services
- LLM API (OpenAI, Anthropic, or local)
- Optional: Vector database for entity storage

---

## Success Metrics

### Implementation Success
- ✅ All core functions implemented
- ✅ IES calculation matches paper methodology
- ✅ Unbinding reduces PI in evaluation
- ✅ End-to-end pipeline runs successfully

### Research Success
- ✅ Demonstrate PI effects in multi-turn dialogue
- ✅ Show unbinding mechanism improves IES scores
- ✅ Analyze task complexity effects
- ✅ Compare with baselines

### Code Quality
- ✅ >80% test coverage
- ✅ All functions documented
- ✅ Code passes linting
- ✅ Examples and tutorials available

---

## Risk Mitigation

### Risks
1. **LangGraph learning curve** → Use existing examples, start simple
2. **IES calculation accuracy** → Validate against paper examples
3. **Unbinding effectiveness** → Iterative refinement, ablation studies
4. **Data processing complexity** → Modular design, unit tests
5. **Cost overruns** → Local model option, caching, batch processing

### Mitigation Strategies
- Prototype critical components early
- Validate against known examples
- Implement fallback mechanisms
- Monitor API costs closely
- Regular code reviews

---

## Next Steps (Immediate Actions)

1. **Review & Approve Plan** - Ensure alignment with proposal goals
2. **Set Up Development Environment** - Install dependencies, setup project structure
3. **Start Phase 1** - Implement topic detection and enhanced unbinding
4. **Create Issue Tracker** - Track tasks and progress
5. **Set Up CI/CD** - Automated testing and validation

---

## References

1. Li et al. (2024). "GraphReader: Building graph-based agent to enhance long-context abilities of large language models." arXiv:2406.14550
2. Wang & Sun (2025). "Unable to forget: Proactive Interference reveals working memory limits in LLMs beyond context length." arXiv:2506.08184
3. Bai et al. (2024). "MT-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues." arXiv:2402.14762
4. LangGraph Documentation: https://langchain-ai.github.io/langgraph/

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Status:** Planning Phase



