#!/usr/bin/env python3
"""
Main entry point for Topic Switch experiments.
Controls different steps of the experimental workflow via command-line arguments.

Usage:
    python main.py prepare_data --data_path data.json --output_dir output/
    python main.py build_graph --text_file output/intervened_texts.txt --output_dir output/
    python main.py run_agent --question "What is X?" --graph_file output/graph.pkl
    python main.py evaluate --results_file output/results.json
    python main.py full_pipeline --data_path data.json --output_dir output/
"""

import argparse
import json
import pickle
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_processing import (
    create_intervened_dataset,
    load_data
)
from src.graph_construction import extract_key_elements_and_atomic_facts
from src.graph_agent import run_graph_agent
from src.state import KnowledgeGraph
from src.config import (
    SAMPLE_NUM_FOR_EACH_DATA,
    INTERVENE_DATA_NUM,
    SAMPLE_SESSION_NUM,
    DEFAULT_DATA_PATH
)


def prepare_data_step(
    data_path: str,
    output_dir: str,
    sample_num: Optional[int] = None,
    intervene_num: Optional[int] = None,
    session_num: Optional[int] = None,
    limit_groups: Optional[int] = None
):
    """
    Step 1: Prepare intervened dataset from LongMemEval.
    
    Creates synthetic dialogues by concatenating multiple conversation threads
    to simulate natural topic switching.
    """
    print("=" * 80)
    print("STEP 1: Preparing Intervened Dataset")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create intervened dataset
    data, intervene_sampled_session_data, intervened_texts = create_intervened_dataset(
        data_path=data_path,
        sample_num_for_each_data=sample_num,
        intervene_data_num=intervene_num,
        sample_session_num=session_num,
        limit_groups=limit_groups
    )
    
    # Save intervened texts
    intervened_texts_file = output_path / "intervened_texts.txt"
    with open(intervened_texts_file, "w", encoding="utf-8") as f:
        for i, text in enumerate(intervened_texts):
            f.write(f"=== Group {i} ===\n\n{text}\n\n{'='*80}\n\n")
    
    # Save metadata
    metadata = {
        "num_groups": len(intervened_texts),
        "sample_num_for_each_data": sample_num or SAMPLE_NUM_FOR_EACH_DATA,
        "intervene_data_num": intervene_num or INTERVENE_DATA_NUM,
        "sample_session_num": session_num or SAMPLE_SESSION_NUM,
        "intervene_sampled_session_data": [
            [
                {"idx": item["idx"], "session_idx": item["session_idx"]}
                for item in group
            ]
            for group in intervene_sampled_session_data
        ]
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Save original data with QA pairs
    qa_pairs = []
    for i, item in enumerate(data):
        qa_pairs.append({
            "idx": i,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "ability": item.get("ability", "")
        })
    
    qa_file = output_path / "qa_pairs.json"
    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"\n✓ Saved {len(intervened_texts)} intervened text groups to {intervened_texts_file}")
    print(f"✓ Saved metadata to {metadata_file}")
    print(f"✓ Saved {len(qa_pairs)} QA pairs to {qa_file}")
    
    return intervened_texts, metadata, qa_pairs


def build_graph_step(
    text_file: Optional[str] = None,
    text_content: Optional[str] = None,
    output_dir: str = "output/",
    max_chunks: Optional[int] = None,
    graph_id: Optional[int] = None
):
    """
    Step 2: Build knowledge graph from text.
    
    Extracts atomic facts and key elements, then constructs a knowledge graph.
    """
    print("=" * 80)
    print("STEP 2: Building Knowledge Graph")
    print("=" * 80)
    
    # Load text
    if text_file:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif text_content:
        text = text_content
    else:
        raise ValueError("Either text_file or text_content must be provided")
    
    print(f"Processing text ({len(text)} characters)...")
    
    # Extract key elements and atomic facts
    graph = extract_key_elements_and_atomic_facts(
        text=text,
        max_chunks=max_chunks
    )
    
    # Save graph
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if graph_id is not None:
        graph_file = output_path / f"graph_{graph_id}.pkl"
    else:
        graph_file = output_path / "graph.pkl"
    
    with open(graph_file, "wb") as f:
        pickle.dump(graph, f)
    
    print(f"\n✓ Created graph with {len(graph.nodes)} nodes")
    print(f"✓ Extracted {len(graph.atomic_facts)} atomic facts")
    print(f"✓ Saved graph to {graph_file}")
    
    return graph, graph_file


def run_agent_step(
    question: str,
    graph_file: str,
    output_dir: str = "output/",
    max_iterations: int = 20,
    result_id: Optional[int] = None
):
    """
    Step 3: Run graph agent to answer a question.
    
    Uses the graph-based agent to explore the knowledge graph and generate an answer.
    """
    print("=" * 80)
    print("STEP 3: Running Graph Agent")
    print("=" * 80)
    
    # Load graph
    with open(graph_file, "rb") as f:
        graph = pickle.load(f)
    
    print(f"Question: {question}")
    print(f"Graph has {len(graph.nodes)} nodes")
    
    # Run agent
    final_state = run_graph_agent(
        question=question,
        knowledge_graph=graph,
        max_iterations=max_iterations
    )
    
    # Save result
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result = {
        "question": question,
        "answer": final_state.get("final_answer", ""),
        "rational_plan": final_state.get("rational_plan", ""),
        "visited_nodes": final_state.get("visited_nodes", []),
        "visited_chunks": final_state.get("visited_chunks", []),
        "irrelevant_nodes": final_state.get("irrelevant_nodes", []),
        "notebooks": final_state.get("notebooks", []),
        "previous_actions": final_state.get("previous_actions", [])
    }
    
    if result_id is not None:
        result_file = output_path / f"result_{result_id}.json"
    else:
        result_file = output_path / "result.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Final Answer: {result['answer']}")
    print(f"✓ Visited {len(result['visited_nodes'])} nodes")
    print(f"✓ Saved result to {result_file}")
    
    return result, result_file


def evaluate_step(
    results_file: str,
    qa_pairs_file: str,
    output_dir: str = "output/",
    calculate_ies: bool = True
):
    """
    Step 4: Evaluate results and calculate IES (Interference Endurance Score).
    
    Compares agent answers with ground truth and calculates metrics.
    """
    print("=" * 80)
    print("STEP 4: Evaluating Results")
    print("=" * 80)
    
    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Load QA pairs
    with open(qa_pairs_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    
    # Evaluate (simple accuracy for now)
    # TODO: Implement full IES calculation
    evaluation = {
        "num_questions": len(results) if isinstance(results, list) else 1,
        "results": results if isinstance(results, list) else [results],
        "ies_score": None
    }
    
    if calculate_ies:
        print("\n⚠ IES calculation not yet implemented")
        print("  This will calculate Interference Endurance Score based on:")
        print("  - Retrieval accuracy across dialogue turns")
        print("  - Area under the curve (AUC) of accuracy")
    
    # Save evaluation
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    eval_file = output_path / "evaluation.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    
    print(f"\n✓ Saved evaluation to {eval_file}")
    
    return evaluation, eval_file


def full_pipeline_step(
    data_path: str,
    output_dir: str = "output/",
    sample_num: Optional[int] = None,
    intervene_num: Optional[int] = None,
    session_num: Optional[int] = None,
    limit_groups: Optional[int] = None,
    max_chunks: Optional[int] = None,
    max_iterations: int = 20,
    skip_data: bool = False,
    skip_graph: bool = False,
    skip_agent: bool = False
):
    """
    Run the complete pipeline: data preparation -> graph construction -> agent -> evaluation.
    """
    print("=" * 80)
    print("FULL PIPELINE: Topic Switch Experiment")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare data
    if not skip_data:
        intervened_texts, metadata, qa_pairs = prepare_data_step(
            data_path=data_path,
            output_dir=output_dir,
            sample_num=sample_num,
            intervene_num=intervene_num,
            session_num=session_num,
            limit_groups=limit_groups
        )
    else:
        print("Skipping data preparation step...")
        # Load existing data
        metadata_file = output_path / "metadata.json"
        qa_file = output_path / "qa_pairs.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        with open(qa_file, "r") as f:
            qa_pairs = json.load(f)
        intervened_texts_file = output_path / "intervened_texts.txt"
        with open(intervened_texts_file, "r") as f:
            content = f.read()
            # Simple parsing - split by group markers
            groups = content.split("=== Group")
            intervened_texts = [g.split("===")[-1].strip() for g in groups[1:]]
    
    # Step 2: Build graphs for each intervened text group
    graphs = []
    if not skip_graph:
        print(f"\nBuilding graphs for {len(intervened_texts)} text groups...")
        for i, text in enumerate(tqdm(intervened_texts, desc="Building graphs")):
            graph, graph_file = build_graph_step(
                text_content=text,
                output_dir=output_dir,
                max_chunks=max_chunks,
                graph_id=i
            )
            graphs.append((graph, graph_file))
    else:
        print("Skipping graph construction step...")
        # Load existing graphs
        for i in range(len(intervened_texts)):
            graph_file = output_path / f"graph_{i}.pkl"
            if graph_file.exists():
                with open(graph_file, "rb") as f:
                    graph = pickle.load(f)
                    graphs.append((graph, graph_file))
    
    # Step 3: Run agent for each QA pair
    results = []
    if not skip_agent and graphs:
        print(f"\nRunning agent for {len(qa_pairs)} QA pairs...")
        for i, qa in enumerate(tqdm(qa_pairs, desc="Running agent")):
            # Use corresponding graph (modulo if fewer graphs than QA pairs)
            graph_idx = i % len(graphs)
            graph, graph_file = graphs[graph_idx]
            
            result, result_file = run_agent_step(
                question=qa["question"],
                graph_file=str(graph_file),
                output_dir=output_dir,
                max_iterations=max_iterations,
                result_id=i
            )
            result["qa_idx"] = i
            result["ground_truth"] = qa.get("answer", "")
            result["ability"] = qa.get("ability", "")
            results.append(result)
        
        # Save all results
        all_results_file = output_path / "all_results.json"
        with open(all_results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved all results to {all_results_file}")
    
    # Step 4: Evaluate
    if results:
        qa_file = output_path / "qa_pairs.json"
        results_file = output_path / "all_results.json"
        evaluation, eval_file = evaluate_step(
            results_file=str(results_file),
            qa_pairs_file=str(qa_file),
            output_dir=output_dir
        )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Topic Switch Experiment - Control workflow steps via command-line arguments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Prepare intervened dataset
  python main.py prepare_data --data_path data.json --output_dir output/
  
  # Step 2: Build knowledge graph
  python main.py build_graph --text_file output/intervened_texts.txt --output_dir output/
  
  # Step 3: Run agent on a question
  python main.py run_agent --question "What is X?" --graph_file output/graph.pkl
  
  # Step 4: Evaluate results
  python main.py evaluate --results_file output/all_results.json --qa_pairs_file output/qa_pairs.json
  
  # Run full pipeline
  python main.py full_pipeline --data_path data.json --output_dir output/ --limit_groups 10
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Prepare data command
    parser_prepare = subparsers.add_parser("prepare_data", help="Step 1: Prepare intervened dataset")
    parser_prepare.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                               help="Path to LongMemEval JSON file")
    parser_prepare.add_argument("--output_dir", type=str, default="output/",
                               help="Output directory for intervened texts and metadata")
    parser_prepare.add_argument("--sample_num", type=int, default=None,
                               help=f"Number of samples per data point (default: {SAMPLE_NUM_FOR_EACH_DATA})")
    parser_prepare.add_argument("--intervene_num", type=int, default=None,
                               help=f"Number of haystacks to combine (default: {INTERVENE_DATA_NUM})")
    parser_prepare.add_argument("--session_num", type=int, default=None,
                               help=f"Number of sessions per haystack (default: {SAMPLE_SESSION_NUM})")
    parser_prepare.add_argument("--limit_groups", type=int, default=None,
                               help="Limit number of intervention groups to create")
    
    # Build graph command
    parser_graph = subparsers.add_parser("build_graph", help="Step 2: Build knowledge graph")
    parser_graph.add_argument("--text_file", type=str, default=None,
                             help="Path to text file to process")
    parser_graph.add_argument("--text_content", type=str, default=None,
                             help="Text content directly (alternative to text_file)")
    parser_graph.add_argument("--output_dir", type=str, default="output/",
                             help="Output directory for graph")
    parser_graph.add_argument("--max_chunks", type=int, default=None,
                             help="Maximum number of chunks to process")
    parser_graph.add_argument("--graph_id", type=int, default=None,
                             help="ID for graph file (e.g., graph_0.pkl)")
    
    # Run agent command
    parser_agent = subparsers.add_parser("run_agent", help="Step 3: Run graph agent")
    parser_agent.add_argument("--question", type=str, required=True,
                             help="Question to answer")
    parser_agent.add_argument("--graph_file", type=str, required=True,
                             help="Path to pickled knowledge graph")
    parser_agent.add_argument("--output_dir", type=str, default="output/",
                             help="Output directory for results")
    parser_agent.add_argument("--max_iterations", type=int, default=20,
                             help="Maximum iterations for agent (default: 20)")
    parser_agent.add_argument("--result_id", type=int, default=None,
                             help="ID for result file (e.g., result_0.json)")
    
    # Evaluate command
    parser_eval = subparsers.add_parser("evaluate", help="Step 4: Evaluate results")
    parser_eval.add_argument("--results_file", type=str, required=True,
                            help="Path to results JSON file")
    parser_eval.add_argument("--qa_pairs_file", type=str, required=True,
                            help="Path to QA pairs JSON file")
    parser_eval.add_argument("--output_dir", type=str, default="output/",
                            help="Output directory for evaluation")
    parser_eval.add_argument("--no_ies", action="store_true",
                            help="Skip IES calculation")
    
    # Full pipeline command
    parser_pipeline = subparsers.add_parser("full_pipeline", help="Run complete pipeline")
    parser_pipeline.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                                help="Path to LongMemEval JSON file")
    parser_pipeline.add_argument("--output_dir", type=str, default="output/",
                                help="Output directory")
    parser_pipeline.add_argument("--sample_num", type=int, default=None,
                                help="Number of samples per data point")
    parser_pipeline.add_argument("--intervene_num", type=int, default=None,
                                help="Number of haystacks to combine")
    parser_pipeline.add_argument("--session_num", type=int, default=None,
                                help="Number of sessions per haystack")
    parser_pipeline.add_argument("--limit_groups", type=int, default=None,
                                help="Limit number of intervention groups")
    parser_pipeline.add_argument("--max_chunks", type=int, default=None,
                                help="Maximum chunks per graph")
    parser_pipeline.add_argument("--max_iterations", type=int, default=20,
                                help="Maximum agent iterations")
    parser_pipeline.add_argument("--skip_data", action="store_true",
                                help="Skip data preparation step")
    parser_pipeline.add_argument("--skip_graph", action="store_true",
                                help="Skip graph construction step")
    parser_pipeline.add_argument("--skip_agent", action="store_true",
                                help="Skip agent execution step")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "prepare_data":
            prepare_data_step(
                data_path=args.data_path,
                output_dir=args.output_dir,
                sample_num=args.sample_num,
                intervene_num=args.intervene_num,
                session_num=args.session_num,
                limit_groups=args.limit_groups
            )
        
        elif args.command == "build_graph":
            build_graph_step(
                text_file=args.text_file,
                text_content=args.text_content,
                output_dir=args.output_dir,
                max_chunks=args.max_chunks,
                graph_id=args.graph_id
            )
        
        elif args.command == "run_agent":
            run_agent_step(
                question=args.question,
                graph_file=args.graph_file,
                output_dir=args.output_dir,
                max_iterations=args.max_iterations,
                result_id=args.result_id
            )
        
        elif args.command == "evaluate":
            evaluate_step(
                results_file=args.results_file,
                qa_pairs_file=args.qa_pairs_file,
                output_dir=args.output_dir,
                calculate_ies=not args.no_ies
            )
        
        elif args.command == "full_pipeline":
            full_pipeline_step(
                data_path=args.data_path,
                output_dir=args.output_dir,
                sample_num=args.sample_num,
                intervene_num=args.intervene_num,
                session_num=args.session_num,
                limit_groups=args.limit_groups,
                max_chunks=args.max_chunks,
                max_iterations=args.max_iterations,
                skip_data=args.skip_data,
                skip_graph=args.skip_graph,
                skip_agent=args.skip_agent
            )
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
