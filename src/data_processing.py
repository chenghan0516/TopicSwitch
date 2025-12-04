"""
Data processing functions for creating intervened datasets.
"""
import json
import random
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm

from .config import RANDOM_SEED

random.seed(RANDOM_SEED)


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        data_path: Path to the JSON file containing the data
        
    Returns:
        List of data dictionaries
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def sample_intervene_data(
    data_len: int, 
    sample_num_for_each_data: int, 
    intervene_data_num: int
) -> Tuple[Tuple[int, ...], ...]:
    """
    Sample sets of data indices to intervene.
    
    Args:
        data_len: Total length of the dataset
        sample_num_for_each_data: Number of samples to create for each data point
        intervene_data_num: Number of data points to include in each intervention
        
    Returns:
        Tuple of tuples, where each inner tuple contains indices of data to intervene
        Shape: (len(data) * sample_num_for_each_data, intervene_data_num)
    """
    intervene_list = []
    sample_data_num = intervene_data_num - 1
    
    for i in range(data_len):
        buffer = []
        pool = [j for j in range(data_len) if j != i]
        
        for _ in range(sample_num_for_each_data):
            random.shuffle(pool)
            sampled_list = [i] + pool[:sample_data_num]
            buffer.append(tuple(sampled_list))
        
        intervene_list += buffer
    
    return tuple(intervene_list)


def sample_session(
    data: List[Dict[str, Any]], 
    intervene_haystack_idx: Tuple[Tuple[int, ...], ...], 
    sample_session_num: int
) -> List[Tuple[Dict[str, Any], ...]]:
    """
    Sample sessions to intervene for each data group.
    
    Args:
        data: List of data dictionaries containing haystack sessions
        intervene_haystack_idx: Tuple of tuples containing haystack indices to intervene
        sample_session_num: Number of sessions to sample for each haystack
        
    Returns:
        List of tuples, where each tuple contains dictionaries with 'idx' and 'session_idx' keys
    """
    intervene_sampled_session_data = []
    
    for haystack_group in tqdm(intervene_haystack_idx):
        buffer = []
        
        for haystack_idx in haystack_group:
            temp = {"idx": haystack_idx, "session_idx": []}
            cur_data_session_ids = data[haystack_idx]["haystack_session_ids"]
            pool = [i for i in range(len(cur_data_session_ids))]
            
            # First get the index of session that contains the answer in its id
            for session_idx in range(len(cur_data_session_ids)):
                if "answer" in cur_data_session_ids[session_idx]:
                    temp["session_idx"].append(session_idx)
                    if session_idx in pool:
                        pool.remove(session_idx)
            
            # Then sample the rest session_idx from the available sessions
            cur_sample_num = sample_session_num - len(temp["session_idx"])
            if cur_sample_num > 0:
                sampled_session_idx = random.sample(pool, cur_sample_num)
                temp["session_idx"].extend(sampled_session_idx)
            
            # Sort the session_idx
            temp["session_idx"] = temp["session_idx"][:sample_session_num]
            temp["session_idx"].sort()
            buffer.append(temp)
        
        intervene_sampled_session_data.append(tuple(buffer))
    
    return intervene_sampled_session_data


def get_string_from_session(session: List[Dict[str, str]]) -> str:
    """
    Convert a session (list of messages) into a formatted string.
    
    Args:
        session: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Formatted string representation of the session
    """
    buffer = ""
    for message in session:
        buffer += f"{message['role']}: {message['content']}\n\n"
    return buffer.strip()


def retrieve_session_data(
    data: List[Dict[str, Any]], 
    haystack_group: Tuple[Dict[str, Any], ...]
) -> str:
    """
    Concatenate session data for each data pair.
    
    Args:
        data: List of data dictionaries containing haystack sessions
        haystack_group: Tuple of dictionaries with 'idx' and 'session_idx' keys
        
    Returns:
        Concatenated string of all session data
    """
    buffer = ""
    haystack_idx_list = [item["idx"] for item in haystack_group]
    session_idx_list = [item["session_idx"] for item in haystack_group]
    
    for cur_session_idx in range(len(session_idx_list[0])):
        for cur_haystack_idx in range(len(haystack_idx_list)):
            haystack_idx = haystack_idx_list[cur_haystack_idx]
            session_idx = session_idx_list[cur_haystack_idx][cur_session_idx]
            buffer += get_string_from_session(
                data[haystack_idx]["haystack_sessions"][session_idx]
            ) + "\n\n\n\n"
    
    return buffer.strip()


def create_intervened_dataset(
    data_path: str,
    sample_num_for_each_data: int = None,
    intervene_data_num: int = None,
    sample_session_num: int = None,
    limit_groups: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], ...]], List[str]]:
    """
    Complete pipeline for creating intervened datasets from LongMemEval data.
    
    This function combines all the steps:
    1. Load data
    2. Sample intervention haystacks
    3. Sample sessions from each haystack
    4. Retrieve concatenated session data
    
    Args:
        data_path: Path to LongMemEval JSON file
        sample_num_for_each_data: Number of samples per data point (uses config default if None)
        intervene_data_num: Number of haystacks to combine (uses config default if None)
        sample_session_num: Number of sessions per haystack (uses config default if None)
        limit_groups: Optional limit on number of intervention groups to create
        
    Returns:
        Tuple of:
        - data: Original loaded data
        - intervene_sampled_session_data: List of intervention groups with session indices
        - intervened_texts: List of concatenated text strings (one per group)
    """
    from .config import (
        SAMPLE_NUM_FOR_EACH_DATA,
        INTERVENE_DATA_NUM,
        SAMPLE_SESSION_NUM
    )
    
    # Use config defaults if not provided
    if sample_num_for_each_data is None:
        sample_num_for_each_data = SAMPLE_NUM_FOR_EACH_DATA
    if intervene_data_num is None:
        intervene_data_num = INTERVENE_DATA_NUM
    if sample_session_num is None:
        sample_session_num = SAMPLE_SESSION_NUM
    
    # Step 1: Load data
    data = load_data(data_path)
    
    # Step 2: Sample intervention haystacks
    intervene_haystack_idx = sample_intervene_data(
        len(data),
        sample_num_for_each_data,
        intervene_data_num
    )
    
    # Limit groups if specified
    if limit_groups:
        intervene_haystack_idx = intervene_haystack_idx[:limit_groups]
    
    # Step 3: Sample sessions
    intervene_sampled_session_data = sample_session(
        data,
        intervene_haystack_idx,
        sample_session_num
    )
    
    # Step 4: Retrieve concatenated texts
    intervened_texts = []
    for haystack_group in tqdm(intervene_sampled_session_data, desc="Retrieving session data"):
        text = retrieve_session_data(data, haystack_group)
        intervened_texts.append(text)
    
    return data, intervene_sampled_session_data, intervened_texts
