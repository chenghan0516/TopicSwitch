# Prepare Data Parameters Reference

This document explains the parameters used in the `prepare_data_step` function for creating intervened datasets from LongMemEval data.

## Overview

The `prepare_data_step` function creates synthetic multi-topic dialogues by combining sessions from different haystacks (topics). This simulates natural topic switching in conversations and is used to study proactive interference in multi-turn dialogues.

## Parameters

### 1. `sample_num` (also `sample_num_for_each_data`)

**Type:** `Optional[int]`  
**Default:** `1` (from `SAMPLE_NUM_FOR_EACH_DATA` in `src/config.py`)

**Description:**  
Number of intervention groups to create for each haystack in the dataset.

**How it works:**
- For each haystack in the dataset, the system creates this many different intervention combinations
- Each combination includes the current haystack plus other randomly selected haystacks
- The total number of intervention groups = `(number of haystacks) × sample_num`

**Example:**
- If you have 100 haystacks and `sample_num=1`, you get 100 intervention groups
- If `sample_num=2`, you get 200 intervention groups (2 different combinations per haystack)

**Use case:**  
Increase this value to create more diverse intervention combinations for each haystack, useful for more comprehensive experiments.

---

### 2. `intervene_num` (also `intervene_data_num`)

**Type:** `Optional[int]`  
**Default:** `2` (from `INTERVENE_DATA_NUM` in `src/config.py`)

**Description:**  
Number of haystacks (topics) to combine in each intervention group.

**How it works:**
- Each intervention group contains this many haystacks
- The system selects one haystack as the "anchor" (the one being tested)
- Then randomly picks `intervene_num - 1` other haystacks to combine with it
- Sessions from these haystacks are interleaved to create topic-switching dialogues

**Example:**
- If `intervene_num=2`, each group contains 2 haystacks (e.g., Haystack 5 + Haystack 12)
- If `intervene_num=3`, each group contains 3 haystacks (e.g., Haystack 5 + Haystack 12 + Haystack 33)

**Use case:**  
Increase this value to create more complex multi-topic dialogues with more topic switches. Higher values may increase proactive interference effects.

---

### 3. `session_num` (also `sample_session_num`)

**Type:** `Optional[int]`  
**Default:** `2` (from `SAMPLE_SESSION_NUM` in `src/config.py`)

**Description:**  
Number of sessions to sample from each haystack in an intervention group.

**How it works:**
- For each haystack in a group, the system samples this many sessions
- The sampling process prioritizes sessions that contain "answer" in their session IDs
- If there are answer sessions, they are included first
- Remaining slots are filled by randomly sampling from other available sessions
- All selected session indices are sorted

**Example:**
- If `session_num=2` and a group has 3 haystacks, you get:
  - 2 sessions from Haystack 1
  - 2 sessions from Haystack 2
  - 2 sessions from Haystack 3
  - Total: 6 sessions per intervention group

**Use case:**  
Increase this value to include more conversation turns per topic, creating longer dialogues. This may help study interference effects over longer sequences.

---

### 4. `limit_groups`

**Type:** `Optional[int]`  
**Default:** `None` (no limit)

**Description:**  
Limits the total number of intervention groups created. Useful for testing or debugging.

**How it works:**
- After creating all intervention groups based on `sample_num`, the system truncates the list to the first `limit_groups` groups
- If `None`, all groups are used
- Applied before session sampling, so it affects the total number of final intervened texts

**Example:**
- If you would normally get 200 groups but set `limit_groups=10`, only the first 10 groups are processed
- This is useful for quick testing without processing the entire dataset

**Use case:**  
Use this parameter during development or testing to quickly validate the pipeline with a small subset of data.

---

## Complete Example

### Scenario
- Dataset: 100 haystacks
- `sample_num=1`
- `intervene_num=2`
- `session_num=2`
- `limit_groups=None`

### Process Flow

1. **Create intervention groups:**
   - Total groups = 100 × 1 = 100 groups
   - Each group contains 2 haystacks

2. **Sample sessions:**
   - From each haystack: sample 2 sessions
   - Each group has 4 sessions total (2 from each of 2 haystacks)

3. **Concatenate sessions:**
   - Sessions are interleaved to create topic-switching dialogues
   - Result: 100 intervened text groups

### With `limit_groups=5`
- Only the first 5 groups are processed
- Result: 5 intervened text groups

---

## Visual Summary

```
Total Groups = (number of haystacks) × sample_num
              ↓ (limited by limit_groups if set)

Each Group:
  ├─ Haystack 1 → session_num sessions
  ├─ Haystack 2 → session_num sessions
  └─ ... (intervene_num haystacks total)
  
Final Output:
  └─ List of intervened texts (one per group)
```

---

## Parameter Relationships

- **`sample_num`** controls **diversity** of combinations per haystack
- **`intervene_num`** controls **complexity** of topic switching (number of topics per dialogue)
- **`session_num`** controls **length** of dialogue per topic
- **`limit_groups`** controls **scale** of the experiment (total number of test cases)

---

## Default Configuration

From `src/config.py`:
```python
SAMPLE_NUM_FOR_EACH_DATA = 1
INTERVENE_DATA_NUM = 2
SAMPLE_SESSION_NUM = 2
```

This default configuration creates:
- One intervention group per haystack
- Two haystacks per group (simple topic switch)
- Two sessions per haystack (moderate dialogue length)

---

## Command-Line Usage

```bash
python main.py prepare_data \
    --data_path data/longmemeval/longmemeval_s_cleaned.json \
    --output_dir output/ \
    --sample_num 1 \
    --intervene_num 2 \
    --session_num 2 \
    --limit_groups 10
```

---

## Notes

- All parameters are optional and will use config defaults if not specified
- The random seed is set in `src/config.py` (default: 42) for reproducibility
- Sessions with "answer" in their IDs are prioritized to ensure answer sessions are included
- The final intervened texts are saved to `intervened_texts.txt` with group separators
- Metadata including all parameter values is saved to `metadata.json`
