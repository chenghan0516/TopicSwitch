"""
Configuration constants for the Topic Switch project.
"""
import random

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Data processing parameters
SAMPLE_NUM_FOR_EACH_DATA = 1
INTERVENE_DATA_NUM = 2
SAMPLE_SESSION_NUM = 2

# Model parameters
FLAN_T5_MODEL_NAME = "google/flan-t5-base"
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_SAVE_PATH = "/content/drive/MyDrive/Research/pretrained_models/llama-3.1-8B"

# Generation parameters
MAX_NEW_TOKENS = 256
MAX_TOKENS_PER_CHUNK = 4096

# Data paths
DEFAULT_DATA_PATH = "data/longmemeval/longmemeval_s_cleaned.json"

