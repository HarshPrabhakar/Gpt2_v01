"""
=========================================================
Project : MyGPT2
File    : config.py
Purpose : Global Project Configuration
Author  : Harsh Prabhakar
=========================================================
"""

from pathlib import Path

# =========================================================
# Project Information
# =========================================================

PROJECT_NAME = "MyGPT2"

VERSION = "0.1.0"

AUTHOR = "Harsh Prabhakar"

# =========================================================
# Project Root
# =========================================================

# Automatically finds the directory where config.py exists.
# This means you never have to hardcode your project path.

PROJECT_ROOT = Path(__file__).resolve().parent

# =========================================================
# Hugging Face Cache
# =========================================================

# All HuggingFace downloads will be stored here.

HF_CACHE_DIR = Path(r"D:\HuggingFaceCache")

# =========================================================
# Dataset Root
# =========================================================

DATASETS_DIR = PROJECT_ROOT / "datasets"

# =========================================================
# Individual Dataset Directories
# =========================================================

TINYSTORIES_DIR = DATASETS_DIR / "TinyStories"

WIKITEXT_DIR = DATASETS_DIR / "WikiText103"

OPENWEBTEXT_DIR = DATASETS_DIR / "OpenWebText"

FINEWEB_DIR = DATASETS_DIR / "FineWeb"

GUTENBERG_DIR = DATASETS_DIR / "Gutenberg"

# =========================================================
# Raw Dataset Directories
# =========================================================

TINYSTORIES_RAW = TINYSTORIES_DIR / "raw"

WIKITEXT_RAW = WIKITEXT_DIR / "raw"

OPENWEBTEXT_RAW = OPENWEBTEXT_DIR / "raw"

FINEWEB_RAW = FINEWEB_DIR / "raw"

GUTENBERG_RAW = GUTENBERG_DIR / "raw"

# =========================================================
# Processed Dataset Directories
# =========================================================

TINYSTORIES_PROCESSED = TINYSTORIES_DIR / "processed"

WIKITEXT_PROCESSED = WIKITEXT_DIR / "processed"

OPENWEBTEXT_PROCESSED = OPENWEBTEXT_DIR / "processed"

FINEWEB_PROCESSED = FINEWEB_DIR / "processed"

GUTENBERG_PROCESSED = GUTENBERG_DIR / "processed"

# =========================================================
# Corpus
# =========================================================

CORPUS_DIR = PROJECT_ROOT / "corpus"

TRAIN_CORPUS = CORPUS_DIR / "train.txt"

VALID_CORPUS = CORPUS_DIR / "validation.txt"

TEST_CORPUS = CORPUS_DIR / "test.txt"

MERGED_CORPUS = CORPUS_DIR / "merged.txt"

# =========================================================
# Tokenizer
# =========================================================

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"

VOCAB_FILE = TOKENIZER_DIR / "vocab.json"

MERGES_FILE = TOKENIZER_DIR / "merges.txt"

# =========================================================
# Model
# =========================================================

MODEL_DIR = PROJECT_ROOT / "model"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

LOG_DIR = PROJECT_ROOT / "logs"

# =========================================================
# Dataset Download Configuration
# =========================================================

# Download percentages.
# These keep the total dataset size manageable for local training.

OPENWEBTEXT_SPLIT = "train[:10%]"

FINEWEB_SPLIT = "train[:2%]"

TINYSTORIES_SPLIT = None

WIKITEXT_SPLIT = None

# =========================================================
# Create Required Directories Automatically
# =========================================================

DIRECTORIES = [

    HF_CACHE_DIR,

    DATASETS_DIR,

    TINYSTORIES_RAW,
    TINYSTORIES_PROCESSED,

    WIKITEXT_RAW,
    WIKITEXT_PROCESSED,

    OPENWEBTEXT_RAW,
    OPENWEBTEXT_PROCESSED,

    FINEWEB_RAW,
    FINEWEB_PROCESSED,

    GUTENBERG_RAW,
    GUTENBERG_PROCESSED,

    CORPUS_DIR,

    TOKENIZER_DIR,

    MODEL_DIR,

    CHECKPOINT_DIR,

    LOG_DIR

]

for directory in DIRECTORIES:

    directory.mkdir(parents=True, exist_ok=True)