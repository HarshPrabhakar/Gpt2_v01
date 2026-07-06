from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "datasets"

CORPUS_ROOT = PROJECT_ROOT / "corpus"

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

# ==========================================
# Hugging Face Cache
# ==========================================

HF_CACHE = Path(r"D:\HuggingFaceCache")

# ==========================================
# Dataset Paths
# ==========================================

TINYSTORIES_PATH = DATASET_ROOT / "TinyStories" / "raw"

WIKITEXT_PATH = DATASET_ROOT / "WikiText103" / "raw"

OPENWEBTEXT_PATH = DATASET_ROOT / "OpenWebText" / "raw"

FINEWEB_PATH = DATASET_ROOT / "FineWeb" / "raw"

GUTENBERG_PATH = DATASET_ROOT / "Gutenberg" / "raw"