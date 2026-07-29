from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TokenizerConfig:

    vocab_size: int = 32000

    max_length: int = 512

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    output_dir: Path = Path("artifacts/tokenizer")

    tokenizer_name: str = "mygpt2_tokenizer"

    lowercase: bool = False

    min_frequency: int = 2

    show_progress: bool = True

    # Optional limit for faster experiments
    max_documents_per_dataset: int | None = None