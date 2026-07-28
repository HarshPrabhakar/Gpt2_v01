"""
=========================================================
Project : MyGPT2
File    : train_tokenizer.py
Purpose : Train the MyGPT2 tokenizer.
=========================================================
"""

from pathlib import Path
import time

from tokenizer.tokenizer_config import TokenizerConfig
from tokenizer.tokenizer_trainer import TokenizerTrainer


def main() -> None:
    """
    Train and save the tokenizer.
    """

    print("=" * 70)
    print("                 MyGPT2 Tokenizer Training")
    print("=" * 70)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = TokenizerConfig()

    dataset_paths = [

        Path("datasets/TinyStories/raw"),

        Path("datasets/WikiText103/raw"),

        Path("datasets/OpenWebText/raw"),

        Path("datasets/FineWeb/raw"),

    ]

    print("\nDatasets")

    for dataset in dataset_paths:
        print(f"  • {dataset}")

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    trainer = TokenizerTrainer(config)

    print("\nStarting tokenizer training...\n")

    start = time.time()

    tokenizer = trainer.train(dataset_paths)

    end = time.time()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_dir = config.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_path = output_dir / "tokenizer.json"

    tokenizer.save(tokenizer_path)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 70)

    print("Tokenizer training completed successfully.")

    print(f"\nVocabulary Size : {tokenizer.vocabulary_size:,}")

    print(f"Tokenizer Saved : {tokenizer_path}")

    print(f"Training Time   : {end - start:.2f} seconds")

    print("=" * 70)


if __name__ == "__main__":
    main()