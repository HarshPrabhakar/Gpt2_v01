from pathlib import Path
from datasets import load_from_disk
from tokenizer.my_tokenizer import MyGPTTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent

TOKENIZER_PATH = (
    PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
)

DATASETS = [
    ("TinyStories", PROJECT_ROOT / "datasets" / "TinyStories" / "raw"),
    ("WikiText103", PROJECT_ROOT / "datasets" / "WikiText103" / "raw"),
    ("OpenWebText", PROJECT_ROOT / "datasets" / "OpenWebText" / "raw"),
    ("FineWeb", PROJECT_ROOT / "datasets" / "FineWeb" / "raw"),
]

SEQUENCE_LENGTH = 512
BATCH_SIZE = 8

EXCLUDED_SPLITS = {
    "test",
    "validation",
    "valid",
    "dev",
    "eval",
    "evaluation",
}


def get_training_splits(dataset):
    if hasattr(dataset, "items"):
        splits = []

        for name, split in dataset.items():
            normalized = str(name).strip().lower()

            if normalized in EXCLUDED_SPLITS:
                continue

            if normalized == "train":
                splits.append((str(name), split))

        if splits:
            return splits

        for name, split in dataset.items():
            normalized = str(name).strip().lower()

            if normalized not in EXCLUDED_SPLITS:
                return [(str(name), split)]

        return []

    return [("dataset", dataset)]


def count_samples(tokenizer, text):
    if not isinstance(text, str):
        if text is None:
            return 0
        text = str(text)

    text = text.strip()

    if not text:
        return 0

    token_ids = tokenizer.encode(text)

    token_count = len(token_ids)

    if token_count < SEQUENCE_LENGTH + 1:
        return 0

    # Exactly matches GPTTextDataset._create_samples()
    return (token_count - 1) // SEQUENCE_LENGTH


print("=" * 80)
print("MyGPT2 EXACT FULL-DATASET STEP CALCULATOR")
print("=" * 80)
print()
print(f"Tokenizer       : {TOKENIZER_PATH}")
print(f"Sequence Length : {SEQUENCE_LENGTH}")
print(f"Batch Size      : {BATCH_SIZE}")
print(f"Stride          : {SEQUENCE_LENGTH}")
print()
print("Loading tokenizer...")

tokenizer = MyGPTTokenizer.load(TOKENIZER_PATH)

print("Tokenizer loaded.")
print(f"Vocabulary Size : {tokenizer.vocabulary_size:,}")
print()

total_documents = 0
total_processed = 0
total_skipped = 0
total_samples = 0
total_tokens = 0

dataset_results = []

for dataset_name, dataset_path in DATASETS:

    print("=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)
    print(f"Path: {dataset_path}")
    print()

    dataset = load_from_disk(str(dataset_path))

    dataset_samples = 0
    dataset_documents = 0
    dataset_processed = 0
    dataset_skipped = 0

    training_splits = get_training_splits(dataset)

    for split_name, split in training_splits:

        print(f"Training Split: {split_name}")
        print(f"Rows: {len(split):,}")
        print()

        for index, record in enumerate(split):

            dataset_documents += 1
            total_documents += 1

            text = record.get("text", "")

            if text is None:
                dataset_skipped += 1
                total_skipped += 1
                continue

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if not text:
                dataset_skipped += 1
                total_skipped += 1
                continue

            try:
                token_ids = tokenizer.encode(text)
            except Exception as exc:
                dataset_skipped += 1
                total_skipped += 1

                print()
                print(f"WARNING: tokenizer failed at row {index:,}")
                print(f"Error: {exc}")
                continue

            if not token_ids:
                dataset_skipped += 1
                total_skipped += 1
                continue

            dataset_processed += 1
            total_processed += 1

            token_count = len(token_ids)

            samples = count_samples(tokenizer, text)

            dataset_samples += samples
            total_samples += samples

            total_tokens += samples * SEQUENCE_LENGTH

            if (index + 1) % 100000 == 0:
                print(
                    f"  Processed: {index + 1:,} | "
                    f"Samples: {dataset_samples:,}"
                )

    dataset_steps = (
        (dataset_samples + BATCH_SIZE - 1)
        // BATCH_SIZE
    )

    print()
    print(f"{dataset_name} RESULTS")
    print("-" * 80)
    print(f"Documents seen     : {dataset_documents:,}")
    print(f"Documents processed: {dataset_processed:,}")
    print(f"Documents skipped  : {dataset_skipped:,}")
    print(f"Training samples   : {dataset_samples:,}")
    print(f"Individual steps*  : {dataset_steps:,}")
    print()

    dataset_results.append(
        (
            dataset_name,
            dataset_documents,
            dataset_processed,
            dataset_skipped,
            dataset_samples,
        )
    )


final_steps = (
    total_samples + BATCH_SIZE - 1
) // BATCH_SIZE

print()
print("=" * 80)
print("FINAL TRAINING CALCULATION")
print("=" * 80)
print()

for result in dataset_results:
    name, docs, processed, skipped, samples = result

    print(
        f"{name:15} : "
        f"{samples:,} training samples"
    )

print()
print("-" * 80)
print(f"Total documents       : {total_documents:,}")
print(f"Documents processed   : {total_processed:,}")
print(f"Documents skipped     : {total_skipped:,}")
print(f"Total training samples: {total_samples:,}")
print(f"Sequence length       : {SEQUENCE_LENGTH:,}")
print(f"Batch size            : {BATCH_SIZE:,}")
print()
print(
    "Exact optimizer steps : "
    f"{final_steps:,}"
)
print()
print(
    "Formula:"
)
print(
    "ceil(total_training_samples / batch_size)"
)
print(
    f"ceil({total_samples:,} / {BATCH_SIZE}) "
    f"= {final_steps:,}"
)
print()
print("=" * 80)
print("ONE COMPLETE PASS = EXACTLY THE STEP COUNT ABOVE")
print("=" * 80)
