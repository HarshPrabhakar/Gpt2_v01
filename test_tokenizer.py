"""
=========================================================
Project : MyGPT2
File    : test_tokenizer.py
Purpose : Test the trained tokenizer.
=========================================================
"""

from pathlib import Path

from tokenizer.my_tokenizer import MyGPTTokenizer


def print_separator():

    print("=" * 80)


def main():

    tokenizer_path = Path(
        "artifacts/tokenizer/tokenizer.json"
    )

    tokenizer = MyGPTTokenizer.load(
        tokenizer_path
    )

    print_separator()

    print("Tokenizer Successfully Loaded")

    print_separator()

    print()

    print(
        f"Vocabulary Size : {tokenizer.vocabulary_size:,}"
    )

    print()

    test_sentences = [

        "Hello World!",

        "My name is Harsh Prabhakar.",

        "Artificial Intelligence is changing the world.",

        "Once upon a time there was a brave little dragon.",

        "GPT models learn language using transformers."

    ]

    for index, sentence in enumerate(test_sentences, start=1):

        print_separator()

        print(f"Example {index}")

        print_separator()

        print()

        print("Original Text")

        print(sentence)

        print()

        token_ids = tokenizer.encode(sentence)

        print("Token IDs")

        print(token_ids)

        print()

        print(f"Number of Tokens : {len(token_ids)}")

        print()

        decoded = tokenizer.decode(token_ids)

        print("Decoded Text")

        print(decoded)

        print()

        print("-" * 80)

        print()


if __name__ == "__main__":

    main()