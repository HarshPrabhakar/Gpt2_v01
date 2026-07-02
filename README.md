# MyGPT2

A GPT-2 Small implementation built completely from scratch using **PyTorch**.

The goal of this project is to understand how modern Large Language Models (LLMs) work by implementing every major component manually instead of relying on pre-built transformer libraries.

This project focuses on learning the complete GPT-2 pipeline, including tokenization, transformer architecture, training, and text generation.

---

## Project Goals

* Build a Byte Pair Encoding (BPE) tokenizer
* Implement the GPT-2 architecture from scratch
* Train the model on a custom text dataset
* Generate coherent text using the trained model
* Understand every component of the Transformer architecture
* Maintain clean, modular, and well-documented code

---

## Project Structure

```text
MyGPT2/

├── data/
│   ├── raw/
│   └── processed/
│
├── tokenizer/
│   ├── bpe.py
│   └── tokenizer.py
│
├── model/
│   ├── attention.py
│   ├── transformer.py
│   └── gpt.py
│
├── train.py
├── generate.py
├── config.py
├── requirements.txt
└── README.md
```

---

## Features

### Current

* Project structure
* Configuration management

### Planned

* Byte Pair Encoding (BPE) Tokenizer
* Dataset preprocessing
* Multi-Head Self Attention
* Transformer Blocks
* GPT-2 Small Architecture
* Training Pipeline
* Checkpoint Saving
* Text Generation
* Model Evaluation

---

## GPT-2 Small Configuration

| Parameter       | Value        |
| --------------- | ------------ |
| Layers          | 12           |
| Attention Heads | 12           |
| Hidden Size     | 768          |
| Context Length  | 1024 Tokens  |
| Vocabulary Size | ~50,000      |
| Parameters      | ~124 Million |

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd MyGPT2
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset

The model will be trained on publicly available text datasets.

Possible datasets include:

* TinyStories
* WikiText
* OpenWebText (subset)
* Books
* Custom text corpus

The raw dataset will be stored inside:

```text
data/raw/
```

The processed dataset will be stored inside:

```text
data/processed/
```

---

## Training

Once implemented, training will start with:

```bash
python train.py
```

---

## Text Generation

After training, generate text using:

```bash
python generate.py
```

---

## Learning Objectives

This project is intended as an educational implementation.

Rather than using high-level libraries that abstract away the details, every major component will be implemented and studied individually.

Topics covered include:

* Tokenization
* Embeddings
* Positional Encoding
* Self-Attention
* Multi-Head Attention
* Feed Forward Networks
* Residual Connections
* Layer Normalization
* Transformer Blocks
* Language Modeling
* Text Generation

---

## Development Roadmap

* [ ] Project Setup
* [ ] Dataset Pipeline
* [ ] BPE Tokenizer
* [ ] Vocabulary Generation
* [ ] Token Encoding & Decoding
* [ ] Self-Attention
* [ ] Multi-Head Attention
* [ ] Feed Forward Network
* [ ] Transformer Block
* [ ] GPT-2 Model
* [ ] Training Loop
* [ ] Checkpoint Saving
* [ ] Text Generation
* [ ] Model Evaluation
* [ ] Documentation

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Pandas
* tqdm

---

## Hardware

Development Machine

* CPU: Intel Core i5-14600K
* GPU: NVIDIA RTX 5060 Ti (16 GB VRAM)
* RAM: 16 GB
* Storage: 1 TB NVMe SSD

---

## License

This project is intended for educational and research purposes.

---

## Acknowledgements

This project is inspired by the GPT-2 architecture and the original Transformer research.

The implementation is written from scratch as a learning exercise to understand the internal workings of modern language models.
