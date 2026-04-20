# RNN Architectures and NMT Implementation

This project is learning-first.

The main goal is to build and understand core recurrent architectures from scratch, then apply them to a practical task:

1. Build RNN foundations: Vanilla RNN, LSTM, GRU.
2. Compose them into Seq2Seq and Attention mechanisms.
3. Use that foundation to run an English-to-Vietnamese Neural Machine Translation (NMT) pipeline.

In short: **RNN architectures are the core study target, and NMT is the application built on top of that knowledge.**

## Learning Progression

### 1) Core recurrent layers (from scratch)
- `src/models/layers.py`
  - `VanillaRNN`
  - `LSTM`
  - `GRU`
  - `Embedding`

### 2) Seq2Seq modeling
- `src/models/encoder.py`
- `src/models/decoder.py`
- `src/models/seq2seq.py`

### 3) Attention mechanisms
- `src/models/attention.py`
  - `BahdanauAttention` (additive)
  - `LuongAttention` (multiplicative/general)

### 4) Full NMT pipeline
- `src/builders/build_translator.py` builds the translator used in training/inference.
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/translate.py` provide end-to-end experimentation.

## Repository Structure

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── en_sents
│   └── vi_sents
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── translate.py
│   └── test.py
├── src/
│   ├── builders/
│   │   └── build_translator.py
│   ├── data/
│   ├── models/
│   │   ├── layers.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── seq2seq.py
│   │   └── attention.py
│   ├── training/
│   ├── utils/
│   └── factories.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

Put parallel data files in `data/`:
- `data/en_sents`: English sentences, one sentence per line.
- `data/vi_sents`: Vietnamese sentences, aligned line-by-line with `en_sents`.

Train/validation split is created automatically in code.

## Quick Start

### Train

```bash
# Train with defaults from configs/default.yaml
python scripts/train.py

# Explicit config path
python scripts/train.py --config configs/default.yaml

# Train baseline Seq2Seq without attention
python scripts/train.py --no_attention

# Override selected settings from CLI
python scripts/train.py --epochs 10 --batch_size 32 --lr 0.0005
```

### Evaluate

```bash
# Evaluate best checkpoint in output dir
python scripts/evaluate.py

# Evaluate a specific checkpoint
python scripts/evaluate.py --checkpoint outputs/seq2seq_attention/best_checkpoint.pt

# Limit evaluation samples
python scripts/evaluate.py --max_samples 200
```

### Translate

```bash
# Translate one sentence
python scripts/translate.py --sentence "Hello world"

# Translate using a specific checkpoint
python scripts/translate.py --sentence "How are you?" --checkpoint outputs/seq2seq_attention/best_checkpoint.pt
```

### Smoke Test

```bash
python scripts/train.py --epochs 1 --num_samples 100 --batch_size 8 --output_dir outputs/smoke_test
```

## Configuration

Main experiment settings are in `configs/default.yaml`:
- `model`: `use_attention`, `attention_type`, `embedding_dim`, `hidden_size`
- `training`: `optimizer`, `lr`, `batch_size`, `epochs`, `resume`
- `data`: tokenizer mode (`basic`/`bpe`), length limits, sampling
- `scheduler`: scheduler type and parameters
- `evaluation`: evaluation sample cap, decode length, metrics

## Outputs

After training, artifacts are saved in `project.output_dir` (default: `outputs/seq2seq_attention`):
- checkpoints (`best_checkpoint.pt`, `last_checkpoint.pt`)
- vocab files (`vocab/src_vocab.pkl`, `vocab/tgt_vocab.pkl`)
- resolved config (`resolved_config.json`)
- training history (`train_history.json`, `train_history.csv`)
- training plot (`training_plot.png`)

## Notes

- The project follows a from-scratch spirit for core sequence modeling components while still using practical PyTorch utilities where helpful.
- Default training pipeline currently instantiates GRU-based encoder/decoder in `src/builders/build_translator.py`.
- You can extend this project by swapping recurrent cells in the builder or adding new attention variants.