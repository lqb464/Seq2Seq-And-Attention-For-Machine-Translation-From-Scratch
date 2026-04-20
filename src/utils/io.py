import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_text_data(file_path: str, num_samples: Optional[int] = None) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:num_samples] if num_samples is not None else lines


def split_train_val(src_lines: List[str], tgt_lines: List[str], val_ratio: float = 0.1, seed: int = 42):
    import random

    pairs = list(zip(src_lines, tgt_lines))
    random.Random(seed).shuffle(pairs)
    if len(pairs) <= 1:
        return pairs, []
    val_size = max(1, int(len(pairs) * val_ratio))
    return pairs[val_size:], pairs[:val_size]


def unzip_pairs(pairs: Iterable[Tuple[str, str]]):
    pairs = list(pairs)
    if not pairs:
        return [], []
    src, tgt = zip(*pairs)
    return list(src), list(tgt)


def filter_by_length(src_lines, tgt_lines, src_vocab, tgt_vocab, max_src_len=None, max_tgt_len=None):
    """Filter very long pairs to keep RNN training stable and faster."""
    kept_src, kept_tgt = [], []
    for src, tgt in zip(src_lines, tgt_lines):
        src_len = len(src_vocab.tokenizer.tokenize(src)) + 2
        tgt_len = len(tgt_vocab.tokenizer.tokenize(tgt)) + 2
        if max_src_len is not None and src_len > max_src_len:
            continue
        if max_tgt_len is not None and tgt_len > max_tgt_len:
            continue
        kept_src.append(src)
        kept_tgt.append(tgt)
    return kept_src, kept_tgt
