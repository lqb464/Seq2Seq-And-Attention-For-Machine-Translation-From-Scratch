import argparse
import os
import pickle

import torch

from src.builders.build_translator import build_model
from src.training.evaluate import Evaluator
from src.utils import load_text_data, load_yaml, split_train_val


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained NMT checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = cfg["project"]["output_dir"]
    checkpoint_path = args.checkpoint or os.path.join(output_dir, "best_checkpoint.pt")

    with open(os.path.join(output_dir, "vocab", "src_vocab.pkl"), "rb") as f:
        src_vocab = pickle.load(f)
    with open(os.path.join(output_dir, "vocab", "tgt_vocab.pkl"), "rb") as f:
        tgt_vocab = pickle.load(f)

    src_lines = load_text_data(cfg["data"]["src_path"], cfg["data"].get("num_samples"))
    tgt_lines = load_text_data(cfg["data"]["tgt_path"], cfg["data"].get("num_samples"))
    _, val_pairs = split_train_val(
        src_lines, tgt_lines,
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        seed=cfg["project"].get("seed", 42),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        src_vocab, tgt_vocab,
        embedding_dim=cfg["model"].get("embedding_dim", 128),
        hidden_size=cfg["model"].get("hidden_size", 256),
        device=device,
        use_attention=cfg["model"].get("use_attention", True),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(model, src_vocab, tgt_vocab)
    metrics = evaluator.evaluate_dataset(
        val_pairs,
        max_samples=args.max_samples or cfg["evaluation"].get("max_eval_samples", 200),
        max_len=cfg["evaluation"].get("max_len", 50),
    )
    print(metrics)


if __name__ == "__main__":
    main()
