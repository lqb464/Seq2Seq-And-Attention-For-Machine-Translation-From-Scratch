import argparse
import os
import pickle

import torch

from src.builders.build_translator import build_model
from src.training.evaluate import Evaluator
from src.utils import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Translate a sentence with a trained checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=50)
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
    tokens, _ = evaluator.translate_sentence(args.sentence, max_len=args.max_len)
    print(" ".join(tokens))


if __name__ == "__main__":
    main()
