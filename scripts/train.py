import argparse
import os
import pickle
from copy import deepcopy

import torch

from src.data.vocabulary import Vocabulary
from src.data.dataloader import get_dataloader
from src.builders.build_translator import build_model
from src.training.losses import get_loss_function
from src.training.optimizers import get_optimizer
from src.training.schedulers import get_scheduler
from src.training.trainer import Trainer
from src.training.evaluate import Evaluator
from src.training.visualize import plot_training_history
from src.utils import filter_by_length, load_text_data, load_yaml, save_json, set_seed, split_train_val, unzip_pairs


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args():
    parser = argparse.ArgumentParser(description="Train Seq2Seq Attention NMT")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    cfg = deepcopy(cfg)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.num_samples is not None:
        cfg["data"]["num_samples"] = args.num_samples
    if args.output_dir is not None:
        cfg["project"]["output_dir"] = args.output_dir
    if args.no_attention:
        cfg["model"]["use_attention"] = False
    if args.resume:
        cfg["training"]["resume"] = True
    return cfg


def save_vocabularies(src_vocab, tgt_vocab, output_dir):
    vocab_dir = os.path.join(output_dir, "vocab")
    os.makedirs(vocab_dir, exist_ok=True)
    with open(os.path.join(vocab_dir, "src_vocab.pkl"), "wb") as f:
        pickle.dump(src_vocab, f)
    with open(os.path.join(vocab_dir, "tgt_vocab.pkl"), "wb") as f:
        pickle.dump(tgt_vocab, f)


def main():
    args = parse_args()
    cfg = apply_cli_overrides(load_yaml(args.config), args)
    set_seed(cfg["project"].get("seed", 42))

    output_dir = cfg["project"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    save_json(cfg, os.path.join(output_dir, "resolved_config.json"))

    print("=== 1. Load data and build vocabulary ===")
    src_lines = load_text_data(cfg["data"]["src_path"], cfg["data"].get("num_samples"))
    tgt_lines = load_text_data(cfg["data"]["tgt_path"], cfg["data"].get("num_samples"))
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Source and target line counts differ: {len(src_lines)} vs {len(tgt_lines)}")

    train_pairs, val_pairs = split_train_val(
        src_lines, tgt_lines,
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        seed=cfg["project"].get("seed", 42),
    )
    train_src, train_tgt = unzip_pairs(train_pairs)
    val_src, val_tgt = unzip_pairs(val_pairs)

    src_vocab = Vocabulary(tokenizer_mode=cfg["data"].get("src_tokenizer", "basic"))
    tgt_vocab = Vocabulary(tokenizer_mode=cfg["data"].get("tgt_tokenizer", "basic"))
    src_vocab.build_vocabulary(train_src, min_freq=cfg["data"].get("min_freq", 2))
    tgt_vocab.build_vocabulary(train_tgt, min_freq=cfg["data"].get("min_freq", 2))

    train_src, train_tgt = filter_by_length(
        train_src, train_tgt, src_vocab, tgt_vocab,
        max_src_len=cfg["data"].get("max_src_len"),
        max_tgt_len=cfg["data"].get("max_tgt_len"),
    )
    train_pairs = list(zip(train_src, train_tgt))
    save_vocabularies(src_vocab, tgt_vocab, output_dir)

    print(f"[Data] Train={len(train_pairs)} | Val={len(val_pairs)}")
    print(f"[Vocab] Source={len(src_vocab)} | Target={len(tgt_vocab)}")

    train_loader = get_dataloader(
        train_src, train_tgt, src_vocab, tgt_vocab,
        batch_size=cfg["training"].get("batch_size", 16),
        max_src_len_percentile=cfg["data"].get("max_src_len_percentile", 0.95),
        max_tgt_len_percentile=cfg["data"].get("max_tgt_len_percentile", 0.95),
        max_src_len=cfg["data"].get("max_src_len", 80),
        max_tgt_len=cfg["data"].get("max_tgt_len", 80),
    )

    print("\n=== 2. Build model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        embedding_dim=cfg["model"].get("embedding_dim", 128),
        hidden_size=cfg["model"].get("hidden_size", 256),
        device=device,
        use_attention=cfg["model"].get("use_attention", True),
        attention_type=cfg["model"].get("attention_type", "bahdanau"),
    )
    print(f"[Model] {'Seq2Seq + ' + cfg['model'].get('attention_type', 'bahdanau').capitalize() + ' Attention' if cfg['model'].get('use_attention', True) else 'Seq2Seq baseline'} on {device}")

    optimizer = get_optimizer(model, lr=cfg["training"].get("lr", 0.001), opt_type=cfg["training"].get("optimizer", "adam"))
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=cfg["scheduler"].get("type", "step"),
        step_size=cfg["scheduler"].get("step_size", 5),
        gamma=cfg["scheduler"].get("gamma", 0.5),
        warmup_steps=cfg["scheduler"].get("warmup_steps", 5),
    )
    criterion = get_loss_function(pad_idx=tgt_vocab.PAD_IDX)
    evaluator = Evaluator(model=model, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        tgt_vocab=tgt_vocab,
        epochs=cfg["training"].get("epochs", 15),
        evaluator=evaluator,
        val_pairs=val_pairs,
        scheduler=scheduler,
        save_dir=output_dir,
        max_eval_samples=cfg["evaluation"].get("max_eval_samples", 200),
        eval_max_len=cfg["evaluation"].get("max_len", 50),
        log_every=cfg["training"].get("log_every", 50),
        grad_clip=cfg["training"].get("grad_clip", 1.0),
        teacher_forcing_start=cfg["training"].get("teacher_forcing_start", 0.6),
        teacher_forcing_end=cfg["training"].get("teacher_forcing_end", 0.2),
        early_stopping_patience=cfg["training"].get("early_stopping_patience"),
        config=cfg,
    )

    last_checkpoint = os.path.join(output_dir, "last_checkpoint.pt")
    if cfg["training"].get("resume", False) and os.path.exists(last_checkpoint):
        start_epoch, _ = trainer.load_checkpoint(last_checkpoint)
        trainer.train(start_epoch=start_epoch)
    else:
        trainer.train(start_epoch=1)

    if len(trainer.history) > 0:
        plot_training_history(trainer.history, save_path=os.path.join(output_dir, "training_plot.png"))


if __name__ == "__main__":
    main()
