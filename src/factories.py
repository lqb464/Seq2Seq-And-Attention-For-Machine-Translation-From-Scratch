import torch
from src.builders.build_translator import build_model
from src.data.dataloader import get_dataloader
from src.data.vocabulary import Vocabulary
from src.training.losses import get_loss_function
from src.training.optimizers import get_optimizer
from src.training.schedulers import get_scheduler
from src.training.trainer import Trainer
from src.training.evaluate import Evaluator

_CACHE = {}

def build_experiment(cfg, src_lines, tgt_lines, device):
    """Centralized factory to build all components."""
    # Vocab
    src_vocab = Vocabulary(tokenizer_mode=cfg["data"].get("src_tokenizer", "basic"))
    tgt_vocab = Vocabulary(tokenizer_mode=cfg["data"].get("tgt_tokenizer", "basic"))
    src_vocab.build_vocabulary(src_lines, min_freq=cfg["data"].get("min_freq", 2))
    tgt_vocab.build_vocabulary(tgt_lines, min_freq=cfg["data"].get("min_freq", 2))

    # Model
    model = build_model(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        embedding_dim=cfg["model"].get("embedding_dim", 128),
        hidden_size=cfg["model"].get("hidden_size", 256),
        device=device,
        use_attention=cfg["model"].get("use_attention", True),
        attention_type=cfg["model"].get("attention_type", "bahdanau"),
    )

    # Optimizer
    optimizer = get_optimizer(model, lr=cfg["training"].get("lr", 0.001), opt_type=cfg["training"].get("optimizer", "adam"))

    # Scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=cfg["scheduler"].get("type", "step"),
        step_size=cfg["scheduler"].get("step_size", 5),
        gamma=cfg["scheduler"].get("gamma", 0.5),
        warmup_steps=cfg["scheduler"].get("warmup_steps", 5),
    )

    # Loss
    criterion = get_loss_function(pad_idx=tgt_vocab.PAD_IDX)

    # Evaluator
    evaluator = Evaluator(model=model, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    return {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "evaluator": evaluator,
    }