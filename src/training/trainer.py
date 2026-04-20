import csv
import json
import os
from typing import Optional

import torch


def clip_grad_norm(parameters, max_norm):
    params = list(parameters)
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += (p.grad ** 2).sum().item()
    total_norm = total_norm_sq ** 0.5

    if max_norm is not None and max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
    return total_norm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        tgt_vocab,
        epochs=15,
        evaluator=None,
        val_pairs=None,
        scheduler=None,
        save_dir="checkpoints",
        max_eval_samples=200,
        eval_max_len=50,
        log_every=100,
        grad_clip=1.0,
        teacher_forcing_start=0.6,
        teacher_forcing_end=0.2,
        early_stopping_patience: Optional[int] = None,
        config=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.tgt_vocab = tgt_vocab
        self.epochs = epochs
        self.evaluator = evaluator
        self.val_pairs = val_pairs if val_pairs is not None else []
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.max_eval_samples = max_eval_samples
        self.eval_max_len = eval_max_len
        self.log_every = log_every
        self.grad_clip = grad_clip
        self.teacher_forcing_start = teacher_forcing_start
        self.teacher_forcing_end = teacher_forcing_end
        self.early_stopping_patience = early_stopping_patience
        self.config = config or {}

        self.history = []
        self.best_bleu = float("-inf")
        self.best_loss = float("inf")
        self.no_improve_epochs = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _teacher_forcing_ratio(self, epoch: int) -> float:
        if self.epochs <= 1:
            return float(self.teacher_forcing_end)
        progress = (epoch - 1) / max(self.epochs - 1, 1)
        return float(self.teacher_forcing_start + progress * (self.teacher_forcing_end - self.teacher_forcing_start))

    def _strip_trailing_pad(self, seq, pad_idx=0):
        if seq.dim() != 1:
            seq = seq.view(-1)
        last = len(seq) - 1
        while last >= 0 and int(seq[last].item()) == pad_idx:
            last -= 1
        return seq[:max(last + 1, 1)]

    def _compute_seq_loss(self, decoder_outputs, tgt_seq):
        loss = 0.0
        valid_steps = 0
        for t in range(len(decoder_outputs)):
            target_idx = tgt_seq[t + 1].item() if (t + 1) < len(tgt_seq) else self.tgt_vocab.EOS_IDX
            if target_idx == self.tgt_vocab.PAD_IDX:
                continue
            pred_step = decoder_outputs[t].unsqueeze(0)
            target_step = torch.tensor([target_idx], device=pred_step.device)
            loss = loss + self.criterion(pred_step, target_step)
            valid_steps += 1
        if valid_steps > 0:
            return loss / valid_steps
        return torch.tensor(0.0, requires_grad=True, device=decoder_outputs.device)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        teacher_forcing_ratio = self._teacher_forcing_ratio(epoch)
        device = next(self.model.parameters()).device

        for batch_idx, (src_batch, tgt_batch) in enumerate(self.train_loader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            self.optimizer.zero_grad()
            batch_loss = None

            for i in range(src_batch.shape[0]):
                src_seq = self._strip_trailing_pad(src_batch[i], pad_idx=0)
                tgt_seq = self._strip_trailing_pad(tgt_batch[i], pad_idx=self.tgt_vocab.PAD_IDX)
                decoder_outputs, _ = self.model(
                    src_tokens=src_seq,
                    tgt_tokens=tgt_seq,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )
                seq_loss = self._compute_seq_loss(decoder_outputs, tgt_seq)
                batch_loss = seq_loss if batch_loss is None else batch_loss + seq_loss

            batch_loss = batch_loss / src_batch.shape[0]
            batch_loss.backward()
            grad_norm = clip_grad_norm(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_grad_norm += grad_norm

            if self.log_every and (batch_idx + 1) % self.log_every == 0:
                print(
                    f"Epoch {epoch:02d} | Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss={batch_loss.item():.4f} | GradNorm={grad_norm:.4f} | TF={teacher_forcing_ratio:.3f}"
                )

        avg_loss = epoch_loss / max(len(self.train_loader), 1)
        avg_grad_norm = epoch_grad_norm / max(len(self.train_loader), 1)
        print(f"Epoch {epoch:02d}/{self.epochs} | AvgLoss={avg_loss:.4f} | AvgGradNorm={avg_grad_norm:.4f}")
        return avg_loss, avg_grad_norm, teacher_forcing_ratio

    def _save_history(self):
        json_path = os.path.join(self.save_dir, "train_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(self.save_dir, "train_history.csv")
        if not self.history:
            return
        keys = ["epoch", "avg_loss", "grad_norm", "teacher_forcing_ratio", "lr", "bleu", "rouge_l", "chrf", "is_best"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow({k: row.get(k) for k in keys})

    def save_checkpoint(self, epoch, avg_loss, metrics=None, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "avg_loss": float(avg_loss),
            "metrics": metrics or {},
            "best_bleu": None if self.best_bleu == float("-inf") else float(self.best_bleu),
            "best_loss": None if self.best_loss == float("inf") else float(self.best_loss),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if hasattr(self.optimizer, "state_dict") else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None and hasattr(self.scheduler, "state_dict") else None,
            "history": self.history,
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, "last_checkpoint.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_model_weights.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best_checkpoint.pt"))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model_weights.pt"))
        self._save_history()

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=next(self.model.parameters()).device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is not None and hasattr(self.optimizer, "load_state_dict"):
            self.optimizer.load_state_dict(opt_state)
        sched_state = checkpoint.get("scheduler_state_dict")
        if sched_state is not None and self.scheduler is not None and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(sched_state)
        self.history = checkpoint.get("history", [])
        self.best_bleu = checkpoint.get("best_bleu") or float("-inf")
        self.best_loss = checkpoint.get("best_loss") or float("inf")
        return checkpoint.get("epoch", 0) + 1, checkpoint

    def train(self, start_epoch=1):
        print("\n=== TRAINING ===")
        for epoch in range(start_epoch, self.epochs + 1):
            avg_loss, grad_norm, tf_ratio = self.train_epoch(epoch)
            metrics = {}
            if self.evaluator is not None and len(self.val_pairs) > 0:
                metrics = self.evaluator.evaluate_dataset(
                    self.val_pairs,
                    max_samples=self.max_eval_samples,
                    max_len=self.eval_max_len,
                )

            bleu = metrics.get("bleu") if isinstance(metrics, dict) else None
            if bleu is not None:
                is_best = bleu > self.best_bleu
                if is_best:
                    self.best_bleu = bleu
                    self.no_improve_epochs = 0
                else:
                    self.no_improve_epochs += 1
            else:
                is_best = avg_loss < self.best_loss
                self.no_improve_epochs = 0 if is_best else self.no_improve_epochs + 1

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss

            if self.scheduler is not None:
                self.scheduler.step(epoch)
            lr = self.scheduler.get_lr() if self.scheduler is not None else getattr(self.optimizer, "lr", None)

            row = {
                "epoch": epoch,
                "avg_loss": float(avg_loss),
                "grad_norm": float(grad_norm),
                "teacher_forcing_ratio": float(tf_ratio),
                "lr": None if lr is None else float(lr),
                "bleu": metrics.get("bleu") if isinstance(metrics, dict) else None,
                "rouge_l": metrics.get("rouge_l") if isinstance(metrics, dict) else None,
                "chrf": metrics.get("chrf") if isinstance(metrics, dict) else None,
                "is_best": bool(is_best),
            }
            self.history.append(row)
            self.save_checkpoint(epoch=epoch, avg_loss=avg_loss, metrics=metrics, is_best=is_best)
            print(f"[Checkpoint] Saved | best BLEU={self.best_bleu if self.best_bleu != float('-inf') else 0.0:.4f}")

            if self.early_stopping_patience is not None and self.no_improve_epochs >= self.early_stopping_patience:
                print(f"[EarlyStopping] Stop after {self.no_improve_epochs} non-improving epochs.")
                break

        print("\n[Done] Training finished.")
        return self.history
