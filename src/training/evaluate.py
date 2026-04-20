import torch

from src.training.metrics import compute_bleu, compute_chrf, compute_rouge_l


class Evaluator:
    """Evaluate and translate with the Seq2Seq model."""

    def __init__(self, model, src_vocab, tgt_vocab):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def translate_sentence(self, sentence, max_len=50):
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            src_indices = [self.src_vocab.SOS_IDX] + self.src_vocab.numericalize(sentence) + [self.src_vocab.EOS_IDX]
            src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device)

            decoder_outputs, attn_weights = self.model(
                src_tokens=src_tensor,
                tgt_tokens=None,
                teacher_forcing_ratio=0.0,
                max_len=max_len,
            )

            translated = []
            for output in decoder_outputs:
                token_idx = output.argmax(dim=-1).item()
                if token_idx == self.tgt_vocab.EOS_IDX:
                    break
                if token_idx in (self.tgt_vocab.SOS_IDX, self.tgt_vocab.PAD_IDX):
                    continue
                translated.append(self.tgt_vocab.idx2word.get(token_idx, self.tgt_vocab.UNK_TOKEN))

        if was_training:
            self.model.train()
        return translated, attn_weights

    def evaluate_dataset(self, test_pairs, max_samples=None, max_len=50):
        if max_samples is not None:
            test_pairs = test_pairs[:max_samples]

        if len(test_pairs) == 0:
            print("[Eval] No validation/test data.")
            return {"bleu": 0.0, "rouge_l": 0.0, "chrf": 0.0}

        totals = {"bleu": 0.0, "rouge_l": 0.0, "chrf": 0.0}
        examples = []

        for src_sentence, tgt_sentence in test_pairs:
            hyp, _ = self.translate_sentence(src_sentence, max_len=max_len)
            ref = self.tgt_vocab.tokenizer.tokenize(tgt_sentence)
            totals["bleu"] += compute_bleu(hyp, ref)
            totals["rouge_l"] += compute_rouge_l(hyp, ref)
            totals["chrf"] += compute_chrf(hyp, ref)
            if len(examples) < 3:
                examples.append({"src": src_sentence, "ref": tgt_sentence, "hyp": " ".join(hyp)})

        metrics = {k: v / max(len(test_pairs), 1) for k, v in totals.items()}
        metrics["examples"] = examples
        print(
            f"[Eval] BLEU={metrics['bleu']:.4f} | "
            f"ROUGE-L={metrics['rouge_l']:.4f} | chrF={metrics['chrf']:.4f}"
        )
        return metrics
