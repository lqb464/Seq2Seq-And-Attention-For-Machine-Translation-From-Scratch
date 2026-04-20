import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    """
    Seq2Seq orchestration from scratch:
    - source embedding
    - encoder
    - autoregressive decoder
    - teacher forcing
    - attention is inside the decoder
    """

    def __init__(
        self,
        src_embedding,
        tgt_embedding,
        encoder,
        decoder,
        sos_token_id,
        eos_token_id=None
    ):
        super().__init__()
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

    def forward(
        self,
        src_tokens=None,
        tgt_tokens=None,
        teacher_forcing_ratio=0.0,
        max_len=None,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        """
        Args:
            src_tokens:
                tensor/list shape (src_len,)
            tgt_tokens:
                tensor/list shape (tgt_len,)
                if provided -> training mode with teacher forcing
            teacher_forcing_ratio:
                probability of using ground-truth token at the next step
            max_len:
                maximum number of decode steps during inference
                if tgt_tokens is provided and max_len=None then decode_steps = len(tgt_tokens) - 1

        Returns:
            logits_seq: tensor shape (out_len, vocab_size)
            attn_seq: tensor shape (out_len, src_len) or None
        """

        if src_tokens is None:
            raise ValueError("src_tokens must be provided")

        device = self.src_embedding.W.device

        if not torch.is_tensor(src_tokens):
            src_tokens = torch.tensor(src_tokens, dtype=torch.long, device=device)
        else:
            src_tokens = src_tokens.to(device=device, dtype=torch.long)

        if tgt_tokens is not None:
            if not torch.is_tensor(tgt_tokens):
                tgt_tokens = torch.tensor(tgt_tokens, dtype=torch.long, device=device)
            else:
                tgt_tokens = tgt_tokens.to(device=device, dtype=torch.long)

        src_embedded = self.src_embedding(src_tokens)
        encoder_outputs, encoder_final_state = self.encoder(src_embedded)

        decoder_state = encoder_final_state
        input_token = torch.tensor(self.sos_token_id, dtype=torch.long, device=device)

        if tgt_tokens is not None:
            decode_steps = max(0, tgt_tokens.size(0) - 1) if max_len is None else max_len
        else:
            if max_len is None:
                raise ValueError("During inference, max_len must be provided")
            decode_steps = max_len

        logits_list = []
        attn_list = []

        for t in range(decode_steps):
            x_t = self.tgt_embedding(input_token)

            logits_t, decoder_state, attn_t = self.decoder.forward_step(
                x_t=x_t,
                prev_state=decoder_state,
                encoder_outputs=encoder_outputs
            )

            logits_list.append(logits_t)

            if attn_t is not None:
                attn_list.append(attn_t)

            # Sampling logic
            if top_k is not None or top_p is not None:
                scaled_logits = logits_t / max(temperature, 1e-9)
                probs = torch.softmax(scaled_logits, dim=-1)
                
                if top_k is not None and top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
                    mask = torch.zeros_like(probs, dtype=torch.bool)
                    mask.scatter_(-1, top_k_indices, True)
                    probs[~mask] = 0.0
                
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
                    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                
                probs_sum = probs.sum(dim=-1, keepdim=True)
                probs = probs / torch.where(probs_sum > 0, probs_sum, torch.ones_like(probs_sum))
                if (probs > 0).any():
                    pred_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    pred_token = torch.argmax(logits_t, dim=-1).detach()
            else:
                pred_token = torch.argmax(logits_t, dim=-1).detach()

            use_teacher_forcing = False
            if tgt_tokens is not None:
                use_teacher_forcing = (
                    torch.rand(1, device=device).item() < teacher_forcing_ratio
                )

            if use_teacher_forcing and tgt_tokens is not None:
                input_token = tgt_tokens[t + 1]
            else:
                input_token = pred_token

            if tgt_tokens is None and self.eos_token_id is not None:
                if int(input_token.item()) == self.eos_token_id:
                    break

        if len(logits_list) == 0:
            logits_seq = torch.empty((0, self.decoder.output_vocab_size), device=device)
        else:
            logits_seq = torch.stack(logits_list, dim=0)

        if len(attn_list) == 0:
            attn_seq = None
        else:
            attn_seq = torch.stack(attn_list, dim=0)

        return logits_seq, attn_seq