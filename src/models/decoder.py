import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder:
    - only calls self.rnn.step(...)
    - attention is optional
    - each forward_step is one decode step
    """

    def __init__(self, rnn_model, attention_model=None, output_vocab_size=None):
        super().__init__()
        self.rnn = rnn_model
        self.attention = attention_model
        self.output_vocab_size = output_vocab_size
        self.hidden_size = rnn_model.hidden_size

        if output_vocab_size is None:
            raise ValueError("output_vocab_size must not be None")

        proj_input_dim = self.hidden_size * 2 if attention_model is not None else self.hidden_size
        self.W_vocab = nn.Parameter(torch.randn(output_vocab_size, proj_input_dim) * 0.01)
        self.b_vocab = nn.Parameter(torch.zeros(output_vocab_size, 1))

    def _extract_hidden(self, state):
        return state[0] if isinstance(state, tuple) else state

    def forward_step(self, x_t, prev_state, encoder_outputs=None):
        """
        Args:
            x_t: embedding vector shape (embedding_dim,) or (embedding_dim, 1)
            prev_state:
                - VanillaRNN / GRU: h_prev
                - LSTM: (h_prev, c_prev)
            encoder_outputs:
                list of hidden states from encoder, each element shape (hidden_size, 1)

        Returns:
            logits: shape (vocab_size,)
            new_state:
                - VanillaRNN / GRU: h_t
                - LSTM: (h_t, c_t)
            attn_weights: shape (src_len,) or None
        """
        device = self.W_vocab.device
        x_t = x_t.view(-1, 1).to(device)

        new_state = self.rnn.step(x_t, prev_state)
        h_t = self._extract_hidden(new_state)

        attn_weights = None

        if self.attention is not None and encoder_outputs is not None:
            context, attn_weights = self.attention(h_t, encoder_outputs)
            combined = torch.cat((h_t, context), dim=0)
            logits = self.W_vocab @ combined + self.b_vocab
        else:
            logits = self.W_vocab @ h_t + self.b_vocab

        logits = logits.squeeze(1)
        return logits, new_state, attn_weights