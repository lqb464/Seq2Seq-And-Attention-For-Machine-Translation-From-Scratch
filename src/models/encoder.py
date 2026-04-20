import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder:
    - only loops through the sequence
    - calls self.rnn.step(...) and self.rnn.init_hidden(...)
    """

    def __init__(self, rnn_model):
        super().__init__()
        self.rnn = rnn_model
        self.hidden_size = rnn_model.hidden_size

    def forward(self, embedded_inputs):
        """
        embedded_inputs:
            - tensor shape (seq_len, embedding_dim)
            - or list of embedding vectors

        Returns:
            encoder_outputs: list of hidden states at each step, each element shape (hidden_size, 1)
            final_state:
                - VanillaRNN / GRU: h_T
                - LSTM: (h_T, c_T)
        """
        if isinstance(embedded_inputs, torch.Tensor):
            if embedded_inputs.dim() != 2:
                raise ValueError(
                    "embedded_inputs tensor must have shape (seq_len, embedding_dim)"
                )
            sequence = [embedded_inputs[t] for t in range(embedded_inputs.size(0))]
            device = embedded_inputs.device
        else:
            if len(embedded_inputs) == 0:
                raise ValueError("embedded_inputs cannot be empty")
            sequence = embedded_inputs
            device = sequence[0].device

        state = self.rnn.init_hidden(device=device)
        encoder_outputs = []

        for x_t in sequence:
            state = self.rnn.step(x_t, state)

            if isinstance(state, tuple):
                h_t, _ = state
                encoder_outputs.append(h_t)
            else:
                encoder_outputs.append(state)

        return encoder_outputs, state