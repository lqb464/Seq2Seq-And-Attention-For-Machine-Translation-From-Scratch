import torch
import torch.nn as nn
from src.models.activations import softmax, tanh


class BahdanauAttention(nn.Module):
    """
    Additive attention from scratch:
        e_ti = v^T tanh(W_h h_t + W_e e_i + b)
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_e = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_size, 1))
        self.v = nn.Parameter(torch.randn(1, hidden_size) * 0.01)

    def score(self, decoder_hidden, encoder_output):
        energy = tanh(
            self.W_h @ decoder_hidden +
            self.W_e @ encoder_output +
            self.b
        )
        return self.v @ energy

    def forward(self, decoder_hidden, encoder_outputs):
        if len(encoder_outputs) == 0:
            raise ValueError("encoder_outputs cannot be empty")

        device = decoder_hidden.device
        scores = []

        for enc_out in encoder_outputs:
            enc_out = enc_out.to(device)
            s = self.score(decoder_hidden, enc_out)
            scores.append(s.squeeze())

        scores = torch.stack(scores)
        attn_weights = softmax(scores)

        context = torch.zeros((self.hidden_size, 1), device=device)
        for i, enc_out in enumerate(encoder_outputs):
            context = context + attn_weights[i] * enc_out.to(device)

        return context, attn_weights


class LuongAttention(nn.Module):
    """
    Luong attention from scratch (general form):
        score(h_t, e_i) = h_t^T W e_i
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

    def score(self, decoder_hidden, encoder_output):
        return decoder_hidden.T @ self.W @ encoder_output

    def forward(self, decoder_hidden, encoder_outputs):
        if len(encoder_outputs) == 0:
            raise ValueError("encoder_outputs cannot be empty")

        device = decoder_hidden.device
        scores = []

        for enc_out in encoder_outputs:
            enc_out = enc_out.to(device)
            s = self.score(decoder_hidden, enc_out)
            scores.append(s.squeeze())

        scores = torch.stack(scores)
        attn_weights = softmax(scores)

        context = torch.zeros((self.hidden_size, 1), device=device)
        for i, enc_out in enumerate(encoder_outputs):
            context = context + attn_weights[i] * enc_out.to(device)

        return context, attn_weights