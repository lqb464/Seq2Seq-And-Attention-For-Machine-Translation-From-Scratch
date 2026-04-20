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
        # encoder_outputs hiện là list các tensor (hidden_size, 1)
        # Convert it to a single tensor: (src_len, hidden_size)
        enc_states = torch.stack(encoder_outputs).squeeze(-1) 
        
        # Vectorize the computation: (src_len, hidden_size)
        # W_h @ decoder_hidden -> (hidden_size, 1)
        # W_e @ enc_states.T -> (hidden_size, src_len)
        energy = tanh(self.W_h @ decoder_hidden + self.W_e @ enc_states.T + self.b)
        
        # scores: (1, src_len)
        scores = self.v @ energy
        attn_weights = softmax(scores.squeeze(0)) # Softmax trên src_len

        # context: (hidden_size, 1)
        context = (enc_states.T @ attn_weights.unsqueeze(1))
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
        enc_states = torch.stack(encoder_outputs).squeeze(-1) # (src_len, hidden_size)
        
        # score = h_t^T @ W @ e_i
        # Compute all scores simultaneously: (1, hidden_size) @ (hidden_size, hidden_size) @ (hidden_size, src_len)
        scores = decoder_hidden.T @ self.W @ enc_states.T
        
        attn_weights = softmax(scores.squeeze(0))
        context = (enc_states.T @ attn_weights.unsqueeze(1))
        return context, attn_weights