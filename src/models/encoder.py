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
        # embedded_inputs: (seq_len, embedding_dim)
        # Thay vì loop, ta dùng hàm forward của rnn_model đã viết sẵn
        # (Hàm forward này xử lý cả chuỗi nhanh hơn nhiều)
        encoder_outputs, final_state = self.rnn.forward(embedded_inputs)
        return encoder_outputs, final_state