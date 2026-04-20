import torch


class CrossEntropyLoss:
    """
    Custom Cross Entropy Loss implementation from mathematical formula.
    Does not use torch.nn.CrossEntropyLoss.
    """

    def __init__(self, ignore_index=-100):
        self.ignore_index = ignore_index

    def __call__(self, logits, targets):
        """
        Args:
            logits: tensor shape (batch_size, num_classes)
            targets: tensor shape (batch_size,)
        """
        batch_size = logits.shape[0]

        max_val = logits.max(dim=1, keepdim=True)[0]
        shifted = logits - max_val
        log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1))
        log_softmax = logits - max_val - log_sum_exp.unsqueeze(1)

        mask = (targets != self.ignore_index)
        valid_count = mask.sum().item()

        if valid_count == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        loss = 0.0
        # Lấy ra log_softmax của các target tương ứng
        # targets shape: (batch_size,), log_softmax shape: (batch_size, num_classes)
        target_log_probs = log_softmax[torch.arange(batch_size), targets]
        loss = -target_log_probs[mask].sum() / valid_count

        loss = loss / valid_count
        return loss


def get_loss_function(pad_idx):
    """Return custom Cross Entropy loss function, ignoring PAD index."""
    return CrossEntropyLoss(ignore_index=pad_idx)