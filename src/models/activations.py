import torch

def sigmoid(x):
    """
    Compute the sigmoid of x.
    """
    return 1 / (1 + torch.exp(-x))

def relu(x):
    """
    Compute the ReLU of x.
    """
    return torch.maximum(torch.tensor(0.0, device=x.device), x)

def tanh(x):
    """
    Compute the hyperbolic tangent of x.
    """
    exp_pos = torch.exp(x)
    exp_neg = torch.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)

def softmax(x):
    """
    Compute the softmax of x.
    """
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return e_x / torch.sum(e_x, dim=-1, keepdim=True)

__all__ = [
    'sigmoid',
    'relu',
    'tanh',
    'softmax'
]
