import torch


class SGD:
    """
    Custom Stochastic Gradient Descent implementation.
    Does not use torch.optim.SGD.
    """

    def __init__(self, parameters, lr=0.01):
        self.params = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param -= self.lr * param.grad

    def state_dict(self):
        return {
            "type": "sgd",
            "lr": self.lr
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get("lr", self.lr)


class Adam:
    """
    Custom Adam optimizer implementation.
    Does not use torch.optim.Adam.
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        self.t += 1

        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                g = param.grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {
            "type": "adam",
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "t": self.t,
            "m": [tensor.clone() for tensor in self.m],
            "v": [tensor.clone() for tensor in self.v],
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get("lr", self.lr)
        self.beta1 = state_dict.get("beta1", self.beta1)
        self.beta2 = state_dict.get("beta2", self.beta2)
        self.eps = state_dict.get("eps", self.eps)
        self.t = state_dict.get("t", self.t)

        loaded_m = state_dict.get("m", None)
        loaded_v = state_dict.get("v", None)

        if loaded_m is not None and len(loaded_m) == len(self.params):
            self.m = [
                loaded_m[i].to(self.params[i].device)
                for i in range(len(self.params))
            ]

        if loaded_v is not None and len(loaded_v) == len(self.params):
            self.v = [
                loaded_v[i].to(self.params[i].device)
                for i in range(len(self.params))
            ]


class AdamW:
    """
    AdamW optimizer using torch.optim.AdamW for efficiency.
    Supports weight decay grouping.
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.params = list(parameters)
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"}
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ], lr=lr, betas=(beta1, beta2), eps=eps)

    def named_parameters(self):
        for i, param in enumerate(self.params):
            yield f"param_{i}", param

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def get_optimizer(model, lr=0.01, opt_type='adam'):
    """Create custom optimizer for model."""
    if opt_type.lower() == 'adam':
        return Adam(model.parameters(), lr=lr)
    elif opt_type.lower() == 'adamw':
        return AdamW(model.parameters(), lr=lr)
    elif opt_type.lower() == 'sgd':
        return SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(
            f"Optimizer '{opt_type}' chưa được hỗ trợ. Chọn 'adam', 'adamw', hoặc 'sgd'."
        )