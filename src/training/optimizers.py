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
                # 1. Ensure m, v are on the same device as param (only need to check once or initialize correctly)
                if self.m[i].device != param.device:
                    self.m[i] = self.m[i].to(param.device)
                    self.v[i] = self.v[i].to(param.device)

                # 2. Update m and v (use in-place to boost speed and save memory)
                # m = beta1 * m + (1 - beta1) * g
                self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                # v = beta2 * v + (1 - beta2) * (g * g)
                self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

                # 3. Calculate bias correction
                bias_correction1 = 1 - self.beta1 ** self.t
                bias_correction2 = 1 - self.beta2 ** self.t

                # 4. Update parameters
                step_size = self.lr / bias_correction1
                denom = (self.v[i].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                
                param.addcdiv_(self.m[i], denom, value=-step_size)

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
    AdamW optimizer not using torch.optim.AdamW for efficiency.
    Supports weight decay grouping.
    """
    def __init__(self, named_parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        # Lưu lại tên và tham số để lọc weight decay
        self.param_list = []
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # Phân loại tham số ngay từ đầu
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"}
        
        self.params = []
        self.should_decay = []

        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            self.params.append(param)
            # Nếu tên không nằm trong danh sách no_decay thì sẽ áp dụng decay
            self.should_decay.append(not any(nd in name for nd in no_decay))

        # Khởi tạo m và v
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def named_parameters(self):
        for i, param in enumerate(self.params):
            yield f"param_{i}", param

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

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