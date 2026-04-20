"""
Custom Learning Rate Scheduler implementation from scratch.
Does not use torch.optim.lr_scheduler.

Supports:
- StepLRScheduler: Decrease lr every N epochs by gamma factor.
- WarmupScheduler: Gradually increase lr in first N epochs (warm-up), then decrease.
"""


class StepLRScheduler:
    """
    Decrease learning rate every `step_size` epoch by gamma factor.
    
    Example:
        initial lr = 0.001, step_size=5, gamma=0.5
        → Epoch 1-5:  lr = 0.001
        → Epoch 6-10: lr = 0.0005
        → Epoch 11+:  lr = 0.00025
    """

    def __init__(self, optimizer, step_size=5, gamma=0.5):
        """
        Args:
            optimizer: Custom optimizer (SGD or Adam) — must have .lr attribute
            step_size: After how many epochs to decrease lr
            gamma: Factor to multiply lr by (0 < gamma < 1)
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self.current_epoch = 0

    def step(self, epoch=None):
        """
        Call after each epoch to update the learning rate.

        Args:
            epoch: Current epoch number (1-indexed). If None, increment the counter automatically.
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # Số lần đã giảm = epoch // step_size
        num_decays = self.current_epoch // self.step_size
        new_lr = self.base_lr * (self.gamma ** num_decays)
        self.optimizer.lr = new_lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.lr

    def state_dict(self):
        return {
            "type": "step_lr",
            "step_size": self.step_size,
            "gamma": self.gamma,
            "base_lr": self.base_lr,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict):
        self.step_size = state_dict.get("step_size", self.step_size)
        self.gamma = state_dict.get("gamma", self.gamma)
        self.base_lr = state_dict.get("base_lr", self.base_lr)
        self.current_epoch = state_dict.get("current_epoch", self.current_epoch)


class WarmupScheduler:
    """
    Warm-up: Increase lr linear in `warmup_steps` first epochs,
    after which decrease gradually according to the inverse square root.
    
    Formula:
        - During warm-up phase (epoch <= warmup_steps):
            lr = base_lr * (epoch / warmup_steps)
        - After warm-up (epoch > warmup_steps):
            lr = base_lr * (warmup_steps^0.5 / epoch^0.5)
    
    Example:
        base_lr = 0.001, warmup_steps = 5
        → Epoch 1: lr = 0.0002
        → Epoch 2: lr = 0.0004
        → Epoch 5: lr = 0.001  (đỉnh)
        → Epoch 10: lr ≈ 0.000707
        → Epoch 20: lr = 0.0005
    """

    def __init__(self, optimizer, warmup_steps=5):
        """
        Args:
            optimizer: Custom optimizer
            warmup_steps: Number of warm-up epochs (gradually increase lr)
        """
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.base_lr = optimizer.lr
        self.current_epoch = 0

    def step(self, epoch=None):
        """
        Call after each epoch to update the learning rate.
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        epoch = max(1, self.current_epoch)

        if epoch <= self.warmup_steps:
            # Warm-up phase: linear increase
            new_lr = self.base_lr * (epoch / self.warmup_steps)
        else:
            # After warm-up: decrease according to inverse square root
            new_lr = self.base_lr * (self.warmup_steps ** 0.5) / (epoch ** 0.5)

        self.optimizer.lr = new_lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.lr

    def state_dict(self):
        return {
            "type": "warmup",
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.base_lr = state_dict.get("base_lr", self.base_lr)
        self.current_epoch = state_dict.get("current_epoch", self.current_epoch)


def get_scheduler(optimizer, scheduler_type="step", **kwargs):
    """
    Function to create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer (SGD or Adam)
        scheduler_type: 'step' or 'warmup'
        **kwargs: Additional parameters for the scheduler
    
    Returns:
        Corresponding scheduler
    """
    if scheduler_type.lower() == "step":
        return StepLRScheduler(
            optimizer,
            step_size=kwargs.get("step_size", 5),
            gamma=kwargs.get("gamma", 0.5),
        )
    elif scheduler_type.lower() == "warmup":
        return WarmupScheduler(
            optimizer,
            warmup_steps=kwargs.get("warmup_steps", 5),
        )
    else:
        raise ValueError(
            f"Scheduler '{scheduler_type}' is not supported. Choose 'step' or 'warmup'."
        )
