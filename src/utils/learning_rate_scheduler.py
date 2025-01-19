from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class TransformerLRScheduler(LambdaLR):
    """
    Learning rate scheduler for Transformer models based on the formula:
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        d_model (int): Dimensionality of the model.
        warmup_steps (int): Number of warmup steps. Default is 4000.
    """

    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int = 4000) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        # Prevent division by zero
        step = max(step, 1)
        scale = step ** -0.5
        warmup_scale = step * self.warmup_steps ** -1.5
        return self.d_model ** -0.5 * min(scale, warmup_scale)