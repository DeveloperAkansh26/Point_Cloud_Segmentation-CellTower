import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of a model's weights (parameters AND buffers).

    Maintains a shadow copy of the full state_dict, updated after every optimizer
    step:  shadow = d*shadow + (1-d)*current. Buffers matter here because the model
    uses BatchNorm — its running_mean/var must be averaged too; integer buffers
    (e.g. num_batches_tracked) are copied verbatim.

    A timm-style decay warmup, d = min(decay, (1+n)/(10+n)), lets the average track
    fast while there are few updates (this run has relatively few optimizer steps per
    epoch) and settle to `decay` as n grows.

    Use store()/copy_to()/restore() to temporarily swap the EMA weights into the live
    model for validation / checkpointing, then restore the raw training weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        # Detached clones on the model's own device; these are the EMA weights.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def store(self, model: nn.Module):
        """Back up the model's current (raw training) weights."""
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model: nn.Module):
        """Load the EMA weights into the model."""
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module):
        """Restore the weights saved by the last store()."""
        assert self._backup is not None, "restore() called without a matching store()"
        model.load_state_dict(self._backup, strict=True)
        self._backup = None
