"""Persistent scalar pretraining metrics for comparing loss ablations."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback


class ExperimentMetrics(Callback):
    def __init__(self, path, fold):
        self.path = Path(path)
        self.fold = int(fold)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                scalar = float(value.detach().cpu())
                metrics[name] = scalar if math.isfinite(scalar) else None
        row = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                   fold=self.fold, epoch=int(trainer.current_epoch),
                   global_step=int(trainer.global_step), metrics=metrics)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, allow_nan=False) + "\n")
