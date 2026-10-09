"""MSHCL geometry and independently configurable cross-modal loss scales.

The MSHCL reference normalizes features, calls PoincareBall.expmap0, and uses
negative (unsquared) geodesic distance. Its emotion auxiliary term is CE;
our optional emotion term is cross-modal supervised contrastive learning.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveGeometry(nn.Module):
    def __init__(self, geometry="hyperbolic", curvature=0.1):
        super().__init__()
        if geometry not in ("euclidean", "hyperbolic"):
            raise ValueError(f"Unknown contrastive geometry: {geometry}")
        if not math.isfinite(curvature) or curvature <= 0:
            raise ValueError("Curvature c must be finite and positive (curvature is -c)")
        self.geometry = geometry
        if geometry == "hyperbolic":
            import geoopt
            self.ball = geoopt.PoincareBall(c=curvature, learnable=False)

    @staticmethod
    def _normalize(features):
        return F.normalize(features.reshape(features.shape[0], -1).float(), dim=1)

    def project(self, features):
        # Disable AMP for geometry; expmap/dist near the boundary need FP32.
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = self._normalize(features)
            if self.geometry == "hyperbolic":
                return self.ball.expmap0(features)
            return features

    def forward(self, eeg, targets):
        if eeg.shape[0] != targets.shape[0]:
            raise ValueError("EEG and targets must have the same batch size")
        with torch.autocast(device_type=eeg.device.type, enabled=False):
            eeg, targets = self.project(eeg), self.project(targets)
            if eeg.shape[1] != targets.shape[1]:
                raise ValueError("EEG and target embedding dimensions must match")
            if self.geometry == "hyperbolic":
                return -self.ball.dist(eeg[:, None, :], targets[None, :, :])
            return eeg @ targets.T


def emotion_positive_mask(*, labels, **metadata):
    if labels is None:
        raise ValueError("Emotion scale requires training/evaluation emotion labels")
    labels = labels.reshape(-1)
    # Includes the corresponding target and every same-emotion target.
    return labels[:, None].eq(labels[None, :])


# Add future scales here; each builder receives labels and dataset metadata.
# Scale-specific masks are separate from alignment false-negative masking.
POSITIVE_MASK_BUILDERS = {"emotion": emotion_positive_mask}


def multi_positive_loss(logits, positive_mask, symmetric=True):
    """Average log probability over positives, then over valid anchors."""
    def directional(scores, positives):
        positives = positives.to(device=scores.device, dtype=torch.bool)
        count = positives.sum(dim=1)
        valid = count > 0
        if not valid.any():
            return scores.sum() * 0.0
        log_prob = F.log_softmax(scores[valid], dim=1)
        return -(log_prob * positives[valid]).sum(dim=1).div(count[valid]).mean()

    loss = directional(logits, positive_mask)
    if symmetric:
        loss = (loss + directional(logits.T, positive_mask.T)) / 2
    return loss


class MultiScaleContrastiveLoss(nn.Module):
    """Composable auxiliary losses for EEG versus fused semantic targets."""
    def __init__(self, geometry, scales):
        super().__init__()
        self.geometry = geometry
        self.scales = {}
        for name, options in scales.items():
            enabled = bool(options.get("enabled", False))
            weight = float(options.get("weight", 0.1))
            temperature = float(options.get("temperature", 0.07))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name}: weight must be finite and nonnegative")
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError(f"{name}: temperature must be finite and positive")
            if enabled and name not in POSITIVE_MASK_BUILDERS:
                raise ValueError(f"No positive-mask builder registered for scale {name}")
            self.scales[name] = dict(enabled=enabled, weight=weight,
                                     temperature=temperature,
                                     symmetric=bool(options.get("symmetric", True)))

    def forward(self, eeg, targets, **metadata):
        total = eeg.sum() * 0.0
        components = {}
        scores = None
        for name, options in self.scales.items():
            if not options["enabled"]:
                components[name] = total.detach() * 0.0
                continue
            if scores is None:
                scores = self.geometry(eeg, targets)
            mask = POSITIVE_MASK_BUILDERS[name](**metadata)
            if mask.shape != scores.shape:
                raise ValueError(f"{name}: mask shape does not match contrastive scores")
            term = multi_positive_loss(scores / options["temperature"], mask,
                                       symmetric=options["symmetric"])
            components[name] = term
            total = total + options["weight"] * term
        return total, components


class ContrastiveExperimentMixin:
    """Attach geometry and optional scales without changing AMA fusion rules."""
    def _init_contrastive_experiment(self, cfg):
        options = cfg.get("contrastive", {})
        geometry = options.get("geometry", "euclidean")
        hyperbolic = options.get("hyperbolic", {})
        self.contrastive_geometry = ContrastiveGeometry(
            geometry, float(hyperbolic.get("curvature", 0.1)))
        self.scale_criterion = MultiScaleContrastiveLoss(
            self.contrastive_geometry, options.get("scales", {}))
        criterion = self.distill_criterion if self.pretrain_mode in (2, 3) else self.criterion
        if geometry == "hyperbolic":
            criterion.geometry = self.contrastive_geometry
        criterion.symmetric = bool(options.get("symmetric", True))
        print(f"[HyperAMA] geometry={geometry}, c={hyperbolic.get('curvature', 0.1)}, "
              f"temperature={cfg.loss_temp}, scales={self.scale_criterion.scales}")

    def _scale_losses(self, eeg, targets, phase, **metadata):
        total, components = self.scale_criterion(eeg, targets, **metadata)
        for name, term in components.items():
            self.log(f"ext/{phase}/loss_{name}", term,
                     on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"ext/{phase}/loss_scales_weighted", total,
                 on_step=False, on_epoch=True, prog_bar=False)
        return total
