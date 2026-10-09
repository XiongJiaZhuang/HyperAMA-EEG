"""Geometry, false-negative masks, multiscale supervision and checkpoint tests."""
import math
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F

from model import get_extractor_class
from model.loss.con_loss import MultiModalInfoNCELoss
from model.loss.seed_con_loss import MultiModalInfoNCELoss as SeedAlignment
from model.loss.hyperbolic import (
    ContrastiveGeometry, MultiScaleContrastiveLoss, multi_positive_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def config(dataset="FACED", experiment="hyper_distance"):
    with initialize_config_dir(config_dir=str(ROOT / "cfgs"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"data={dataset}",
                        f"+experiment={experiment}"])


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1024, 1024)

    def forward(self, x, **kwargs):
        return self.net(x)

    def set_saveFea(self, flag):
        pass


class HyperbolicTests(unittest.TestCase):
    def test_mshcl_mapping_against_closed_form(self):
        geometry = ContrastiveGeometry(curvature=0.1)
        features = torch.tensor([[1., 0., 0.], [0., 5., 0.], [-2., 0., 0.]])
        projected = geometry.project(features)
        radius = math.tanh(math.sqrt(0.1)) / math.sqrt(0.1)
        expected = F.normalize(features, dim=1) * radius
        torch.testing.assert_close(projected, expected)
        # Independent Poincare distance formula for orthogonal points.
        squared_distance = 2 * radius ** 2
        distance = math.acosh(1 + 2 * 0.1 * squared_distance /
                              (1 - 0.1 * radius ** 2) ** 2) / math.sqrt(0.1)
        self.assertAlmostEqual(geometry(features, features)[0, 1].item(), -distance, places=5)
        self.assertFalse(geometry.ball.k.requires_grad)

    def test_distance_only_preserves_faced_and_seed_masks(self):
        torch.manual_seed(7)
        eeg, targets = torch.randn(6, 8), torch.randn(6, 8)
        vids = torch.tensor([0, 1, 2, 0, 1, 2])
        subs = torch.tensor([0, 0, 0, 1, 1, 1])
        geometry = ContrastiveGeometry()
        scores = geometry(eeg, targets) / 0.07
        mask = vids[:, None].eq(vids[None, :]) & ~subs[:, None].eq(subs[None, :])
        expected_scores = scores.masked_fill(mask, -1e4)
        expected = (F.cross_entropy(expected_scores, torch.arange(6)) +
                    F.cross_entropy(expected_scores.T, torch.arange(6))) / 2
        criterion = MultiModalInfoNCELoss(temperature=0.07, mask_mode="vid_sub")
        criterion.geometry = geometry
        actual = criterion(eeg, targets, vid_ids=vids, sub_ids=subs)
        torch.testing.assert_close(actual, expected)
        criterion = SeedAlignment(temperature=0.07)
        criterion.geometry = geometry
        torch.testing.assert_close(criterion(eeg, targets, content_ids=vids), expected)

    def test_emotion_multi_positive_loss_has_independent_oracle(self):
        scores = torch.tensor([[3., 1., -2.], [0., 2., -1.], [-1., -2., 4.]])
        labels = [0, 0, 1]
        directional = []
        for matrix in (scores, scores.T):
            anchor_losses = []
            for i in range(3):
                denominator = sum(math.exp(float(matrix[i, j])) for j in range(3))
                positives = [j for j in range(3) if labels[i] == labels[j]]
                anchor_losses.append(sum(math.log(denominator) - float(matrix[i, j])
                                         for j in positives) / len(positives))
            directional.append(sum(anchor_losses) / 3)
        mask = torch.tensor(labels)[:, None].eq(torch.tensor(labels)[None, :])
        self.assertAlmostEqual(multi_positive_loss(scores, mask).item(),
                               sum(directional) / 2, places=6)

    def test_disabled_emotion_needs_no_labels_and_is_exactly_zero(self):
        loss = MultiScaleContrastiveLoss(ContrastiveGeometry(),
                                        {"emotion": {"enabled": False}})
        eeg = torch.randn(4, 8, requires_grad=True)
        value, _ = loss(eeg, torch.randn(4, 8))
        self.assertEqual(value.item(), 0)
        value.backward()
        self.assertTrue(torch.equal(eeg.grad, torch.zeros_like(eeg)))

    def test_configuration_ablation_changes_only_emotion_and_run_identity(self):
        cfg_a = config(experiment="hyper_distance")
        cfg_b = config(experiment="hyper_emotion")
        cfg_b.log = cfg_a.log
        a = OmegaConf.to_container(cfg_a, resolve=True)
        b = OmegaConf.to_container(cfg_b, resolve=True)
        self.assertEqual(a["train"]["max_epochs"], 80)
        self.assertEqual(a["train"]["contrastive"]["hyperbolic"]["curvature"], 0.1)
        self.assertEqual(a["train"]["loss_temp"], 0.07)
        a.pop("log"), b.pop("log")
        a["train"]["contrastive"]["scales"]["emotion"]["enabled"] = True
        self.assertEqual(a, b)
        for dataset in ("SEED", "FACED_def_c2"):
            self.assertEqual(config(dataset).train.max_epochs, 80)

    def test_train_validation_and_checkpoint_roundtrip(self):
        for dataset in ("FACED", "SEED"):
            for experiment in ("hyper_distance", "hyper_emotion"):
                with self.subTest(dataset=dataset, experiment=experiment):
                    cfg = config(dataset, experiment)
                    cfg.train.use_two_stage_projector = False
                    cfg.train.n_class = 3
                    model_class = get_extractor_class(dataset)
                    module = model_class(ToyBackbone(), cfg.train)
                    module.log = lambda *args, **kwargs: None
                    module.log_dict = lambda *args, **kwargs: None
                    module.optimizers = lambda: type("Optimizer", (),
                                        {"param_groups": [{"lr": 0.0007}]})()
                    torch.manual_seed(7)
                    eeg = torch.randn(6, 1024)
                    text, image = torch.randn_like(eeg), torch.randn_like(eeg)
                    labels = torch.tensor([0, 1, 2, 0, 1, 2])
                    vids = torch.tensor([0, 1, 2, 0, 1, 2])
                    subs = torch.tensor([0, 0, 0, 1, 1, 1])
                    batch = (eeg, labels, text, image, vids, subs)
                    loss = module.training_step(batch, 0)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for parameter in (module.model.net.weight, module.text_projector[0].weight,
                                      module.image_projector[0].weight):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                    module.eval()
                    with torch.no_grad():
                        self.assertTrue(torch.isfinite(module.validation_step(batch, 0)))
                    restored = model_class(ToyBackbone(), cfg.train)
                    restored.load_state_dict(module.state_dict(), strict=True)

    def test_bfloat16_input_geometry_is_float32_with_finite_gradients(self):
        eeg = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)
        target = torch.randn_like(eeg, requires_grad=True)
        scores = ContrastiveGeometry()(eeg, target)
        self.assertEqual(scores.dtype, torch.float32)
        scores.sum().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertTrue(torch.isfinite(target.grad).all())


if __name__ == "__main__":
    unittest.main()
