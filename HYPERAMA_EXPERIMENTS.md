# HyperAMA-EEG initial ablations

The server project is `/home/xiongjiazhuang/work/HyperAMA-EEG` and the independent
Conda environment is `hyperama-eeg`. The AMA-EEG source and required semantic
features/alignment weights were copied; old checkpoints and experiment outputs
were excluded. `datasets/FACED` links to the same existing EEG dataset used by
AMA-EEG, without duplicating it.

## Two experiments

Both use FACED nine-class recognition, the original ten subject folds, seed 7,
dynamic fused semantic targets, original EEG/modal projectors and original
alignment false-negative masks. No new learned projection head is added.

| Experiment | Run | Physical GPU | Alignment | Emotion scale |
| --- | --- | --- | --- | --- |
| `hyper_distance` | 201 | 0 | Hyperbolic distance | Disabled |
| `hyper_emotion` | 202 | 1 | Hyperbolic distance | Enabled, weight 0.1 |

Geometry follows the installed MSHCL implementation in `main_pretrain.py` and
`SimCLR.info_nce_loss`: `geoopt==0.5.1`, `PoincareBall(c=0.1, learnable=False)`,
`F.normalize(..., dim=1)`, `expmap0`, and `-ball.dist(...) / 0.07`.
Distance is not squared. Both branches use one common manifold and curvature.
Geometry is computed in FP32 with AMP disabled; the initial experiments use
`32-true` precision and `highest` float32 matmul precision. The normalizing
projection is the MSHCL reference choice; it places points at a common radius
and does not by itself learn radial hierarchies.

AMA-EEG's optimizer, feature dimensions, fusion alpha/probes, data sampling and
downstream processing are retained. The MSHCL backbone, batch size, weight decay
and EEG-only pairing are not substituted, so the two experiments isolate the
additional emotion loss. MSHCL uses an emotion classifier CE auxiliary loss;
the new emotion term here is explicitly a cross-modal supervised contrastive
loss rather than a claim of reproducing that CE term.

The base loss still has corresponding EEG-target pairs as positives and its
original dataset-specific false-negative masking. The optional emotion loss
compares EEG to all fused semantic targets in the batch, treats all matching
emotion labels (including corresponding pairs) as positives, and other emotion
labels as negatives. It averages negative log probability over positive targets
and then over anchors, in both EEG-to-target and target-to-EEG directions.
The alignment mask is not applied to the emotion term. With dynamic FACED
fusion the training objective is:

`L_total = w_clip * (L_alignment + probe_loss_weight * L_probes + emotion.weight * L_emotion)`

When disabled the emotion contribution is exactly zero. Each scale has its own
`enabled`, `weight`, `temperature`, and `symmetric` configuration. Future
stimulus multi-positive scales can be added to `POSITIVE_MASK_BUILDERS` in
`model/loss/hyperbolic.py` and configured under the same `scales` mapping.
Unknown enabled scales fail explicitly instead of silently doing nothing.

Validation retrieval and checkpoint selection use the same geometry as the
alignment loss. Backbone prototype diagnostics and downstream MLP remain
unchanged. The original dynamic fusion uses validation labels to calculate
alpha, and checkpoints are selected on held-out-fold validation retrieval;
this inherited protocol is retained for the initial controlled comparison.
Those validation diagnostics are not an independent zero-shot test result.
Final classification results must be compared using the downstream MLP fold
metrics, with the inherited selection protocol stated explicitly.

## Configuration and launch

Pretraining `max_epochs` is 80 in the base configuration and every dataset
protocol. The initial runs retain minimum 5 epochs and patience 5 for FACED;
80 is an upper bound rather than a requirement to run all 80 epochs.

```bash
conda activate hyperama-eeg
cd /home/xiongjiazhuang/work/HyperAMA-EEG
python train_ext.py data=FACED +experiment=hyper_distance
python train_ext.py data=FACED +experiment=hyper_emotion
```

Individual options can be overridden independently:

```bash
python train_ext.py data=FACED +experiment=hyper_emotion \
  train.contrastive.scales.emotion.enabled=true \
  train.contrastive.scales.emotion.weight=0.1 \
  train.contrastive.scales.emotion.temperature=0.07 \
  train.contrastive.hyperbolic.curvature=0.1
```

`train.contrastive.geometry=euclidean` restores cosine alignment and allows a
future Euclidean-versus-hyperbolic control under the same training budget.

The two tmux windows run `scripts/launch_ablation.sh hyper_distance 0` and
`scripts/launch_ablation.sh hyper_emotion 1`. Each invokes an isolated pipeline:
ten-fold pretraining, feature extraction with shape/finite-value validation,
and ten-fold MLP evaluation. `CUDA_VISIBLE_DEVICES` maps the chosen physical
GPU to logical device 0, and all stages use the independent environment.
Existing server tasks are not stopped; GPU 1 also has an existing MSHCL job.

```bash
tmux attach -t hyperama-ablation
tmux select-window -t hyperama-ablation:distance-gpu0
tmux select-window -t hyperama-ablation:emotion-gpu1
tail -f runs/hyper_distance/pretrain.log
tail -f runs/hyper_emotion/pretrain.log
```

Each experiment has independent checkpoints, features and logs. Its `runs`
directory contains `resolved_config.yaml`, `source_sha256.json`, `commands.json`,
`status.json`, stage logs, `pretrain_metrics.jsonl`, `verified_features.json`,
and finally `results.json` containing all ten held-out fold metrics.
The pipeline fails if extraction or evaluation omits any fold.
Launching an existing run again is rejected to avoid overwriting its results.

## Verification

```bash
OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
```

The added tests cover analytic mapping/distance, both dataset masks, an
independent multi-positive loss oracle, disabled-scale zero gradients,
configuration parity, finite model gradients and checkpoint round trips.
`runs/smoke_results.json` and `runs/smoke_hyper_*/` contain the two real-data
GPU smoke checks; those one-epoch partial checks are distinct from formal runs.
