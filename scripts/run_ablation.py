"""Run isolated pretraining, feature extraction and MLP evaluation on one GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_features(cfg):
    import numpy as np
    directory = ROOT / cfg.ext_fea.output_dir
    labels = np.load(directory / "onesub_label2.npy", mmap_mode="r")
    expected_rows = cfg.data.n_subs * len(labels)
    rows = []
    for fold in range(10):
        matches = list(directory.glob(f"*_f{fold}_fea_{cfg.ext_fea.mode}.npy"))
        if len(matches) != 1:
            raise RuntimeError(f"Fold {fold}: expected one feature file, found {len(matches)}")
        data = np.load(matches[0], mmap_mode="r")
        if data.shape != (expected_rows, 1024) or not np.isfinite(data).all():
            raise RuntimeError(f"Fold {fold}: invalid extracted features {data.shape}")
        rows.append(dict(fold=fold, path=str(matches[0]), shape=list(data.shape), finite=True))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("hyper_distance", "hyper_emotion"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    run_root = ROOT / "runs" / args.experiment
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    if status_path.exists():
        raise RuntimeError(f"Run already has status at {status_path}; choose a fresh run or archive it first")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    overrides = ["data=FACED", f"+experiment={args.experiment}", "seed=7",
                 "train.pretrain_mode=2", "train.max_epochs=80", "train.num_workers=4",
                 "train.gpus=[0]", "start_fold=0", "end_fold=10",
                 f"train.metrics_path={run_root / 'pretrain_metrics.jsonl'}"]
    with initialize_config_dir(config_dir=str(ROOT / "cfgs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.save(cfg, run_root / "resolved_config.yaml", resolve=True)
    hashes = {}
    for folder in ("model", "data", "cfgs", "scripts", "utils"):
        for source in sorted((ROOT / folder).rglob("*")):
            if source.is_file() and source.suffix in (".py", ".yaml", ".sh"):
                hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ("train_ext.py", "extract_features.py", "train_mlp.py", "requirements.txt"):
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    save(run_root / "source_sha256.json", hashes)
    commands = {stage: [sys.executable, "-u", script, *overrides]
                for stage, script in (("pretrain", "train_ext.py"),
                                      ("extract", "extract_features.py"),
                                      ("mlp", "train_mlp.py"))}
    save(run_root / "commands.json", commands)
    status = dict(experiment=args.experiment, physical_gpu=args.gpu, pid=os.getpid(),
                  started_utc=now(), state="running", stage="initializing", stages={})
    save(status_path, status)
    try:
        for stage, command in commands.items():
            status["stage"] = stage
            status["stages"][stage] = dict(started_utc=now(), state="running")
            save(status_path, status)
            print(f"[{now()}] {args.experiment}: starting {stage} on physical GPU {args.gpu}", flush=True)
            with (run_root / f"{stage}.log").open("w", encoding="utf-8") as output:
                result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                        stdout=output, stderr=subprocess.STDOUT)
            status["stages"][stage].update(finished_utc=now(), returncode=result.returncode,
                                           state="complete" if result.returncode == 0 else "failed")
            save(status_path, status)
            if result.returncode:
                raise RuntimeError(f"{stage} exited with {result.returncode}; see {run_root / (stage + '.log')}")
            if stage == "extract":
                save(run_root / "verified_features.json", verify_features(cfg))
            if stage == "mlp":
                results = json.loads((run_root / "results.json").read_text(encoding="utf-8"))
                if len(results["folds"]) != 10:
                    raise RuntimeError("MLP evaluation did not produce all ten held-out folds")
        status.update(state="complete", finished_utc=now())
    except BaseException as error:
        status.update(state="failed", finished_utc=now(), error=str(error))
        save(status_path, status)
        raise
    save(status_path, status)
    print(f"[{now()}] Complete: {run_root / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
