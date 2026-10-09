#!/usr/bin/env bash
set -euo pipefail
cd /home/xiongjiazhuang/work/HyperAMA-EEG
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec /data/xiongjiazhuang/anaconda3/envs/hyperama-eeg/bin/python -u scripts/run_ablation.py --experiment "$1" --gpu "$2"
