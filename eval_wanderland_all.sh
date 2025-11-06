#!/usr/bin/env bash
set -euo pipefail

# ground-truth root (from sqf or unpacked)
ROOT="/wanderland_eval"
PRED_ROOT="/scratch/rc5832/vggt_results"
PY="/ext3/miniconda3/envs/vggt/bin/python"
EVAL_SCRIPT="/scratch/rc5832/vggt/code/benchmark/reconstruction/eval_colmap_poses_safe.py"
MATCH_MODE="exact"

LOGDIR="${PRED_ROOT}/_eval_logs"
mkdir -p "$LOGDIR"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

: > "${LOGDIR}/failed_eval.txt"

find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r scene_dir; do
  base="$(basename "$scene_dir")"

  # new predicted directory
  pred_dir="${PRED_ROOT}/${base}"

  # ground truth stays the same
  gt_dir="$scene_dir/sparse/0"

  # output evaluation text goes into prediction root
  out_txt="${pred_dir}/${base}.txt"
  log="${LOGDIR}/${base}.log"

  # check existence
  if [[ ! -s "$pred_dir/images.bin" ]]; then
    echo "[SKIP] $base (missing ${pred_dir}/images.bin)"
    continue
  fi
  if [[ ! -s "$gt_dir/images.bin" ]]; then
    echo "[SKIP] $base (missing ${gt_dir}/images.bin)"
    continue
  fi

  echo "[EVAL] $base → logging to $log"
  if "$PY" "$EVAL_SCRIPT" "$pred_dir" "$gt_dir" --match "$MATCH_MODE" --output "$out_txt" \
      2>&1 | tee "$log"; then
    echo "[DONE] $base (result: $out_txt)"
  else
    echo "$scene_dir" >> "${LOGDIR}/failed_eval.txt"
    echo "[FAIL] $base (logged to $log)"
  fi
done
