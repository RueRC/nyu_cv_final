#!/usr/bin/env bash
set -euo pipefail

########################################
# ========== USER CONFIG (EDIT) ========
########################################
# Read-only source of scenes (each scene has a subfolder "images/")
SCENE_ROOT="/media/huge/Game/test"

# Writable root for all final outputs (one folder per scene under here)
OUTPUT_ROOT="/media/huge/Game/vggt-long-results"

# Where your current vggt_long.py writes its stage-1 outputs (it may be hardcoded there)
# We'll copy the two txt files from here into OUTPUT_ROOT/<scene>/exp/
VGGT_EXP_ROOT="/media/huge/Game/vggt-long-results/exp"

# vggt_long config and Python
CONFIG="./configs/base_config.yaml"
PYTHON="/home/huge/anaconda3/envs/vggt-long/bin/python"

# Exporter & evaluator scripts
EXPORT_SCRIPT="./benchmark/export_to_colmap.py"
EVAL_SCRIPT="./benchmark/eval_colmap_poses_safe.py"

# GT sparse relative subdir (under the read-only scene folder)
GT_SUBDIR="sparse/0"

# Name of the COLMAP sparse output subfolder (under OUTPUT_ROOT/<scene>/)
OUT_COLMAP_SUBDIR="vggt-long-sparse"   # -> OUTPUT_ROOT/<scene>/vggt-long-sparse/{cameras.bin,images.bin,points3D.bin}

# Eval matching mode
MATCH="exact"

# Overwrite policy: 0 = skip if results exist, 1 = force re-run (recreate)
OVERWRITE=0

# ----------- Per-stage logs (fixed locations as requested) -----------
LOG_ROOT="/media/huge/Game/vggt-long-results/_logs"
LOG_STAGE1="${LOG_ROOT}/stage1_vggt_long"
LOG_STAGE2="${LOG_ROOT}/stage2_to_colmap"
LOG_STAGE3="${LOG_ROOT}/stage3_eval"
########################################
# ======== END OF USER CONFIG ==========
########################################

# Sanity checks
[[ -f "vggt_long.py" ]] || { echo "[ERROR] vggt_long.py not found in current dir"; exit 1; }
[[ -f "$EXPORT_SCRIPT" ]] || { echo "[ERROR] $EXPORT_SCRIPT not found"; exit 1; }
[[ -f "$EVAL_SCRIPT"   ]] || { echo "[ERROR] $EVAL_SCRIPT not found"; exit 1; }

# Prepare dirs
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$VGGT_EXP_ROOT"
mkdir -p "$LOG_STAGE1" "$LOG_STAGE2" "$LOG_STAGE3"

echo "[INFO] SCENE_ROOT=$SCENE_ROOT"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[INFO] VGGT_EXP_ROOT=$VGGT_EXP_ROOT"
echo "[INFO] CONFIG=$CONFIG"
echo "[INFO] EXPORT_SCRIPT=$EXPORT_SCRIPT"
echo "[INFO] EVAL_SCRIPT=$EVAL_SCRIPT"
echo "[INFO] GT_SUBDIR=$GT_SUBDIR"
echo "[INFO] OUT_COLMAP_SUBDIR=$OUT_COLMAP_SUBDIR"
echo "[INFO] MATCH=$MATCH"
echo "[INFO] OVERWRITE=$OVERWRITE (1=force, 0=skip-if-exists)"
echo "[INFO] LOGS:"
echo "       stage1 → $LOG_STAGE1"
echo "       stage2 → $LOG_STAGE2"
echo "       stage3 → $LOG_STAGE3"
echo

TS="$(date +%Y%m%d_%H%M%S)"

# Helper: normalize exporter layout if it accidentally created a "0" subdir
normalize_colmap_layout() {
  local pred_sparse="$1"   # path to OUTPUT_ROOT/<scene>/vggt-long-sparse
  if [[ -d "$pred_sparse/0" ]] && [[ -f "$pred_sparse/0/cameras.bin" ]]; then
    echo "[INFO]   Normalizing COLMAP layout: moving files out of '$pred_sparse/0'..."
    shopt -s nullglob
    for f in "$pred_sparse/0"/*; do
      mv -f "$f" "$pred_sparse/"
    done
    rmdir "$pred_sparse/0" || true
  fi
}

# Iterate all scenes by finding "images" folders under SCENE_ROOT
find "$SCENE_ROOT" -type d -name "images" -print0 | while IFS= read -r -d '' IMG_DIR; do
  SCENE_DIR="$(dirname "$IMG_DIR")"                # read-only scene dir
  SCENE_NAME="$(basename "$SCENE_DIR")"

  # Final per-scene output root (writable)
  SCENE_OUT="$OUTPUT_ROOT/$SCENE_NAME"
  EXP_DIR="$SCENE_OUT/exp"                         # Stage 1 final location
  PRED_SPARSE="$SCENE_OUT/$OUT_COLMAP_SUBDIR"      # Stage 2 final location (no /0)
  EVAL_TXT="$SCENE_OUT/vggt_long_pose_eval_results.txt"  # Stage 3 final location

  # Where vggt_long.py originally writes (we copy from here)
  SRC_STAGE1_DIR="$VGGT_EXP_ROOT/$SCENE_NAME"
  SRC_POSES_TXT="$SRC_STAGE1_DIR/camera_poses.txt"
  SRC_INTR_TXT="$SRC_STAGE1_DIR/intrinsic.txt"

  # The two files we ultimately want under EXP_DIR
  DST_POSES_TXT="$EXP_DIR/camera_poses.txt"
  DST_INTR_TXT="$EXP_DIR/intrinsic.txt"

  # Ground-truth (read-only)
  GT_DIR="$SCENE_DIR/$GT_SUBDIR"

  # Logs for this scene
  LOG1="$LOG_STAGE1/${SCENE_NAME}_${TS}.log"
  LOG2="$LOG_STAGE2/${SCENE_NAME}_${TS}.log"
  LOG3="$LOG_STAGE3/${SCENE_NAME}_${TS}.log"

  echo "========================================"
  echo "[SCENE] $SCENE_NAME"
  echo "[IMGS ] $IMG_DIR"
  echo "[OUT  ] $SCENE_OUT"
  echo "----------------------------------------"
  echo "[STAGE1 dst] $EXP_DIR"
  echo "[STAGE2 dst] $PRED_SPARSE"
  echo "[STAGE3 dst] $EVAL_TXT"
  echo "[GT     dir] $GT_DIR"
  echo "[LOG1   ] $LOG1"
  echo "[LOG2   ] $LOG2"
  echo "[LOG3   ] $LOG3"
  echo

  ###################################
  # Stage 1: run vggt_long.py (writes to VGGT_EXP_ROOT), then copy to OUTPUT_ROOT/<scene>/exp/
  ###################################
  if [[ "$OVERWRITE" == "1" ]] || [[ ! -f "$DST_POSES_TXT" ]] || [[ ! -f "$DST_INTR_TXT" ]]; then
    echo "[RUN 1/3] vggt_long.py  →  $SRC_STAGE1_DIR"
    mkdir -p "$SRC_STAGE1_DIR"
    if [[ "$OVERWRITE" == "1" ]]; then
      { echo "[CMD] $PYTHON vggt_long.py --image_dir \"$IMG_DIR\" --config \"$CONFIG\" --overwrite";
        "$PYTHON" vggt_long.py --image_dir "$IMG_DIR" --config "$CONFIG" --overwrite; } \
        |& tee "$LOG1"
    else
      { echo "[CMD] $PYTHON vggt_long.py --image_dir \"$IMG_DIR\" --config \"$CONFIG\"";
        "$PYTHON" vggt_long.py --image_dir "$IMG_DIR" --config "$CONFIG"; } \
        |& tee "$LOG1"
    fi

    # Copy the two txt files into the final OUTPUT_ROOT layout
    mkdir -p "$EXP_DIR"
    if [[ ! -f "$SRC_POSES_TXT" ]] || [[ ! -f "$SRC_INTR_TXT" ]]; then
      echo "[ERROR] Missing stage-1 outputs in $SRC_STAGE1_DIR (camera_poses.txt or intrinsic.txt)."
      exit 1
    fi
    cp -f "$SRC_POSES_TXT" "$DST_POSES_TXT"
    cp -f "$SRC_INTR_TXT"  "$DST_INTR_TXT"
    echo "[INFO] Copied stage-1 txts → $EXP_DIR"
  else
    echo "[SKIP 1/3] Found existing $DST_POSES_TXT and $DST_INTR_TXT"
  fi
  echo

  ###################################
  # Stage 2: export to COLMAP into OUTPUT_ROOT/<scene>/<OUT_COLMAP_SUBDIR> (no /0)
  ###################################
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$PRED_SPARSE"
  fi
  if [[ ! -f "$PRED_SPARSE/cameras.bin" ]] || [[ ! -f "$PRED_SPARSE/images.bin" ]]; then
    echo "[RUN 2/3] export_to_colmap.py  →  $PRED_SPARSE"
    mkdir -p "$PRED_SPARSE"
    { echo "[CMD] $PYTHON $EXPORT_SCRIPT --images \"$IMG_DIR\" --poses \"$DST_POSES_TXT\" --intr \"$DST_INTR_TXT\" --out \"$PRED_SPARSE\"";
      "$PYTHON" "$EXPORT_SCRIPT" \
        --images "$IMG_DIR" \
        --poses "$DST_POSES_TXT" \
        --intr  "$DST_INTR_TXT" \
        --out   "$PRED_SPARSE"; } \
        |& tee "$LOG2"
    # Normalize if exporter created a '0/' subdir
    normalize_colmap_layout "$PRED_SPARSE"
  else
    echo "[SKIP 2/3] COLMAP trio already present in $PRED_SPARSE"
  fi
  echo

  ###################################
  # Stage 3: evaluate (read-only GT under SCENE_ROOT; write result into OUTPUT_ROOT/<scene>/)
  ###################################
  if [[ ! -d "$GT_DIR" ]]; then
    echo "[SKIP 3/3] GT dir not found: $GT_DIR"
    echo
    continue
  fi
  if [[ ! -f "$PRED_SPARSE/cameras.bin" ]]; then
    echo "[SKIP 3/3] Missing predicted cameras.bin: $PRED_SPARSE/cameras.bin"
    echo
    continue
  fi

  if [[ "$OVERWRITE" == "1" ]] || [[ ! -f "$EVAL_TXT" ]]; then
    echo "[RUN 3/3] eval_colmap_poses_safe.py  →  $EVAL_TXT"
    mkdir -p "$SCENE_OUT"
    { echo "[CMD] $PYTHON $EVAL_SCRIPT \"$PRED_SPARSE\" \"$GT_DIR\" --match \"$MATCH\" --output \"$EVAL_TXT\"";
      "$PYTHON" "$EVAL_SCRIPT" "$PRED_SPARSE" "$GT_DIR" --match "$MATCH" --output "$EVAL_TXT"; } \
        |& tee "$LOG3"
  else
    echo "[SKIP 3/3] Eval result already exists: $EVAL_TXT"
  fi

  echo
  echo "[DONE] $SCENE_NAME"
  echo "----------------------------------------"
  echo
done

echo "============== All scenes processed. =============="
