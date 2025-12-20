#!/bin/bash
#SBATCH --job-name=must3r-seq-array
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yz10442@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --account=pr_133_general
#SBATCH --partition=rtx8000
#SBATCH --qos=gpu48
#SBATCH --array=1-201
#SBATCH --output=logs/must3r_seq_%A_%a.out
#SBATCH --error=logs/must3r_seq_%A_%a.err
#SBATCH --requeue

# =====================================================
# 环境设置
# =====================================================
module purge

OVERLAY_ENV=/scratch/yz10442/must3r/overlay-15GB-500K.ext3
OVERLAY_DATA=/vast/xl3136/DATA/wanderland_eval.sqf
SIF_PATH=/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
WORKDIR=/scratch/yz10442/must3r
SCRIPT=${WORKDIR}/slam.py
CHKPT=/scratch/yz10442/must3r/checkpoint/MUSt3R_512.pth

# 从列表文件中读取第 N 行对应的数据集路径
DATASET_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${WORKDIR}/dataset_list.txt)
echo ">>> [Task ${SLURM_ARRAY_TASK_ID}] Processing dataset: ${DATASET_NAME}"

# 输出路径
OUTDIR=/scratch/yz10442/must3r/output/${DATASET_NAME}
mkdir -p ${OUTDIR}

# =====================================================
# 在 Singularity 容器中运行 MUSt3R SLAM
# =====================================================
singularity exec --nv \
  --overlay ${OVERLAY_ENV}:ro \
  --overlay ${OVERLAY_DATA}:ro \
  ${SIF_PATH} /bin/bash -c "
    source /ext3/env.sh
    conda activate must3r

    export HF_HOME=/scratch/yz10442/huggingface
    export TRANSFORMERS_CACHE=/scratch/yz10442/huggingface
    export TORCH_HOME=/scratch/yz10442/torch_cache
    export MPLCONFIGDIR=/scratch/yz10442/matplotlib
    mkdir -p \$HF_HOME \$TRANSFORMERS_CACHE \$TORCH_HOME \$MPLCONFIGDIR

    cd ${WORKDIR}
    echo '>>> Launching MUSt3R SLAM on dataset ${DATASET_NAME} ...'

    python ${SCRIPT} \
      --chkpt \"${CHKPT}\" \
      --res 512 \
      --subsamp 2 \
      --keyframe_overlap_thr 0.05 \
      --min_conf_keyframe 1.5 \
      --overlap_percentile 85 \
      --input \"/wanderland_eval/${DATASET_NAME}/images\" \
      --output \"${OUTDIR}\" \
      --device cuda:0
  "

echo ">>> Finished dataset ${DATASET_NAME}"
