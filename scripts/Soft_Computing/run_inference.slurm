#!/bin/bash
# =============================================================================
# run_inference.slurm
# SC Stellar Inference — Lab-SB CIMAT (Partition: GPU)
# Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
#
# Hardware per node (g-0-X):
#   GPU  : 2x NVIDIA Titan RTX 24 GB
#   RAM  : 128 GB
#   CPU  : Intel Xeon Silver 4214, 24 cores @ 2.2 GHz
#
# Usage:
#   sbatch run_inference.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f inference_manager_sc.log
#
# Cancel:
#   scancel <JOB_ID>
#
# TODO before submitting:
#   [1] Set --time to the actual walltime limit of the GPU partition.
#       Verify with: scontrol show partition GPU | grep MaxTime
#       MAGMA-01 ran under 72h; use that as reference if unconfirmed.
#   [2] Set MODEL_PATH to the absolute path of AstroSage-Llama on the cluster.
#   [3] Set PROJECT_DIR to your home directory path on the cluster.
#   [4] Verify that your Python environment name is correct (CONDA_ENV below).
# =============================================================================

#SBATCH --job-name=SC_stellar
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00          # TODO [1]: verify actual GPU partition walltime
#SBATCH --output=logs/sc_%j.out  # %j = SLURM_JOB_ID
#SBATCH --error=logs/sc_%j.err

# =============================================================================
# Environment — fill in your actual paths
# =============================================================================

# TODO [3]: set your project root on the cluster
PROJECT_DIR="/home/${USER}/stellar_classifier"

# TODO [2]: set the absolute path to AstroSage-Llama weights on the cluster
MODEL_PATH="/home/${USER}/models/AstroSage-Llama"

# TODO [4]: set your conda environment name (the one with transformers + torch)
CONDA_ENV="astro"

# Derived paths — do not modify
CORPUS="${PROJECT_DIR}/data/stellar_corpus.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/sc"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

# =============================================================================
# Runtime setup
# =============================================================================

echo "============================================================"
echo "SC Stellar Inference — SLURM Job ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Date    : $(date)"
echo "Project : ${PROJECT_DIR}"
echo "Model   : ${MODEL_PATH}"
echo "Corpus  : ${CORPUS}"
echo "Output  : ${OUTPUT_DIR}"
echo "============================================================"

# Create output and log directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Verify GPU visibility
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Verify model path exists before launching
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found at ${MODEL_PATH}"
    exit 1
fi

# Verify corpus exists
if [ ! -f "${CORPUS}" ]; then
    echo "ERROR: Corpus not found at ${CORPUS}"
    echo "Run corpus_builder.py first."
    exit 1
fi

# =============================================================================
# Inference
# =============================================================================

echo ""
echo "Starting inference — $(date)"

python3 "${SCRIPTS_DIR}/inference_manager.py" \
    --corpus         "${CORPUS}"              \
    --model          "${MODEL_PATH}"           \
    --output         "${OUTPUT_DIR}"           \
    --batch-size     50                        \
    --max-retries    3                         \
    --max-new-tokens 512                       \
    --job-id         "${SLURM_JOB_ID}"

EXIT_CODE=$?

echo ""
echo "Inference finished — $(date) — exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
