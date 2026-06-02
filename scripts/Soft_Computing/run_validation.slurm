#!/bin/bash
# =============================================================================
# run_validation.slurm
# SC Validation — Lab-SB CIMAT (Partition: GPU or C2)
# Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
#
# This job runs validator.py on the SC results produced by run_inference.slurm.
# It does NOT require a GPU — the validator is CPU-only (sklearn + pandas).
# However, it is listed under GPU partition for simplicity of scheduling.
# If you have access to C2 nodes, change --partition=GPU to --partition=C2
# and remove the --gres line to save GPU resources.
#
# Usage:
#   # After run_inference.slurm completes:
#   sbatch run_validation.slurm
#
#   # Or chain it as a dependency:
#   sbatch --dependency=afterok:<INFERENCE_JOB_ID> run_validation.slurm
#
# TODO before submitting:
#   [1] Set SC_JOB_ID to the SLURM_JOB_ID of the completed inference run.
#       This determines which sc_results_<JID>.json file to validate.
#   [2] Set PROJECT_DIR and CONDA_ENV (same as run_inference.slurm).
# =============================================================================

#SBATCH --job-name=SC_validation
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/validation_%j.out
#SBATCH --error=logs/validation_%j.err

# =============================================================================
# Environment
# =============================================================================

# TODO [2]: match these to run_inference.slurm
PROJECT_DIR="/home/${USER}/stellar_classifier"
CONDA_ENV="astro"

# TODO [1]: set the SLURM_JOB_ID of the completed inference run
SC_JOB_ID="REPLACE_WITH_INFERENCE_JOB_ID"

# Derived paths
RESULTS="${PROJECT_DIR}/outputs/sc/sc_results_${SC_JOB_ID}.json"
CORPUS="${PROJECT_DIR}/data/stellar_corpus.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/validation"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

# =============================================================================
# Runtime setup
# =============================================================================

echo "============================================================"
echo "SC Validation — SLURM Job ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Date      : $(date)"
echo "Results   : ${RESULTS}"
echo "Corpus    : ${CORPUS}"
echo "Output    : ${OUTPUT_DIR}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Verify inputs
if [ ! -f "${RESULTS}" ]; then
    echo "ERROR: SC results not found at ${RESULTS}"
    echo "Check that SC_JOB_ID=${SC_JOB_ID} is correct."
    exit 1
fi

if [ ! -f "${CORPUS}" ]; then
    echo "ERROR: Corpus not found at ${CORPUS}"
    exit 1
fi

# =============================================================================
# Validation
# =============================================================================

echo ""
echo "Starting validation — $(date)"

python3 "${SCRIPTS_DIR}/validator.py" \
    --results "${RESULTS}"            \
    --corpus  "${CORPUS}"             \
    --output  "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "Validation finished — $(date) — exit code: ${EXIT_CODE}"
echo "Outputs written to: ${OUTPUT_DIR}"
echo "============================================================"

exit ${EXIT_CODE}
