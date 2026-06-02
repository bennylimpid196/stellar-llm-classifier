#!/bin/bash
#SBATCH --job-name=STELLAR_val_v6
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/val_v6_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/val_v6_%j.err

PROJECT_DIR="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing"
CONDA_ENV="prometheus"

# ── SC_JOB_ID: set this to the SLURM job ID of the inference run ─────────────
# After launching run_inference_v6.slurm, check the job ID with:
#   squeue -u $USER
# Then set SC_JOB_ID here and launch this script:
#   sbatch run_validation_v6.slurm
SC_JOB_ID="184113"
# ─────────────────────────────────────────────────────────────────────────────

RESULTS="${PROJECT_DIR}/outputs/sc/sc_results_v6_${SC_JOB_ID}.json"
CORPUS="${PROJECT_DIR}/Data/stellar_corpus_v5.json"
REFERENCES="${PROJECT_DIR}/Data/reference_descriptions.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/validation_v6"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "STELLAR V6 Validation — Job ${SLURM_JOB_ID}"
echo "Results   : ${RESULTS}"
echo "Corpus    : ${CORPUS}"
echo "References: ${REFERENCES}"
echo "Output    : ${OUTPUT_DIR}"

if [ ! -f "${RESULTS}" ]; then
    echo "ERROR: Results file not found: ${RESULTS}"
    echo "Check that SC_JOB_ID is set correctly and inference completed successfully."
    exit 1
fi
if [ ! -f "${REFERENCES}" ]; then
    echo "ERROR: References file not found: ${REFERENCES}"
    exit 1
fi
if [ ! -f "${CORPUS}" ]; then
    echo "ERROR: Corpus not found: ${CORPUS}"
    exit 1
fi

# rouge-score must be installed in the conda env — install if missing
python3 -c "from rouge_score import rouge_scorer" 2>/dev/null || {
    echo "Installing rouge-score into conda env ${CONDA_ENV}..."
    pip install rouge-score
}

cd "${SCRIPTS_DIR}"

python3 validator_v6.py \
    --results    "${RESULTS}"    \
    --corpus     "${CORPUS}"     \
    --references "${REFERENCES}" \
    --output     "${OUTPUT_DIR}"

EXIT_CODE=$?
echo "Validation exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Validation complete."
    echo "Report  : ${OUTPUT_DIR}/validation_report_v6.json"
    echo "Summary : ${OUTPUT_DIR}/validation_summary_v6.txt"
    echo "A/B     : ${OUTPUT_DIR}/ab_boundary_analysis_v6.json"
    echo "ROUGE   : ${OUTPUT_DIR}/rouge_scores_v6.json"
    echo "Confusion: ${OUTPUT_DIR}/confusion_spectral_v6.csv"
else
    echo "ERROR: Validation failed with exit code ${EXIT_CODE}."
fi

exit ${EXIT_CODE}
