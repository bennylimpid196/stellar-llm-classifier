#!/bin/bash
#SBATCH --job-name=STELLAR_val_v7
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/val_v7_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/val_v7_%j.err

PROJECT_DIR="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing"
CONDA_ENV="prometheus"

# ── SC_JOB_ID: set this to the SLURM job ID of the V7 inference run ──────────
# After launching run_inference_v7.slurm, check with:
#   squeue -u $USER
# Then set SC_JOB_ID here and launch:
#   sbatch SLURM/run_validation_v7.slurm
SC_JOB_ID="192067"
# ─────────────────────────────────────────────────────────────────────────────

RESULTS="${PROJECT_DIR}/outputs/sc_v7/sc_results_v7_${SC_JOB_ID}.json"
CORPUS="${PROJECT_DIR}/Data/stellar_corpus_v7.json"
REFERENCES="${PROJECT_DIR}/Data/reference_descriptions.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/validation_v7"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "STELLAR V7 Validation — Job ${SLURM_JOB_ID}"
echo "Results   : ${RESULTS}"
echo "Corpus    : ${CORPUS}"
echo "References: ${REFERENCES}"
echo "Output    : ${OUTPUT_DIR}"

if [ ! -f "${RESULTS}" ]; then
    echo "ERROR: Results file not found: ${RESULTS}"
    echo "Check that SC_JOB_ID=${SC_JOB_ID} is correct and inference completed."
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

# Install dependencies if missing
python3 -c "from rouge_score import rouge_scorer" 2>/dev/null || {
    echo "Installing rouge-score..."
    pip install rouge-score
}

# bert-score is optional — validator skips BERTScore gracefully if absent
python3 -c "import bert_score" 2>/dev/null || {
    echo "bert-score not found — installing for BERTScore (Level 4a-bis)..."
    pip install bert-score --break-system-packages
}

cd "${SCRIPTS_DIR}"

python3 validator_v7.py \
    --results    "${RESULTS}"    \
    --corpus     "${CORPUS}"     \
    --references "${REFERENCES}" \
    --output     "${OUTPUT_DIR}"

EXIT_CODE=$?
echo "Validation exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Validation complete."
    echo "Report    : ${OUTPUT_DIR}/validation_report_v7.json"
    echo "Summary   : ${OUTPUT_DIR}/validation_summary_v7.txt"
    echo "ROUGE     : ${OUTPUT_DIR}/rouge_scores_v7.json"
    echo "BERTScore : ${OUTPUT_DIR}/bertscore_v7.json"
    echo "RAG impact: ${OUTPUT_DIR}/rag_impact_v7.json"
    echo "Confusion : ${OUTPUT_DIR}/confusion_spectral_v7.csv"
else
    echo "ERROR: Validation failed with exit code ${EXIT_CODE}."
fi

exit ${EXIT_CODE}
