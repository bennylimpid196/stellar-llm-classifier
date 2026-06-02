#!/bin/bash
#SBATCH --job-name=STELLAR_inf_v6
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/sc_v6_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/sc_v6_%j.err

PROJECT_DIR="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing"
MODEL_PATH="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/AstroLlama/AstroSage-8B"
CONDA_ENV="prometheus"

CORPUS="${PROJECT_DIR}/Data/stellar_corpus_v5.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/sc"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "STELLAR V6 Inference — Job ${SLURM_JOB_ID}"
echo "Corpus    : ${CORPUS}"
echo "Model     : ${MODEL_PATH}"
echo "Output    : ${OUTPUT_DIR}"
echo "Scripts   : ${SCRIPTS_DIR}"

if [ ! -f "${CORPUS}" ]; then echo "ERROR: corpus not found: ${CORPUS}"; exit 1; fi
if [ ! -d "${MODEL_PATH}" ]; then echo "ERROR: model not found: ${MODEL_PATH}"; exit 1; fi

cd "${SCRIPTS_DIR}"

python3 inference_manager_v6.py \
    --corpus         "${CORPUS}"     \
    --model          "${MODEL_PATH}" \
    --output         "${OUTPUT_DIR}" \
    --batch-size     50              \
    --max-retries    3               \
    --max-new-tokens 1024            \
    --job-id         "${SLURM_JOB_ID}"

EXIT_CODE=$?
echo "Inference exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Inference completed successfully."
    echo "Results: ${OUTPUT_DIR}/sc_results_v6_${SLURM_JOB_ID}.json"
else
    echo "ERROR: Inference failed with exit code ${EXIT_CODE}."
fi

exit ${EXIT_CODE}
