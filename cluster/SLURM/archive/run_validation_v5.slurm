#!/bin/bash
#SBATCH --job-name=STELLAR_val_v5
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/val_v5_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/val_v5_%j.err

PROJECT_DIR="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing"
CONDA_ENV="prometheus"

# TODO: reemplazar con el JOB_ID de la inferencia v5
SC_JOB_ID="184062"

RESULTS="${PROJECT_DIR}/outputs/sc/sc_results_v5_${SC_JOB_ID}.json"
CORPUS="${PROJECT_DIR}/Data/stellar_corpus.json"
REFERENCES="${PROJECT_DIR}/Data/reference_descriptions.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/validation_v5"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "STELLAR V5 Validation — Job ${SLURM_JOB_ID}"
echo "Results   : ${RESULTS}"
echo "References: ${REFERENCES}"

if [ ! -f "${RESULTS}" ]; then echo "ERROR: ${RESULTS} not found"; exit 1; fi
if [ ! -f "${REFERENCES}" ]; then echo "ERROR: ${REFERENCES} not found"; exit 1; fi

# Instalar rouge-score si no está disponible
pip install rouge-score --quiet --break-system-packages 2>/dev/null || true

python3 "${SCRIPTS_DIR}/validator_v5.py" \
    --results    "${RESULTS}"    \
    --corpus     "${CORPUS}"     \
    --references "${REFERENCES}" \
    --output     "${OUTPUT_DIR}"

echo "Done. Exit: $?"
