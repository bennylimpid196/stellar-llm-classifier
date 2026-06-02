#!/bin/bash
# =============================================================================
# run_validation.slurm
# SC Validation — Lab-SB CIMAT (Partition: GPU)
# Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
#
# Este job es CPU-only. Si tienes acceso a C2, cambia --partition=GPU
# por --partition=C2 y elimina la línea --gres para ahorrar recursos GPU.
#
# Usage:
#   # Manual (después de que termine la inferencia):
#   sbatch SLURM/run_validation.slurm
#
#   # Encadenado automático:
#   sbatch --dependency=afterok:<INFERENCE_JOB_ID> SLURM/run_validation.slurm
#
# TODO antes de lanzar:
#   [1] Llenar SC_JOB_ID con el JOB_ID devuelto por run_inference.slurm
# =============================================================================

#SBATCH --job-name=STELLAR_validation
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/validation_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/validation_%j.err

# =============================================================================
# Rutas del proyecto
# =============================================================================

PROJECT_DIR="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing"
CONDA_ENV="prometheus"

# TODO [1]: reemplaza con el JOB_ID real de la inferencia completada
SC_JOB_ID="184048"

# Derivadas — no modificar
RESULTS="${PROJECT_DIR}/outputs/sc/sc_results_v4_${SC_JOB_ID}.json"
CORPUS="${PROJECT_DIR}/Data/stellar_corpus.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/validation_v4"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
LOGS_DIR="${PROJECT_DIR}/logs"

# =============================================================================
# Setup
# =============================================================================

echo "============================================================"
echo "STELLAR Validation — SLURM Job ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Date      : $(date)"
echo "Results   : ${RESULTS}"
echo "Corpus    : ${CORPUS}"
echo "Output    : ${OUTPUT_DIR}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# Verificar inputs
if [ ! -f "${RESULTS}" ]; then
    echo "ERROR: Resultados SC no encontrados en ${RESULTS}"
    echo "Verifica que SC_JOB_ID=${SC_JOB_ID} es correcto."
    exit 1
fi

if [ ! -f "${CORPUS}" ]; then
    echo "ERROR: Corpus no encontrado en ${CORPUS}"
    exit 1
fi

# =============================================================================
# Validación
# =============================================================================

echo ""
echo "Iniciando validación — $(date)"

python3 "${SCRIPTS_DIR}/validator.py" \
    --results "${RESULTS}"            \
    --corpus  "${CORPUS}"             \
    --output  "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "Validación terminada — $(date) — exit code: ${EXIT_CODE}"
echo "Outputs en: ${OUTPUT_DIR}"
echo "============================================================"

exit ${EXIT_CODE}
