#!/bin/bash
# =============================================================================
# run_inference_v2.slurm
# STELLAR — SC Stellar Inference — System Prompt v2
# Lab-SB CIMAT (Partition: GPU) — Pipeline Version HC-2.0
#
# Cambios vs run_inference.slurm (v1):
#   - Usa inference_manager_v2.py (importa system_prompt_v2.py)
#   - Outputs: batch_v2_*.json, sc_results_v2_<JID>.json, sc_failed_v2_<JID>.json
#   - Log: inference_manager_sc_v2.log
#   - Job name: STELLAR_v2 (distinguible en squeue)
#   - No mezcla con resultados v1 — Stateful Batching lee solo batch_v2_*.json
#
# Usage:
#   sbatch run_inference_v2.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/sc_v2_<JOB_ID>.out
# =============================================================================

#SBATCH --job-name=STELLAR_v2
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/sc_v2_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/sc_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cesar.aguirre@cimat.mx

# =============================================================================
# Entorno
# =============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate prometheus

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# =============================================================================
# Rutas del proyecto
# =============================================================================

PROJECT_DIR="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing"
MODEL_PATH="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/AstroLlama/AstroSage-8B"

CORPUS="${PROJECT_DIR}/Data/stellar_corpus.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/sc"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
LOGS_DIR="${PROJECT_DIR}/logs"

# =============================================================================
# Setup
# =============================================================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

echo "============================================================"
echo " STELLAR — SC Inference v2 (system_prompt_v2)"
echo " Job ID  : ${SLURM_JOB_ID}"
echo " Nodo    : ${SLURMD_NODENAME}"
echo " Inicio  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo " Modelo  : ${MODEL_PATH}"
echo " Corpus  : ${CORPUS}"
echo " Output  : ${OUTPUT_DIR}"
echo " Outputs : batch_v2_*.json | sc_results_v2_<JID>.json"
echo "============================================================"

START_TIME=$SECONDS

echo ""
echo "[1/2] Verificando entorno y GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Modelo no encontrado en ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${CORPUS}" ]; then
    echo "[ERROR] Corpus no encontrado en ${CORPUS}"
    exit 1
fi

# =============================================================================
# Inferencia v2
# =============================================================================

echo ""
echo "[2/2] Iniciando inferencia con system_prompt_v2..."

python3 "${SCRIPTS_DIR}/inference_manager_v2.py" \
    --corpus         "${CORPUS}"                 \
    --model          "${MODEL_PATH}"              \
    --output         "${OUTPUT_DIR}"              \
    --batch-size     50                           \
    --max-retries    3                            \
    --max-new-tokens 600                          \
    --job-id         "${SLURM_JOB_ID}"

EXIT_CODE=$?

RESULTADOS="${OUTPUT_DIR}/sc_results_v2_${SLURM_JOB_ID}.json"
if [ ! -f "${RESULTADOS}" ]; then
    echo "[ADVERTENCIA] sc_results_v2_${SLURM_JOB_ID}.json no generado. Revisar inference_manager_sc_v2.log"
else
    N=$(python3 -c "import json; print(len(json.load(open('${RESULTADOS}'))))")
    echo "[OK] Resultados v2: ${N} estrellas clasificadas → ${RESULTADOS}"
fi

echo ""
echo "============================================================"
echo " STELLAR v2 — Inferencia completada"
echo " Fin         : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Tiempo total: $((SECONDS - START_TIME))s ($((( SECONDS - START_TIME) / 60)) min)"
echo " Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
