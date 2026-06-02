#!/bin/bash
# =============================================================================
# run_inference_v3.slurm
# STELLAR — SC Stellar Inference — System Prompt v3
# Lab-SB CIMAT (Partition: GPU) — Pipeline Version HC-2.0
#
# Cambios vs v2:
#   - Step B separado en B1 (letra) y B2 (sub-tipo) — fix al G-collapse
#   - Step E con deducciones IF/THEN explícitas — fix confidence mecánica
#   - Outputs: batch_v5_*.json, sc_results_v5_<JID>.json
#   - Log: sc_v3_<JID>.out | inference_manager_sc_v3.log
#   - Job name: STELLAR_v5
#
# PUEDE CORRER EN PARALELO CON v2 — no comparten archivos de output.
#
# Usage:
#   sbatch run_inference_v3.slurm
# =============================================================================

#SBATCH --job-name=STELLAR_v5
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/sc_v5_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/sc_v5_%j.err
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
# Rutas
# =============================================================================

PROJECT_DIR="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing"
MODEL_PATH="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/AstroLlama/AstroSage-8B"

CORPUS="${PROJECT_DIR}/Data/stellar_corpus_v5.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/sc"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
LOGS_DIR="${PROJECT_DIR}/logs"

# =============================================================================
# Setup
# =============================================================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

echo "============================================================"
echo " STELLAR — SC Inference v3 (HC pre-computes letter+population; LLM writes description)"
echo " Job ID  : ${SLURM_JOB_ID}"
echo " Nodo    : ${SLURMD_NODENAME}"
echo " Inicio  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo " Modelo  : ${MODEL_PATH}"
echo " Corpus  : ${CORPUS}"
echo " Outputs : batch_v5_*.json | sc_results_v5_<JID>.json"
echo "============================================================"

START_TIME=$SECONDS

echo ""
echo "[1/2] GPU status:"
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
# Inferencia v3
# =============================================================================

echo ""
echo "[2/2] Iniciando inferencia con system_prompt_v5..."

python3 "${SCRIPTS_DIR}/inference_manager_v5.py" \
    --corpus         "${CORPUS}"                 \
    --model          "${MODEL_PATH}"              \
    --output         "${OUTPUT_DIR}"              \
    --batch-size     50                           \
    --max-retries    3                            \
    --max-new-tokens 900                          \
    --job-id         "${SLURM_JOB_ID}"

EXIT_CODE=$?

RESULTADOS="${OUTPUT_DIR}/sc_results_v5_${SLURM_JOB_ID}.json"
if [ -f "${RESULTADOS}" ]; then
    N=$(python3 -c "import json; print(len(json.load(open('${RESULTADOS}'))))")
    echo "[OK] Resultados v3: ${N} estrellas → ${RESULTADOS}"
else
    echo "[ADVERTENCIA] sc_results_v5_${SLURM_JOB_ID}.json no generado."
fi

echo ""
echo "============================================================"
echo " STELLAR v3 — Completado"
echo " Fin         : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Tiempo total: $((SECONDS - START_TIME))s ($((( SECONDS - START_TIME) / 60)) min)"
echo " Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
