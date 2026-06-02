#!/bin/bash
# =============================================================================
# run_inference.slurm
# STELLAR — SC Stellar Inference
# Lab-SB CIMAT (Partition: GPU) — Pipeline Version HC-2.0
#
# Usage:
#   sbatch run_inference.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/sc_<JOB_ID>.out
#
# Cancel:
#   scancel <JOB_ID>
# =============================================================================

#SBATCH --job-name=STELLAR_inference
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/sc_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/logs/sc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cesar.aguirre@cimat.mx

# =============================================================================
# Entorno — igual que MAGMA-01
# =============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate prometheus

# Modo offline obligatorio — HuggingFace no debe intentar conectarse
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# =============================================================================
# Rutas del proyecto
# =============================================================================

PROJECT_DIR="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing"
MODEL_PATH="/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/AstroLlama/AstroSage-8B"

# Derivadas — no modificar
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
echo " STELLAR — SC Inference"
echo " Job ID  : ${SLURM_JOB_ID}"
echo " Nodo    : ${SLURMD_NODENAME}"
echo " Inicio  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo " Modelo  : ${MODEL_PATH}"
echo " Corpus  : ${CORPUS}"
echo " Output  : ${OUTPUT_DIR}"
echo "============================================================"

START_TIME=$SECONDS

# Verificar GPU
echo ""
echo "[1/2] Verificando entorno y GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Verificar modelo
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Modelo no encontrado en ${MODEL_PATH}"
    exit 1
fi

# Verificar corpus
if [ ! -f "${CORPUS}" ]; then
    echo "[ERROR] Corpus no encontrado en ${CORPUS}"
    echo "        Corre primero: python3 scripts/corpus_builder.py"
    exit 1
fi

# =============================================================================
# Inferencia
# =============================================================================

echo ""
echo "[2/2] Iniciando inferencia con AstroSage-Llama..."
echo "      Corpus      : ${CORPUS}"
echo "      Lotes       : 50 estrellas/lote"
echo "      Reintentos  : 3"
echo "      Max tokens  : 512"

python3 "${SCRIPTS_DIR}/inference_manager.py" \
    --corpus         "${CORPUS}"              \
    --model          "${MODEL_PATH}"           \
    --output         "${OUTPUT_DIR}"           \
    --batch-size     50                        \
    --max-retries    3                         \
    --max-new-tokens 512                       \
    --job-id         "${SLURM_JOB_ID}"

EXIT_CODE=$?

# Verificar output
RESULTADOS="${OUTPUT_DIR}/sc_results_${SLURM_JOB_ID}.json"
if [ ! -f "${RESULTADOS}" ]; then
    echo "[ADVERTENCIA] sc_results_${SLURM_JOB_ID}.json no generado. Revisar inference_manager_sc.log"
else
    N=$(python3 -c "import json; print(len(json.load(open('${RESULTADOS}'))))")
    echo "[OK] Resultados: ${N} estrellas clasificadas → ${RESULTADOS}"
fi

echo ""
echo "============================================================"
echo " STELLAR — Inferencia completada"
echo " Fin         : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Tiempo total: $((SECONDS - START_TIME))s ($((( SECONDS - START_TIME) / 60)) min)"
echo " Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}