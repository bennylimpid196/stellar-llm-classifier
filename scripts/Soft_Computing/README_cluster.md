# Guía de Despliegue — Lab-SB CIMAT
## Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0

---

## Estructura de directorios esperada en el clúster

```
/home/<usuario>/stellar_classifier/
├── scripts/
│   ├── corpus_builder.py
│   ├── system_prompt.py
│   ├── inference_manager.py
│   └── validator.py
├── data/
│   ├── hc_contracts.json          ← output del pipeline HC
│   ├── ground_truth_final.csv
│   └── stellar_corpus.json        ← generado por corpus_builder.py
├── models/
│   └── AstroSage-Llama/           ← pesos del modelo (ya en el clúster)
├── outputs/
│   ├── sc/                        ← resultados de inferencia
│   └── validation/                ← métricas de validación
├── logs/                          ← stdout/stderr de SLURM
├── run_inference.slurm
└── run_validation.slurm
```

---

## Pasos antes del primer run

### 1. Verificar el walltime de la partición GPU

```bash
scontrol show partition GPU | grep MaxTime
```

Actualiza `--time` en `run_inference.slurm` con el valor que devuelva.
MAGMA-01 operó bajo un límite de 72h — usa ese como referencia si no confirmas.

### 2. Verificar módulos disponibles

```bash
module avail
```

Si el entorno Python no viene preinstalado como módulo, usa conda:

```bash
conda create -n astro python=3.10
conda activate astro
pip install torch transformers pandas scikit-learn scipy
```

### 3. Editar las variables TODO en los scripts SLURM

En `run_inference.slurm`:
```bash
PROJECT_DIR="/home/<tu_usuario>/stellar_classifier"
MODEL_PATH="/home/<tu_usuario>/models/AstroSage-Llama"
CONDA_ENV="astro"
```

En `run_validation.slurm`:
```bash
PROJECT_DIR="/home/<tu_usuario>/stellar_classifier"
CONDA_ENV="astro"
SC_JOB_ID="REPLACE_WITH_INFERENCE_JOB_ID"   # se llena después
```

---

## Flujo de ejecución completo

### Paso 1 — Construir el corpus (una sola vez, en el nodo de login)

```bash
# No necesita GPU — correr directamente en el nodo de login
conda activate astro
python3 scripts/corpus_builder.py \
    --contracts    data/hc_contracts.json \
    --ground-truth data/ground_truth_final.csv \
    --output       data/
```

Verifica que `data/stellar_corpus.json` existe y tiene 498 entradas.

### Paso 2 — Lanzar inferencia

```bash
sbatch run_inference.slurm
```

El sistema devuelve un `JOB_ID`. Guárdalo.

```bash
# Monitorear el job
squeue -u $USER

# Ver el log en tiempo real
tail -f inference_manager_sc.log

# Ver stdout del job (reemplaza JOBID)
tail -f logs/sc_JOBID.out
```

### Paso 3 — Lanzar validación

Opción A — Manual (después de que termine la inferencia):
```bash
# Editar SC_JOB_ID en run_validation.slurm con el JOB_ID del paso anterior
vim run_validation.slurm
sbatch run_validation.slurm
```

Opción B — Encadenado automático:
```bash
# SLURM lanza la validación solo si la inferencia terminó con éxito
sbatch --dependency=afterok:<INFERENCE_JOB_ID> run_validation.slurm
```

### Paso 4 — Revisar resultados

```bash
cat outputs/validation/validation_summary.txt
cat outputs/validation/validation_report.json
ls  outputs/validation/*.csv
```

---

## Stateful Batching — reanudación tras interrupción SLURM

Si el job es interrumpido (límite de walltime), simplemente vuelve a lanzarlo:

```bash
sbatch run_inference.slurm
```

El `inference_manager.py` detecta automáticamente los `batch_*.json` ya escritos
en `outputs/sc/` y retoma desde donde quedó. **No reprocesa ninguna estrella ya
clasificada.**

---

## Comandos SLURM útiles

```bash
squeue -u $USER                    # jobs activos del usuario
squeue -c GPU                      # estado de la partición GPU
scancel <JOB_ID>                   # cancelar un job
sinfo                              # ver todas las particiones y nodos
scontrol show partition GPU        # detalles de la partición GPU (incluye walltime)
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS   # resumen post-ejecución
```

---

## Recursos del clúster — partición GPU

| Parámetro | Valor |
|---|---|
| Nodos | g-0-1 a g-0-12 (12 nodos) |
| GPU por nodo | 2× NVIDIA Titan RTX 24 GB |
| RAM por nodo | 128 GB |
| CPU por nodo | Intel Xeon Silver 4214, 24 cores |
| Máx. nodos simultáneos por usuario | 4 |
| Walltime | Verificar con `scontrol show partition GPU` |

Fuente: Manual de Acceso a Recursos — Laboratorio de Supercómputo del Bajío (CIMAT).
