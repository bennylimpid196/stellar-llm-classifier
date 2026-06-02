# Handoff Document: SC Progress → V7
## Clasificador Estelar Híbrido HC+SC — STELLAR
**Fecha de corte:** 2026-05-28 | **Preparado para:** Continuación post-inferencia V7

---

## 1. Resumen del Proyecto

Sistema de clasificación espectral estelar automático sobre datos Gaia DR3. Arquitectura de dos capas:

- **Hard Computing (HC):** Completamente implementado y validado. 498 contratos JSON en `hc_contracts.json` bajo `pipeline_version: HC-2.0`.
- **Soft Computing (SC):** V7 en inferencia activa (job en cola/corriendo en el clúster).

El sistema se llama **STELLAR** — *Spectral Type Estimation via Language Learning and Astronomical Reasoning*.

---

## 2. Infraestructura

**Clúster:** Laboratorio de Supercómputo del Bajío (Lab-SB), CIMAT Guanajuato
**Partición SLURM:** `GPU`
**Hardware GPU:** NVIDIA Titan RTX 24 GB × 2 por nodo (nodos `g-0-1` a `g-0-12`)
**Límite QOS:** 1 nodo GPU por usuario simultáneamente
**Walltime real:** 120 horas (inferencia), 2 horas (validación)
**Conda env:** `prometheus`
**Ruta del modelo:** `/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/AstroLlama/AstroSage-8B`
**Ruta del proyecto:** `/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/STELLAR/Soft_Computing/`

**Variables de entorno obligatorias en SLURM:**
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

**Activación de conda:**
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prometheus
```

---

## 3. Estructura de Archivos (estado post-limpieza 2026-05-28)

```
Soft_Computing/
├── knowledge_base/                      ← KB del RAG (6 documentos, NUEVO en V7)
│   ├── mk_classification_rules.md
│   ├── subtype_calibration_guide.md
│   ├── luminosity_class_guide.md
│   ├── quality_flags_interpretation.md
│   ├── population_classification_guide.md
│   └── stellar_description_format.md
├── Data/
│   ├── catalog.csv
│   ├── ground_truth_final.csv
│   ├── hc_contracts.json                ← contratos HC-2.0 (498 estrellas)
│   ├── hc_anchor_report.json
│   ├── hc_anchor_summary.json
│   ├── reference_descriptions.json      ← 17 descripciones humanas para ROUGE/BERTScore
│   ├── corpus_build_report.json
│   ├── stellar_corpus_v5.json           ← corpus V5 (con hc_anchor, español)
│   ├── stellar_corpus_v7.json           ← corpus V7 (hc_anchor en inglés, A/B corregido)
│   ├── rag_cache/                       ← cache pre-encodeo RAG (NUEVO en V7)
│   │   ├── query_vectors.npy            ← matriz (498, 384) float32
│   │   ├── query_index.json             ← {source_id: row_index}
│   │   └── corpus_hash.txt              ← sha256 del corpus v7
│   └── archive/
│       └── stellar_corpus_legacy_v0.json
├── scripts/
│   ├── engine.py                        ← RAGEngine (NUEVO en V7)
│   ├── pre_encode_queries.py            ← pre-encodeo batch (NUEVO en V7)
│   ├── corpus_builder_v7.py             ← genera stellar_corpus_v7.json
│   ├── system_prompt_v7.py              ← prompt V7 con RAG
│   ├── inference_manager_v7.py          ← motor de inferencia V7
│   ├── validator_v7.py                  ← validador V7 con BERTScore y Level 5 RAG
│   └── archive/                         ← versiones V1–V6 preservadas
│       ├── corpus_builder.py / _v5.py
│       ├── inference_manager.py / _v2..v6.py
│       ├── system_prompt.py / _v2..v6.py
│       └── validator.py / _v5.py / _v6.py
├── outputs/
│   ├── sc/                              ← resultados inferencia V1–V5 (legacy)
│   ├── sc_v7/                           ← resultados inferencia V7 (activo)
│   ├── validation_v3/ .. validation_v6/ ← reportes históricos
│   └── validation_v7/                   ← reporte V7 (pendiente post-inferencia)
├── logs/                                ← todos los logs centralizados (49+ archivos)
├── SLURM/
│   ├── run_inference_v7.slurm           ← ACTIVO (job en cola/corriendo)
│   ├── run_validation_v7.slurm          ← listo (requiere SC_JOB_ID)
│   └── archive/                         ← SLURMs V1–V6
└── README_cluster.md
```

---

## 4. Pipeline V7 — Flujo Completo

```
hc_contracts.json
      ↓ corpus_builder_v7.py          (login node, CPU, ~0.05s)
stellar_corpus_v7.json
      ↓ pre_encode_queries.py         (login node, CPU, ~27s con prometheus)
Data/rag_cache/  (query_vectors.npy + query_index.json + corpus_hash.txt)
      ↓ inference_manager_v7.py       (SLURM GPU, Titan RTX, job en cola)
outputs/sc_v7/sc_results_v7_<JID>.json
      ↓ validator_v7.py               (SLURM GPU, ~2h por BERTScore)
outputs/validation_v7/  (5 archivos de reporte)
```

---

## 5. Versiones del SC — Tabla Comparativa

| Versión | Accuracy | Kappa | Macro F1 | ROUGE-1 | Cambio clave |
|---|---|---|---|---|---|
| V1 | 0.4192 | 0.3015 | 0.3592 | — | Baseline |
| V3 | 0.4499 | 0.3400 | 0.3978 | — | Letra B separada en dos pasos |
| V4 | 0.5542 | 0.4654 | 0.4827 | — | Few-shot por clase |
| V6 ★ | 0.7579 | 0.7083 | 0.6710 | 0.2654 | HC ancla letra y población |
| **V7** | *pendiente* | *pendiente* | *pendiente* | *pendiente* | RAG + A/B hard anchor + BERTScore |

---

## 6. Cambios Arquitectónicos en V7

### V7-CHG-1 | corpus_builder_v7.py — tres transformaciones sobre V5

**1. Regla A/B boundary (nueva):**
```
Si Teff ∈ [9700, 10100) K AND logg ≥ 3.8 → asignar letra A (en lugar de B)
```
Resultado en corpus: **8 estrellas** corregidas de B→A.

```
source_id              Teff (K)   logg    Corrección
1562168842092340352    10000      3.936   B→A
4449357115297455872    10004      3.998   B→A
1335756411268494080    10071      4.044   B→A
4297359494014043264    10075      4.170   B→A
2701125861831057792    10055      4.116   B→A
3177358226524915456    10008      4.033   B→A
3022709854493303680    10009      3.999   B→A
450211563029092352     10075      4.112   B→A
```

**2. Traducción de población a inglés:**
- `"Disco Fino"` → `"Thin Disk"`
- `"Disco Grueso"` → `"Thick Disk"`
- `"Halo"` → `"Halo"` (sin cambio)
- 411 registros afectados, 87 Halo sin cambio.

**3. NaN → null en serialización JSON:**
Los campos `cat_triplet[*].ew_aa` y `fwhm_nm` con NaN se serializan como `null` (JSON estándar). Downstream compatible con `bert-score`, `pandas.read_json`, JS/Rust.

### V7-CHG-2 | RAG integration

**Knowledge base:** 6 documentos Markdown en `knowledge_base/`, ~38 chunks semánticos indexados.

| Documento | Contenido | Chunks aprox. |
|---|---|---|
| `mk_classification_rules.md` | Umbrales HC, regla A/B, prioridad de población | 5 |
| `subtype_calibration_guide.md` | Teff→subtipo por clase, casos problemáticos F/G | 8 |
| `luminosity_class_guide.md` | logg→luminosidad, systematic offsets GSP-Phot | 5 |
| `quality_flags_interpretation.md` | Flags HC, impacto en confidence y descripción | 9 |
| `population_classification_guide.md` | Cadena de decisión, cobertura alpha-fe 53% | 6 |
| `stellar_description_format.md` | Formato exacto, límites de palabras, ejemplos | 5 |

**Modelo de embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (80MB, CPU-only, 384-dim).

**Cache pre-encodeo:**
- `query_vectors.npy`: shape (498, 384), 0.7 MB
- `query_index.json`: 498 entries (source_id → row)
- `corpus_hash.txt`: sha256 `90eb4934d7104aa2f9b07b4fcd493ec1...`
- Tiempo de retrieval con cache: ~1ms/estrella (vs ~665ms sin cache)
- 0 queries vacíos — todas las 498 estrellas tienen contexto RAG

**Flujo en inference_manager_v7.py:**
```python
rag_context, rag_top_label = rag.retrieve(star)   # tuple[str, str]
result["rag_top_chunk"] = rag_top_label            # estampado para Level 5
```

### V7-CHG-3 | system_prompt_v7.py

Cambios respecto a V6:
- A/B boundary: **hard anchor** (el LLM no razona sobre el boundary — ya está resuelto por el builder)
- Binary penalty: floor explícito de **−0.10 mínimo** en spectral_type_confidence (V6 tenía multiplicador ×0.85 que no se aplicaba)
- Población en inglés en schema y en todo el texto
- Límite `population_context` 30 palabras reforzado con ejemplo inline
- Bloque RAG inyectado entre diagnósticos e instrucciones

### V7-CHG-4 | validator_v7.py

**Level 4a-bis — BERTScore (nuevo):**
- Modelo: `microsoft/deberta-xlarge-mnli`
- Mismas 17 estrellas de referencia que ROUGE
- Reporta P/R/F1 por subcampo y agregado
- Si `bert-score` no está instalado, el nivel se salta con warning (ROUGE no se ve afectado)

**Level 5 — RAG Retrieval Impact (reemplaza A/B boundary de V6):**
- Mide si el chunk top recuperado es temáticamente consistente con la clase espectral
- Compara ROUGE-1 de estrellas con chunk relevante vs irrelevante
- Mide flag hit rate (¿se recuperó quality_flags chunk para estrellas con flags activos?)
- Requiere el campo `rag_top_chunk` en cada resultado (estampado por inference_manager_v7)

**Población en inglés:** `_VALID_POPULATIONS = {"Halo", "Thick Disk", "Thin Disk"}`

---

## 7. Bugs Corregidos en Esta Sesión

### Bug 1 — Log duplicado en pre_encode_queries.py

**Síntoma:** todos los mensajes de log aparecían dos veces al correr `pre_encode_queries.py`.

**Causa raíz:** dos problemas acoplados:
1. `engine.py` tenía `logging.basicConfig()` al nivel de módulo — al importarlo registraba un segundo handler en el root logger.
2. `pre_encode_queries.py` tenía un doble intento de import (`from engine` + fallback `from rag.engine`) — en STELLAR ambos tenían éxito, registrando el mensaje dos veces.

**Fix aplicado en el clúster:**

`engine.py` — eliminado `logging.basicConfig()` (librería no debe configurar root logger):
```python
# ANTES:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# DESPUÉS:
log = logging.getLogger(__name__)
```

`pre_encode_queries.py` — eliminado fallback `from rag.engine`:
```python
# ANTES:
try:
    from engine import _build_query
    log.info("_build_query imported from engine.py...")
except ImportError:
    try:
        from rag.engine import _build_query
        log.info("_build_query imported from rag.engine.")
    except ImportError as exc:
        raise ImportError(...) from exc

# DESPUÉS:
try:
    from engine import _build_query
    log.info("_build_query imported from engine.py (consistency guaranteed).")
except ImportError as exc:
    raise ImportError(
        "Could not import _build_query from engine.py. "
        "Run from scripts/ or ensure engine.py is in the Python path."
    ) from exc
```

### Bug 2 — BASE_DIR y capitalización en pre_encode_queries.py

**Síntoma:** rutas por defecto no encontraban `Data/` porque apuntaban a `scripts/data/`.

**Fix:**
```python
# ANTES:
BASE_DIR = Path(__file__).resolve().parent       # → scripts/
DEFAULT_CORPUS = BASE_DIR / "data" / "..."       # minúscula

# DESPUÉS:
BASE_DIR = Path(__file__).resolve().parent.parent  # → Soft_Computing/
DEFAULT_CORPUS = BASE_DIR / "Data" / "..."          # mayúscula (Linux case-sensitive)
```

---

## 8. Estado del Job V7

**Job ID inferencia:** asignado por SLURM al lanzar (ver con `squeue -u $USER`)
**Estado al corte de este documento:** en cola (`PD`) esperando que termine job MAGMA (187536)
**Nodo esperado:** cualquiera de `g-0-1` a `g-0-12` (excepto `g-0-7` ocupado por MAGMA)
**Walltime configurado:** 120 horas

---

## 9. Cómo Lanzar la Validación V7

Una vez termine la inferencia:

```bash
# 1. Obtener el JID del job de inferencia
sacct -u $USER --format=JobID,JobName,State,End | grep STELLAR

# 2. Editar el SLURM de validación
nano SLURM/run_validation_v7.slurm
# Reemplazar: SC_JOB_ID="REPLACE_WITH_INFERENCE_JOB_ID"
# Por:        SC_JOB_ID="<JID real>"

# 3. Lanzar
cd ~/proyects/AstroLLM/STELLAR/Soft_Computing
sbatch SLURM/run_validation_v7.slurm

# O encadenado (si el job de inferencia sigue corriendo):
sbatch --dependency=afterok:<INFERENCE_JID> SLURM/run_validation_v7.slurm
```

**Outputs esperados en `outputs/validation_v7/`:**

| Archivo | Contenido |
|---|---|
| `validation_report_v7.json` | Reporte completo todos los niveles |
| `validation_summary_v7.txt` | Resumen human-readable |
| `rouge_scores_v7.json` | ROUGE-1 y ROUGE-L por estrella y subcampo |
| `bertscore_v7.json` | BERTScore P/R/F1 por estrella (si bert-score instalado) |
| `rag_impact_v7.json` | Level 5: hit rate, ROUGE delta, por clase |
| `confusion_spectral_v7.csv` | Matriz de confusión 7×7 |

---

## 10. Métricas Objetivo V7

Basadas en los hallazgos de V6 y las correcciones implementadas:

| Métrica | V6 | Objetivo V7 | Cambio que lo activa |
|---|---|---|---|
| Overall accuracy | 0.758 | ≥ 0.758 | A/B correction suma 8 estrellas bien clasificadas |
| F subtype accuracy | 0.000 | > 0.100 | KB chunk `subtype_calibration_guide` orienta al LLM |
| Binary penalty aplicada | ❌ | ✅ | Floor explícito −0.10 en system_prompt_v7 |
| Word limit violations | 55.5% | < 30% | Ejemplo inline + instrucción reforzada |
| ROUGE-1 mean | 0.265 | > 0.265 | RAG aporta contexto de formato canónico |
| BERTScore F1 | — | baseline | Primera medición |
| RAG hit rate | — | > 0.70 | Validación de que el retrieval es relevante |

---

## 11. Hallazgos Científicos Acumulados (V1–V6)

1. **Prior fuerte hacia G (V1–V4):** AstroSage-8B tiene sesgo de entrenamiento hacia tipos G. Neutralizado en V6 al anclar la letra con HC.
2. **F subtype colapsa a 0% en V6:** el LLM no discrimina subtipos dentro de F. La banda Teff 6000–7500 K es la más continua espectralmente; el RAG de V7 intenta proveer anclas explícitas.
3. **Confidence invertida persiste:** estrellas incorrectas tienen confianza ≥ correctas en todas las versiones. En V6: Spearman r = −0.171 (p=0.0005). La inversión es menor que en V5 (r=−0.316) pero persiste.
4. **Binary penalty no se aplica con multiplicador:** en V6, binarias tenían confidence 0.784 vs 0.762 en no-binarias — invertido. V7 usa floor explícito −0.10 en lugar de multiplicador.
5. **Regla logg supera al LLM en A/B boundary:** logg ≥ 3.8 → A logra 76.5% de accuracy vs 43.1% del HC y del LLM en V6. Por eso en V7 el builder aplica la regla determinísticamente.
6. **population_context es el campo más verboso:** violaciones de límite en 55.5% de las estrellas en V6, principalmente en este subcampo.

---

## 12. Pendientes Inmediatos Post-Inferencia

1. **Lanzar validación V7** con el JID de inferencia (ver Sección 9).
2. **Instalar bert-score** en prometheus si no está: `pip install bert-score` (requiere ~2GB para descargar deberta-xlarge-mnli la primera vez — necesita conexión o modelo descargado).
3. **Comparar ROUGE V6 vs V7** — si el RAG no mejora ROUGE, revisar si los chunks recuperados son relevantes via `rag_impact_v7.json`.
4. **Analizar Level 3** — verificar que `penalty_applied=True` y `mean_delta ≤ −0.10` en la sección binary_penalty_check del reporte.
5. **Preparar V8 si F sigue en 0%** — posible estrategia: few-shot de subtipos F en el system prompt, similar a los B subtype examples de V6.
