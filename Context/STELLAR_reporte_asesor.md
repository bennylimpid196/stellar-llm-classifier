# STELLAR — Reporte de Avance para Asesoría
## Clasificador Estelar Híbrido HC+SC sobre Gaia DR3
**Fecha:** Mayo 2026 | **Versión de pipeline:** HC-2.0 / SC-V5 completada / SC-V6 en desarrollo  
**Institución:** CIMAT — Laboratorio de Supercómputo del Bajío (Lab-SB)

---

## 1. Visión General del Proyecto

**STELLAR** (*Spectral Type Estimation via Language Learning and Astronomical Reasoning*) es un sistema de clasificación espectral estelar automático, explicable y de alta precisión construido sobre datos del satélite Gaia (Data Release 3). La arquitectura es híbrida en dos capas:

- **Hard Computing (HC):** módulos deterministas que extraen parámetros físicos limpios y generan banderas lógicas a partir de los datos crudos de Gaia. Produce un contrato JSON estructurado por estrella.
- **Soft Computing (SC):** un LLM astrofísico especializado (AstroSage-8B, basado en LLaMA) que recibe el contrato pre-digerido del HC y emite una clasificación espectral completa en formato JSON, incluyendo tipo MK, subtipo numérico, clase de luminosidad, pertenencia galáctica y una descripción científica estructurada en lenguaje natural.

La hipótesis central de la tesis es que la arquitectura híbrida HC+SC supera a cada componente por separado: el HC aporta precisión determinista y trazabilidad; el SC aporta capacidad de razonamiento multiparamétrico y generación de lenguaje explicable.

---

## 2. Dataset — DB-1-2026-04-28

### 2.1 Composición y fuentes

El conjunto de datos fue construido a partir del archivo Gaia DR3 e integra tres fuentes complementarias:

**Catálogo tabular (`catalog.csv`):** 498 estrellas, 22 columnas. Incluye astrometría (paralaje, movimiento propio, coordenadas), fotometría (magnitud G, extinción E(BP-RP)), parámetros físicos del pipeline GSP-Phot de Gaia (Teff, logg, [M/H], [α/Fe]) y datos espectroscópicos del pipeline ESP-ELS (EW H-alpha) y GSP-Spec ([Fe/M]). Las columnas más críticas y sus caveats son:

| Campo | Cobertura | Caveat |
|---|---|---|
| `teff_gspphot` | ~95% | Incertidumbre inherente ±100-200 K |
| `logg_gspphot` | ~95% | Pasa directo al SC sin procesamiento HC |
| `fem_gspspec` | ~44% | [Fe/H] espectroscópico; más preciso cuando disponible |
| `alphafe_gspspec` | ~53% | Clave para discriminar Disco Grueso vs Halo |
| `ew_espels_halpha` | ~60% | Signo negativo = emisión real (convención ESP-ELS) |
| `ebpminrp_gspphot` | ~80% | Alta tasa de NaN; A_G = 0 cuando ausente |

**Espectros:** ~430 espectros RVS (846–870 nm, Ca II Triplete) en formato `.npz` por `source_id`; espectros BP/RP de (498, 343) puntos a 336–1020 nm.

**Ground truth de validación:** construido en dos pasadas paralelas:
- **SIMBAD:** tipo espectral MK canónico disponible para 491/498 estrellas (98.6% cobertura).
- **PASTEL:** catálogo de parámetros estelares de alta resolución (mediana multi-estudio) con Teff disponible para 340/498, logg para 194/498 y [Fe/H] para 183/498. El campo `n_pastel_measurements` indica cuántos estudios independientes contribuyen a cada mediana — una estrella con 32 estudios (e.g., *mu. Ara*) tiene ground truth sustancialmente más confiable que una con 1.

### 2.2 Pipeline de construcción del ground truth

```
Gaia DR3 archive → catalog.csv (22 cols, 498 estrellas)
       ↓ validation_data_fetcher_v2.py
   Resolución SIMBAD de source_ids → nombres canónicos
       ↓ join_pastel_local.py
   Join con pastel.dat.gz (CDS) — dos pasadas: nombre + alias HD/HIP
       ↓ ground_truth_cleaner.py
   Filtro de outliers físicos → ground_truth_clean.csv
```

### 2.3 Distribución espectral del corpus

| Clase | N | Clase de luminosidad dominante |
|---|---|---|
| A | 79 | Enanas / Subgigantes |
| B | 99 | Enanas / Subgigantes |
| F | 93 | Enanas / Subgigantes |
| G | 80 | Enanas / Gigantes |
| K | 90 | Enanas / Gigantes |
| M | 57 | Gigantes / Enanas |

Población galáctica (calculada determinísticamente por el HC): Disco Fino 265 (53.2%), Disco Grueso 146 (29.3%), Halo 87 (17.5%).

---

## 3. Arquitectura HC+SC y Trazabilidad

### 3.1 Módulos del Hard Computing (HC-2.0)

El HC procesa cada estrella de forma secuencial a través de cuatro módulos deterministas:

**AstrometryAgent:** calcula magnitud absoluta M_G = G + 5 + 5·log₁₀(π/1000) − 2.74·E(BP-RP) y velocidad tangencial V_tan = 4.74 · √(μα*² + μδ²) / π. Guards implementados: paralaje negativo/cero → status "partial"; SNR de paralaje < 5 → M_G y V_tan suprimidos; extinción NaN → A_G = 0 con warning.

**BinaryDetectorAgent:** detecta sistemas múltiples no resueltos mediante tres indicadores independientes: RUWE > 1.4 (Lindegren et al. 2021), variabilidad RV con umbral adaptativo por Teff (2–10 km/s), y presencia de solución NSS en Gaia. El flag `is_binary_candidate` es el OR lógico de los tres.

**ContinuumAgent:** normaliza los espectros BP/RP y RVS mediante splines cúbicos con sigma-clipping iterativo. Produce el espectro normalizado para el LineAgent.

**LineAgent:** ajusta perfiles de Voigt sobre el Ca II Triplete (846–870 nm) en los espectros RVS. H-alpha se toma del catálogo (campo ESP-ELS), no del espectro, porque la resolución BP/RP (2 nm/pixel) es insuficiente para ajuste Voigt.

### 3.2 El contrato HC → SC

El output del HC por estrella es un JSON estructurado (*contrato*) que el SC recibe como input. La arquitectura del contrato en V5 se extendió con el campo `hc_anchor`:

```json
{
  "source_id": "...",
  "physical_vector": {
    "teff_k": 5800, "logg": 4.2, "metallicity": -0.3,
    "fe_h": -0.28, "alpha_fe": 0.12, "v_tan": 42.3, "abs_mag": 4.1
  },
  "logical_flags": {
    "is_reliable_parallax": true, "is_giant": false,
    "is_metal_poor": false, "is_binary_candidate": false,
    "is_high_velocity": false, "has_emission": false
  },
  "hc_anchor": {
    "mk_letter": "G",
    "population_group": "Disco Fino",
    "near_teff_boundary": false,
    "near_logg_boundary": false,
    "chemistry_value": -0.28,
    "decision_rule": "chemistry=-0.28 >= -0.2"
  },
  "quality_score": 0.95
}
```

### 3.3 Por qué esta arquitectura mejora la trazabilidad

El contrato HC→SC es el mecanismo central de trazabilidad del sistema. Cada decisión del pipeline tiene origen rastreable:

- Las banderas lógicas (`is_binary_candidate`, `has_emission`, etc.) documentan exactamente qué detectó el HC y bajo qué criterio numérico.
- El `hc_anchor` separa explícitamente qué calculó el HC (letra MK, población, regla de decisión) de qué decidió el LLM (subtipo, clase de luminosidad, confianza, descripción).
- El campo `decision_rule` registra la expresión lógica exacta que produjo cada asignación de población (e.g., `"alpha_fe=0.24>=0.2 AND chemistry=-0.497 in [-1.0,-0.2)"`).
- El `quality_score` actúa como techo de confianza: el LLM no puede reportar confidence mayor que este valor, propagando la incertidumbre del HC hacia el output final.

Esto permite auditar cualquier clasificación del sistema —quién decidió qué, con qué datos y bajo qué regla— sin ambigüedad.

---

## 4. Resultados del Módulo SC — Versiones V1 a V5

### 4.1 Evolución de las versiones

| Versión | Estrategia principal | Accuracy MK letra | Cohen κ | Observaciones |
|---|---|---|---|---|
| V1 | Baseline, prompt simple | ~0.40 | — | Prior G muy fuerte del modelo |
| V2 | Chain-of-thought | — | — | Mejora razonamiento, no accuracy |
| V3 | Structured output | — | — | Inestabilidad por batches |
| V4 | Few-shot (6 ejemplos) | **0.554** | **0.465** | Mejor versión sin ancla HC |
| V5 | HC pre-computa letra + población | **0.832** | **0.798** | Ver nota crítica abajo |

**Nota crítica sobre la accuracy de V5:** la mejora de 83.2% en clasificación MK no representa completamente la capacidad del LLM, ya que el HC ya pre-computó la letra determinísticamente. El LLM recibe la letra como ancla y debe respetarla. La métrica honesta del LLM en V5 es el **subtype accuracy** (Level 4c), que mide lo que el modelo decide verdaderamente de forma autónoma.

### 4.2 Resultados detallados V5

**Level 1 — Clasificación MK gruesa (vs. SIMBAD, n=310):**

| Métrica | Valor |
|---|---|
| Accuracy | 0.832 |
| Cohen κ | 0.798 |
| Macro F1 | 0.704 |
| Near-miss accuracy (d≤1 clase) | 0.997 |

F1 por clase: M=0.989, F=0.938, K=0.892, G=0.880, B=0.733, A=0.500, O=0.000

**Level 4c — Subtype accuracy (lo que realmente decide el LLM):**

| Clase | n | Accuracy subtipo |
|---|---|---|
| M | 48 | **0.833** |
| A | 19 | 0.368 |
| G | 65 | 0.338 |
| K | 38 | 0.342 |
| F | 62 | 0.161 |
| B | 70 | **0.000** |
| **Global** | **302** | **0.305** |

El colapso total de subtipo B (0% de accuracy, 94% de predicciones en "0-1") es el problema de SC más relevante identificado. Ver Sección 5.

**Level 2 — Parámetros físicos (vs. PASTEL):**
- |ΔTeff| medio: 220.6 K
- |Δlogg| medio: 0.216 dex
- Kappa de población vs. PASTEL: −0.125 (ver explicación en Sección 5)

**Level 3 — Calibración de confianza:**
- Confidence cuando el modelo acierta: 0.736
- Confidence cuando el modelo se equivoca: 0.818
- Correlación Spearman r = −0.066 (p=0.24, no significativa)

La confidence está sistemáticamente **invertida**: el modelo es más confiado en sus errores. Este es un problema estructural del LLM que no se resuelve con ajuste de prompt.

**Coverage:** Solo 317/498 estrellas (63.7%) tienen resultado en V5. Las 181 restantes fallaron por truncamiento de JSON (las descripciones excedían el límite de `max_new_tokens=512`) y por artefactos `<<SYS>>` del template LLaMA2. Ambos problemas están corregidos en V6.

---

## 5. Hallazgos Científicos — Lo que Debe Documentarse en la Tesis

### 5.1 El LLM supera al HC en la frontera A/B (hallazgo central)

Este es el resultado más importante para la hipótesis de la tesis.

En la frontera A/B (Teff ≈ 9,950–10,000 K, donde el HC aplica el umbral determinista de 10,000 K), se identificaron 52 estrellas con `near_teff_boundary=True`. De las 18 evaluadas en V5:

| Estrategia | Accuracy en frontera A/B |
|---|---|
| HC determinista (fuerza A siempre) | 50.0% |
| LLM V5 (ancla blanda, decide libremente) | **66.7%** |
| Regla óptima `logg ≥ 3.80 → A` | **83.3%** |

El LLM no ignora la ancla aleatoriamente — utiliza `logg` de forma implícita para desambiguar. Las estrellas donde el LLM acierta y el HC falla tienen logg medio de 3.687 (subgigantes B evolucionadas enfriándose hacia la frontera); las estrellas donde el HC acierta y el LLM falla tienen logg medio de 3.946 (enanas A verdaderas). El LLM reconoce que una estrella con Teff ≈ 10,000 K pero logg bajo es más probablemente una B evolucionada que una A de secuencia principal — un razonamiento multiparamétrico que el umbral determinista del HC no puede capturar.

La regla óptima empírica `logg ≥ 3.80 → A` (83.3% de accuracy) será incorporada al HC en futuras versiones, **validando que el análisis del comportamiento del LLM puede retroalimentar y mejorar el propio HC**.

**Cómo documentarlo en la tesis:** presentar las tres estrategias en una tabla comparativa, con la distribución de logg para los casos LLM✓/HC✗ y LLM✗/HC✓, y argumentar que el LLM actúa como un oráculo estadístico en zonas de ambigüedad donde los umbrales deterministas son insuficientes.

### 5.2 El colapso de subtipo B revela prior de entrenamiento

El modelo colapsa el 94% de las estrellas B al subtipo "0-1" independientemente de su Teff. Las estrellas B del corpus tienen Teff 10,000–14,083 K (mediana 10,715 K), correspondiente a subtipos B4–B9 según los bins de calibración MK. El modelo escribe literalmente `"This B0-1 V star has teff_k=10415 K"` —una contradicción interna en su propio texto— lo que indica que el subtipo fue tomado del prior de entrenamiento (B0-1 son las estrellas B más citadas en la literatura) y no derivado del Teff provisto. Esto se confirmará en V6 inyectando ejemplos few-shot con B4-6 y B7-9: si el colapso desaparece, la causa era el prior; si persiste, hay una limitación más fundamental.

### 5.3 El kappa de población no es un error del LLM

El kappa de población vs. PASTEL resultó en −0.125 (peor que azar). Sin embargo, el análisis cruzado demostró que el LLM respeta el ancla de población del HC al **100%** (kappa LLM vs. HC = 1.0). El kappa negativo vs. PASTEL refleja un desacuerdo entre fuentes de ground truth, no un error del sistema:

- El HC clasifica población usando cinemática (v_tan) + química (Gaia), que es metodológicamente más completo.
- El validador deriva el ground truth de población desde PASTEL usando solo [Fe/H], ignorando la cinemática.
- Se identificaron 84 desacuerdos HC vs. PASTEL, en su mayoría en la frontera Disco Grueso / Disco Fino donde la cinemática y la química no concuerdan.

**Cómo documentarlo:** la métrica correcta de la capacidad del LLM en población es kappa LLM vs. HC = 1.0. El desacuerdo HC vs. PASTEL es una limitación del ground truth de validación, no del sistema, y debe discutirse explícitamente como trabajo futuro (incorporar v_tan de PASTEL o datos cinemáticos de alta resolución como GALAH o APOGEE para construir un ground truth de población más robusto).

### 5.4 Notable features: problema de activación, no de capacidad

El LLM escribe "None identified." en el 59.5% de las estrellas con flags activos. Sin embargo, cuando sí genera texto para notable_features, la calidad es correcta (ejemplo real: `"H-alpha emission detected (EW=−3.2 Å); consistent with chromospheric activity or Be-type phenomenon"`). El problema no es que el modelo no sepa describir flags — es que no tiene la instrucción explícita de cuándo activar esa descripción. Esto se corrige en V6 con reglas condicionales explícitas por flag.

### 5.5 La confidence invertida como limitación estructural del LLM

La confidence está invertida en todas las versiones V1–V5: el modelo reporta mayor confianza en sus predicciones incorrectas que en las correctas. La correlación Spearman entre confidence y acierto es negativa (r = −0.066) y no significativa. Este es un problema conocido de los LLMs: la confianza expresada en lenguaje natural no corresponde a la incertidumbre epistémica real del modelo. Para la tesis, esto argumenta a favor de usar el `quality_score` del HC como mecanismo de calibración externo, en lugar de confiar en los confidence scores del LLM.

---

## 6. Métricas de Evaluación — Explicación y Justificación

### 6.1 Accuracy MK y Cohen κ

La accuracy MK coarse compara la letra espectral predicha (O, B, A, F, G, K, M) contra el tipo SIMBAD. Se usa Cohen κ como métrica principal porque corrige por acuerdo al azar — con 7 clases de distribución desigual, una accuracy de 50% puede ser casi aleatoria. κ > 0.6 es acuerdo sustancial; κ > 0.8 es acuerdo casi perfecto. El near-miss accuracy (d≤1 clase en la secuencia MK) mide si los errores son "vecinos" astrofísicamente plausibles.

### 6.2 ROUGE-1 y ROUGE-L — Qué miden y qué dicen los resultados

ROUGE (*Recall-Oriented Understudy for Gisting Evaluation*) es una familia de métricas que mide solapamiento de n-gramas entre un texto generado (hipótesis) y uno de referencia escrito por humanos. Fue desarrollada originalmente para evaluación automática de resúmenes en NLP.

**ROUGE-1 F** mide solapamiento de unigramas (palabras individuales) en balance precision-recall. Captura si el modelo usa el vocabulario técnico correcto. **ROUGE-L F** mide la subsecuencia común más larga, capturando si el modelo produce frases con la misma estructura que la referencia — es más exigente porque penaliza el orden incorrecto.

En STELLAR se aplica sobre los tres subcampos de `stellar_description`, comparando contra 11 descripciones humanas de referencia:

| Subcampo | ROUGE-1 F (V5) | ROUGE-L F (V5) | Interpretación |
|---|---|---|---|
| `physical_profile` | **0.354** | 0.213 | Vocabulario astrofísico correcto, estructura parcialmente diferente |
| `population_context` | **0.258** | 0.178 | Moderado; el modelo describe la población con giros distintos |
| `notable_features` | **0.029** | 0.029 | Colapso casi total — "None identified." vs. descripciones reales |

El ROUGE de `notable_features` de 0.029 no indica que el modelo no sepa generar este texto (cuando lo genera, es correcto) sino que lo genera con frecuencia incorrecta: el modelo ignora los flags activos en 59.5% de los casos. En V6 se espera que este valor suba a niveles comparables con `physical_profile` (~0.35) al incorporar reglas condicionales explícitas.

Los ROUGE scores de STELLAR son bajos en términos absolutos comparados con benchmarks de NLP general (~0.4–0.6 para resúmenes), pero este es el comportamiento esperado en texto científico especializado: las descripciones de referencia humanas tienen alta especificidad técnica y vocabulario restringido, por lo que cualquier desviación en el giro de una frase produce una penalización significativa. Lo relevante no es el valor absoluto sino la evolución entre versiones.

### 6.3 Coherencia interna (Level 4b)

Verifica que los valores numéricos del physical_vector (Teff, logg, v_tan, [Fe/H]) aparezcan efectivamente mencionados en el texto generado. En V5: coherencia media = 0.998, lo que indica que el modelo no inventa valores numéricos. Esta métrica es un check de trazabilidad, no de calidad del razonamiento.

### 6.4 Subtype accuracy within-class (Level 4c)

Es la métrica más honesta de la capacidad del LLM en V5: dado que el HC fijó la letra, ¿asigna el LLM el bin de subtipo correcto usando Teff como referencia? El 30.5% de accuracy global indica que el LLM tiene capacidad limitada de discriminación fina dentro de cada clase espectral, con la excepción notable de la clase M (83.3%).

### 6.5 Level 5 — Análisis de frontera A/B (nuevo en V6)

Métrica diseñada específicamente para cuantificar el hallazgo de la Sección 5.1: compara la accuracy de HC, LLM y regla logg sobre las estrellas A con `near_teff_boundary=True`. Es la métrica que operacionaliza directamente la hipótesis central de la tesis.

---

## 7. Problemas Identificados y Estado de Corrección

| Problema | Severidad | Estado en V6 |
|---|---|---|
| 181/498 estrellas sin resultado (truncamiento JSON) | Alta | ✅ Corregido — `max_new_tokens` 512 → 1024 |
| Artefacto `<<SYS>>` contaminando JSON | Media | ✅ Corregido — parser robusto heredado |
| Subtype B colapsa a "0-1" (prior de entrenamiento) | Alta | 🔄 En prueba — few-shot B4-6 y B7-9 en V6 |
| Notable features ignora flags activos (59.5%) | Alta | 🔄 En prueba — reglas condicionales explícitas en V6 |
| Frontera A/B: HC fuerza A siempre (50% accuracy) | Media | 🔄 En prueba — ancla blanda logg en V6 |
| Confidence invertida | Baja (estructural) | ⚠️ No corregible por prompt — documentar como limitación |
| Kappa población vs. PASTEL negativo | Baja (ground truth) | ⚠️ No es error del sistema — documentar desacuerdo de fuentes |
| ROUGE notable_features ≈ 0 | Alta | 🔄 Derivado de flags — se espera corrección con V6 |

---

## 8. Estado Actual y Próximos Pasos

### 8.1 Lo que está completo

- Hard Computing (HC-2.0): 498/498 contratos generados, pipeline completamente validado.
- SC V1–V5: inferencia y validación completas. Resultados analizados en profundidad.
- SC V6: todos los scripts desarrollados y listos para lanzar en el clúster.

### 8.2 Scripts V6 listos para ejecución

| Script | Función |
|---|---|
| `system_prompt_v6.py` | Prompt con ancla blanda A/B, few-shot B, reglas de flags |
| `inference_manager_v6.py` | Motor de inferencia con `max_new_tokens=1024` |
| `validator_v6.py` | Validador heredado V5 + Level 5 (análisis frontera A/B) |
| `run_inference_v6.slurm` | Job SLURM de inferencia |
| `run_validation_v6.slurm` | Job SLURM de validación |

### 8.3 Secuencia de lanzamiento

```bash
# 1. Copiar scripts al clúster
scp system_prompt_v6.py inference_manager_v6.py validator_v6.py \
    run_inference_v6.slurm run_validation_v6.slurm \
    <usuario>@<cluster>:~/proyects/AstroLLM/STELLAR/Soft_Computing/scripts/

# 2. Lanzar inferencia
sbatch SLURM/run_inference_v6.slurm

# 3. Tras terminar la inferencia, reemplazar JOB_ID y lanzar validación
sed -i 's/REPLACE_WITH_V6_INFERENCE_JOB_ID/<JOB_ID>/' SLURM/run_validation_v6.slurm
sbatch SLURM/run_validation_v6.slurm
```

### 8.4 Qué se espera de V6

- Coverage: recuperar las 181 estrellas fallidas → pasar de 317 a ~470–490 resultados.
- Subtype B: si el few-shot rompe el colapso, confirma que era prior de entrenamiento.
- ROUGE notable_features: pasar de 0.029 a ~0.30–0.35 con las reglas condicionales.
- Level 5 (A/B): cuantificar si la ancla blanda mejora los 66.7% de V5.

### 8.5 Trabajo futuro (post-V6)

- Incorporar la regla `logg ≥ 3.80` al propio HC como mejora del AstrometryAgent en la frontera A/B.
- Evaluar AstroSage en versión de mayor tamaño si está disponible en el clúster.
- Construir ground truth de población con fuente cinemática (GALAH/APOGEE) para resolver el desacuerdo HC vs. PASTEL.
- Capítulo de resultados de tesis: tabla comparativa V1–V6, matrices de confusión, distribución de confidence, curvas ROUGE por versión.

---

## 9. Infraestructura

**Clúster:** Laboratorio de Supercómputo del Bajío (Lab-SB), CIMAT Guanajuato  
**Partición:** GPU | **Hardware:** NVIDIA Titan RTX 24 GB × 2 por nodo (g-0-1 a g-0-12)  
**Walltime real:** 120 horas | **Env Conda:** `prometheus`  
**Modelo:** AstroSage-8B (LLaMA-based, especializado en astrofísica)  
**Ruta modelo:** `/home/est_posgrado_cesar.aguirre/proyects/AstroLLM/AstroLlama/AstroSage-8B`

---

*Documento generado: Mayo 2026 | Próxima revisión: tras resultados V6*
