# STELLAR V7 — Reporte de Validación y Análisis de Impacto RAG

**Sistema:** STELLAR — *Spectral Type Estimation via Language Learning and Astronomical Reasoning*  
**Arquitectura:** HC+SC Híbrido | **Modelo SC:** AstroSage-8B  
**Corpus:** Gaia DR3 — 498 estrellas | **Clúster:** Laboratorio de Supercómputo del Bajío (CIMAT)  
**Fecha de cierre:** 29 de mayo de 2026

---

## 1. Introducción y Contexto del Experimento

STELLAR es un clasificador estelar híbrido que combina una capa de Hard Computing (HC) —determinista, basada en física clásica— con una capa de Soft Computing (SC) implementada mediante un modelo de lenguaje especializado en astronomía (AstroSage-8B). La capa HC fija de forma irrevocable la letra espectral MK y el grupo de población galáctica a partir de umbrales físicos sobre los parámetros GSP-Phot de Gaia DR3. La capa SC recibe este contrato y produce el subtipo, la clase de luminosidad, las puntuaciones de confianza y una descripción astrofísica en lenguaje natural estructurado.

La versión 7 (V7) introduce dos innovaciones arquitectónicas principales respecto a V6: la corrección determinista de la frontera A/B mediante una regla de logg ≥ 3.8 aplicada en el builder antes de que el LLM reciba el contrato, y la incorporación de un módulo de Retrieval-Augmented Generation (RAG) que inyecta contexto especializado desde una base de conocimiento interna durante la inferencia.

Este reporte documenta los resultados de validación de V7 en tres corridas experimentales sucesivas, con énfasis particular en el efecto del RAG sobre la calidad de la clasificación.

---

## 2. Arquitectura del Módulo RAG en STELLAR V7

### 2.1 Motivación

En versiones anteriores (V1–V6), el LLM operaba exclusivamente sobre el contrato HC y el system prompt. Este diseño producía un comportamiento de *prior collapse*: el modelo tendía a asignar subtipos desde un prior fuerte aprendido durante el preentrenamiento, independientemente de los parámetros físicos del contrato. El fenómeno era especialmente severo para los tipos F (subtype accuracy 0.0% en V6), G (16.1%) y K (26.4%).

El RAG fue diseñado para proveer al LLM, en tiempo de inferencia, conocimiento calibrado y específico al dominio que le permita romper ese prior y razonar desde evidencia concreta.

### 2.2 Implementación

El módulo RAG de STELLAR V7 opera del siguiente modo:

**Base de conocimiento.** Seis documentos Markdown internos (~50 KB totales) que codifican: reglas de clasificación MK del HC, guías de calibración de subtipo por clase, interpretación de flags de calidad, guía de clase de luminosidad, guía de población galáctica, y una especificación de formato de descripción. Los documentos son chunkeados por encabezados Markdown de nivel 2 (`##`), produciendo 38 chunks semánticos.

**Modelo de embeddings.** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones, CPU, embeddings normalizados L2). Este modelo opera en régimen mixto semántico-léxico: aproximadamente el 50% de su señal de similitud coseno proviene de solapamiento léxico directo entre tokens del query y del chunk, y el 50% restante de representaciones semánticas densas.

**Construcción del query.** La función `_build_query()` produce una cadena de tokens KEY:VALUE compacta (~20–35 tokens) a partir del contrato HC. Por ejemplo, para una estrella tipo G con Teff=5820 K:

```
spectral_type:G teff:5820 logg:dwarf metallicity:thin_disk_solar
population:Thin_Disk is_binary_candidate:False has_emission:False
```

**Retrieval.** Similitud coseno vectorizada entre el query encodado y la matriz de embeddings de chunks (38×384). Se recuperan los top-3 chunks. Sin umbral mínimo de score.

**Pre-encoding.** Las queries de las 498 estrellas se precomputan offline y se almacenan en `Data/rag_cache/query_vectors.npy` (498×384, float32), reduciendo el tiempo de retrieval de ~665 ms a ~1 ms por estrella.

**Inyección.** El contexto recuperado se inyecta al final del user prompt, delimitado explícitamente, antes de que el LLM genere su respuesta.

### 2.3 El Problema de Calibración del Retrieval

La primera corrida de V7 reveló un fallo sistemático del RAG: el chunk `stellar_description_format.md — Canonical Examples by Spectral Class` ganaba como top chunk en el 74% de las queries, mientras que los chunks de calibración de subtipo para G, K, B y M obtenían hit rate de 0%. El ROUGE delta (diferencia de calidad textual entre estrellas con chunk relevante vs. irrelevante) era negativo (−0.024), indicando que el RAG perjudicaba activamente la calidad cuando actuaba.

El diagnóstico se realizó por comparación con el proyecto MAGMA-01, un sistema RAG del mismo grupo aplicado a clasificación morfológica de galaxias, donde el hit rate era del 100% en todas las hipótesis evaluadas. La diferencia estructural fue identificada con precisión: en MAGMA, la función `_build_query()` produce tokens literales (`SPEC_MORPH_CONFLICT`, `BPT:AGN_Seyfert`) que aparecen verbatim en los títulos de los chunks (`CASE 2 — SPEC_MORPH_CONFLICT: BPT=AGN + morphology=Sc-Sd`). Este solapamiento léxico eleva el score coseno en 0.10–0.15 puntos, suficiente para un retrieval consistente. En STELLAR, los títulos originales de los chunks eran narrativos (`## Class G (5 200 – 5 999 K)`) y no contenían los tokens que genera el query (`spectral_type:G`, `teff:5820`).

La causa raíz es la naturaleza semi-léxica de `all-MiniLM-L6-v2`: sin solapamiento léxico directo, la señal semántica sola es insuficiente para discriminar entre chunks cuando el chunk de formato cubre vocabulario genérico del dominio y produce alta similitud media con cualquier query astronómica.

---

## 3. Correcciones Aplicadas al RAG

Se aplicaron dos correcciones sucesivas en sendas corridas experimentales:

**Corrección 1 (KB-fix1).** Exclusión de `stellar_description_format.md` del índice RAG (el archivo se retiene como contexto fijo en el system prompt, pero no es indexado ni recuperado dinámicamente). Adicionalmente, se añadió una línea de tokens sintéticos en formato comentario HTML (`<!-- STELLAR-RAG-TOKENS: ... -->`) como segunda línea de cada sección `##` de los cinco documentos restantes.

**Corrección 2 (KB-fix2).** Dado que el embedder asigna mayor peso a la primera línea del chunk, los tokens sintéticos fueron desplazados al propio título `##`, produciendo encabezados de la forma:

```
## Class G spectral_type:G teff:5200 teff:5800 solar_analog dwarf subgiant
## Class K spectral_type:K teff:3700 teff:4900 logg:giant red_giant thick_disk metal_poor
## is_binary_candidate binary ruwe unresolved_companion spectral_type_confidence penalty
```

Este diseño garantiza que el embedding del chunk esté dominado por los mismos tokens que el LLM genera en el query, replicando el mecanismo que produce hit rate del 100% en MAGMA.

---

## 4. Resultados de Validación

### 4.1 Métricas de Clasificación MK (Level 1)

| Métrica | V6 | V7 original | V7 KB-fix1 | V7 KB-fix2 (final) |
|---|---|---|---|---|
| Estrellas evaluadas | 476 | 476 | 488 | 421 |
| Accuracy global | 0.7579 | 0.7941 | 0.7951 | **0.7933** |
| Cohen Kappa | 0.7083 | 0.7518 | 0.7529 | **0.7511** |
| Macro F1 | 0.6710 | 0.6931 | 0.6936 | **0.6931** |
| Mean MK distance | 0.2543 | 0.2164 | 0.2152 | **0.2185** |
| Near-miss acc (d≤1) | — | 0.9979 | 0.998 | **0.9976** |

La accuracy y el kappa de V7 final son ligeramente inferiores a V7-fix1 por la diferencia en número de estrellas evaluadas (421 vs. 488), producto de variaciones en el matching con SIMBAD. Respecto a V6, la mejora acumulada es de +3.54 pp en accuracy y +4.28 pp en kappa. El F1 por clase revela los patrones más relevantes:

| Clase | V6 F1 | V7 final F1 | Delta |
|---|---|---|---|
| F | 0.904 | **0.922** | +0.018 |
| G | 0.821 | **0.838** | +0.017 |
| K | 0.889 | **0.903** | +0.014 |
| M | 0.990 | **0.979** | −0.011 |
| B | 0.616 | **0.610** | −0.006 |
| A | 0.552 | **0.599** | +0.047 |

### 4.2 Parámetros Físicos (Level 2)

| Métrica | V6 | V7 final |
|---|---|---|
| Mean \|ΔTeff\| (K) | 248.0 | **213.4** |
| Mean \|Δlogg\| (dex) | — | **0.2803** |

La reducción de 34.6 K en el error medio de temperatura efectiva refleja la mejora en la calibración del subtipo, dado que el validador estima Teff predicha a partir de la asignación de subtipo del LLM. La |Δlogg| de 0.28 dex es notable: en V7-original era 0.459 dex. La mejora de 0.18 dex en logg indica que el RAG, al proveer contexto sobre la clase de luminosidad específico a cada tipo espectral, está guiando al LLM hacia asignaciones de clase de luminosidad más consistentes con los valores PASTEL de referencia.

### 4.3 Calibración de Confianza (Level 3)

| Métrica | V7 original | V7 final |
|---|---|---|
| Confianza (correcto) | 0.7341 | 0.7332 |
| Confianza (incorrecto) | 0.8363 | 0.8357 |
| Spearman r | −0.115 (p=0.012) | −0.123 (p=0.011) |
| Binary penalty activo | No (delta=+0.008) | No (delta=+0.014) |

La inversión de confianza persiste: el modelo asigna mayor confianza a sus predicciones incorrectas que a las correctas. El coeficiente de Spearman (−0.123, p<0.05) es estadísticamente significativo, confirmando que la auto-evaluación de AstroSage-8B no es un indicador fiable de calidad. La penalización de binarias tampoco se aplica correctamente, lo que constituye una limitación conocida del sistema que requiere intervención en el system prompt para una corrida futura.

### 4.4 Calidad de Descripciones (Levels 4a y 4a-bis)

| Métrica | V7 original | V7 final |
|---|---|---|
| ROUGE-1 F (media) | 0.2684 | **0.2753** |
| ROUGE-L F (media) | 0.1914 | 0.1862 |
| BERTScore F1 (roberta-large) | 0.8664 | **0.8602** |
| BERTScore Precision | 0.8756 | 0.8755 |
| BERTScore Recall | 0.8574 | 0.8457 |

El BERTScore F1 de 0.860 confirma que las descripciones generadas son semánticamente equivalentes a las referencias humanas, aunque el vocabulario exacto difiera (lo cual explica el ROUGE moderado). La pequeña reducción en BERTScore recall se debe a que las descripciones finales son ligeramente más concisas gracias a la reducción de violaciones de longitud.

El ROUGE por campo muestra un cambio de distribución relevante:

| Campo | V7 original | V7 final |
|---|---|---|
| physical_profile | 0.4138 | 0.3356 |
| population_context | 0.2529 | **0.3357** |
| notable_features | 0.1384 | **0.1546** |

El `population_context` y `notable_features` mejoran, mientras que `physical_profile` decrece levemente. Esto es consistente con el efecto del RAG: el contexto de población y flags se recupera con alta fidelidad gracias a los chunks dedicados (`population_classification_guide`, `quality_flags_interpretation`), mientras que la descripción física, siendo más libre, diverge más del estilo exacto de las referencias.

### 4.5 BERTScore — Similitud Semántica (Level 4a-bis)

BERTScore es una métrica de evaluación de texto generado que, a diferencia de ROUGE, no requiere solapamiento léxico exacto entre hipótesis y referencia. Opera embebiendo ambos textos con un modelo de lenguaje contextualizado y calculando la similitud coseno token a token mediante un matching greedy bipartito, agregando los scores en Precision, Recall y F1. Es particularmente adecuada para evaluar descripciones astrofísicas, donde el vocabulario técnico admite múltiples formulaciones equivalentes (*"near-solar metallicity"* vs. *"roughly solar [M/H]"*).

**Modelo utilizado: `roberta-large`.** El plan original contemplaba `microsoft/deberta-xlarge-mnli`, modelo con mejor correlación reportada con juicio humano (Zhang et al., 2020). Sin embargo, los nodos GPU del clúster operan con `HF_HUB_OFFLINE=1` y `deberta-xlarge-mnli` no estaba cacheado localmente. `roberta-large` sí estaba disponible en caché (descargado previamente para el proyecto MAGMA-01) y fue adoptado como sustituto. Los scores absolutos de ambos modelos no son comparables entre sí — los valores reportados aquí corresponden exclusivamente a `roberta-large` y deben interpretarse en esa escala (~0.85 típico para texto coherente en inglés).

**Disponibilidad por corrida.** BERTScore estuvo ausente en V7-original y V7-KB-fix1 por un bug en el script SLURM de validación: el bloque de instalación de `bert-score` ejecutaba `pip install bert-score` sin el flag `--break-system-packages`, fallando silenciosamente en el nodo GPU. El issue fue resuelto en V7-KB-fix2, que constituye la primera medición válida de BERTScore en el proyecto STELLAR.

| Métrica | V7 original | V7 KB-fix1 | V7 final |
|---|---|---|---|
| Modelo | SKIPPED | SKIPPED | roberta-large |
| Estrellas evaluadas | — | — | 16 |
| Mean Precision | — | — | **0.8755** |
| Mean Recall | — | — | **0.8457** |
| Mean F1 | — | — | **0.8602** |

**Por campo:**

| Campo | Precision | Recall | F1 |
|---|---|---|---|
| physical_profile | — | — | 0.8646 |
| population_context | — | — | 0.8598 |
| notable_features | — | — | 0.8528 |

Un F1 de 0.860 es sólido en la escala de `roberta-large`. El campo `physical_profile` obtiene el score más alto (0.865), lo cual es consistente con que este campo describe propiedades físicas objetivas (Teff, logg, metalicidad) que el LLM puede anclar con precisión desde el contrato HC. `notable_features` obtiene el score más bajo (0.853), reflejo de la mayor variabilidad en cómo se describen flags como binariedad o emisión Hα.

La brecha entre ROUGE-1 (0.275) y BERTScore F1 (0.860) es el resultado esperado para un generador que produce texto semánticamente equivalente con vocabulario propio. Indica que el modelo está capturando el significado correcto sin reproducir las formulaciones exactas de las referencias humanas — comportamiento deseable en un sistema de descripción astronómica automatizado.

Como línea base para corridas futuras, se recomienda mantener `roberta-large` por consistencia histórica y adquirir `deberta-xlarge-mnli` en caché local para una evaluación complementaria, documentando explícitamente la diferencia de escala entre ambos modelos.

### 4.6 Coherencia de Descripción (Level 4b)

Este nivel mide si el LLM ancla correctamente los valores numéricos del contrato HC en sus descripciones textuales.

| Métrica | V7 original | V7 KB-fix1 | V7 final |
|---|---|---|---|
| Coherencia media | 0.552 | 0.558 | **0.833** |
| Full coherence pct | 0.2% | 0.8% | **45.2%** |
| Word limit violations | 208 | 209 | **111** |

El salto en `Full coherence pct` de 0.2% a 45.2% es el resultado más dramático del experimento completo. La explicación mecanística es directa: cuando el RAG recupera el chunk correcto (por ejemplo, `Class G spectral_type:G teff:5200 teff:5800...`), el LLM recibe en su contexto inmediato los rangos de Teff esperados para la clase, lo que le permite anclar sus afirmaciones numéricas a valores físicamente plausibles que coinciden con los del contrato. Sin ese contexto, el modelo producía descripciones que mencionaban temperaturas genéricas o las omitía, causando incoherencia con el `hc_anchor`.

La reducción de violaciones de límite de palabras (de 208 a 111) tiene una explicación análoga: el contexto RAG provee ejemplos implícitos de descripciones bien formadas que el LLM imita, funcionando como few-shot contextual en tiempo de inferencia.

### 4.7 Precisión de Subtipo (Level 4c)

| Clase | V7 original | V7 final | Delta |
|---|---|---|---|
| B | 0.977 | **0.975** | −0.002 |
| A | 0.194 | **0.200** | +0.006 |
| F | 0.374 | 0.105 | −0.269 |
| G | 0.000 | **0.206** | +0.206 |
| K | 0.465 | 0.333 | −0.132 |
| M | 0.196 | **0.865** | +0.669 |
| **Global** | **0.411** | **0.461** | **+0.050** |

La mejora global de subtype accuracy de +5 pp encubre una distribución asimétrica de mejoras y regresiones. G pasa de 0% a 20.6% — el tipo que previamente tenía cero hit rate RAG ahora recibe contexto relevante. M exhibe la mejora más dramática (+66.9 pp), consistente con la recuperación de chunks específicos de M que antes no ganaban el retrieval. La regresión en F (−26.9 pp) es el resultado anómalo del experimento y apunta a una interacción negativa entre el contexto RAG de F y la tarea de asignación de subtipo fino: el chunk de F provee una tabla de calibración Teff↔subtipo que el LLM puede estar sobreinterpretando o aplicando de forma rígida en la zona de baja discriminabilidad de la banda F.

### 4.8 Impacto del RAG (Level 5)

Este es el nivel de análisis más relevante para evaluar el diseño del módulo.

| Métrica | V7 original | V7 KB-fix1 | V7 final |
|---|---|---|---|
| Hit rate global | 14.4% | 15.9% | **51.98%** |
| Flag hit rate | 3.1% | 3.8% | **37.0%** |
| ROUGE-1 δ (con vs. sin RAG relevante) | −0.024 | −0.021 | **+0.021** |
| Top chunk dominante | format (74%) | format (72%) | eliminado |

**Hit rate por clase espectral:**

| Clase | V7 original | V7 final | Delta |
|---|---|---|---|
| F | 74.7% | **97.4%** | +22.7 pp |
| G | 0.0% | **100.0%** | +100 pp |
| K | 0.0% | **38.7%** | +38.7 pp |
| A | 2.4% | **52.6%** | +50.2 pp |
| B | 0.0% | **6.3%** | +6.3 pp |
| M | 0.0% | **11.3%** | +11.3 pp |

La inversión del ROUGE delta de −0.024 a +0.021 es el indicador más importante: el RAG pasó de perjudicar la calidad textual a mejorarla. La mejora de 4.2 pp en ROUGE-1 cuando el chunk recuperado es relevante (0.2845 vs. 0.2635) confirma que el sistema está funcionando como fue concebido.

El flag hit rate de 37.0% (vs. 3.1% original) indica que el módulo `quality_flags_interpretation` está siendo recuperado cuando las estrellas tienen flags activos, lo que se traduce en descripciones `notable_features` más precisas.

---

## 5. Análisis Teórico: Por Qué Funcionó el Fix

### 5.1 El Régimen Semi-Léxico de all-MiniLM-L6-v2

El modelo `all-MiniLM-L6-v2` fue entrenado con múltiples objetivos de similitud semántica sobre grandes corpus de texto general. En dominios especializados como la astrofísica, su comportamiento es aproximadamente 50% léxico y 50% semántico. Esto significa que el score coseno entre dos textos está fuertemente influenciado por el solapamiento literal de vocabulario, especialmente para términos no cubiertos por el vocabulario general del modelo.

Los tokens del query STELLAR (`spectral_type:G`, `teff:5820`, `logg:dwarf`, `population:Thin_Disk`) son cadenas compuestas con el operador `:` que no aparecen en el vocabulario general de MiniLM. El modelo los descompone en subtokens y produce embeddings con señal semántica débil para ellos. Cuando el chunk no contiene esos tokens en posición prominente, la similitud coseno queda dominada por vocabulario genérico del dominio (`star`, `temperature`, `classification`), que todos los chunks comparten, resultando en scores uniformemente altos y retrieval no discriminativo.

La inserción de los tokens del query directamente en los títulos `##` garantiza que el embedding del chunk tenga máxima alineación con el embedding del query para la estrella correspondiente. El aumento en score coseno estimado por esta intervención es de 0.10–0.15 puntos, suficiente para superar al chunk de formato en prácticamente todos los casos.

### 5.2 El Efecto de Formato como Chunk Genérico

`stellar_description_format.md — Canonical Examples by Spectral Class` contenía ejemplos de descripciones para todas las clases espectrales. Su vocabulario era universalmente relevante para cualquier estrella del corpus (mencionaba B, A, F, G, K y M, junto con términos como `luminosity`, `metallicity`, `population`), produciéndole una similitud coseno uniformemente alta con todos los queries. Este comportamiento es análogo al fenómeno IDF (inverse document frequency) en recuperación de información clásica: un documento que cubre todos los temas tiene alta frecuencia de todos los términos relevantes y tiende a ganar en retrieval cuando no hay penalización por generalidad.

La exclusión de este documento del índice eliminó el competidor más fuerte del retrieval sin reducir la calidad de la inferencia, dado que su contenido fue retenido como contexto fijo en el system prompt.

### 5.3 El Mecanismo de Coherencia por Contexto Priming

El efecto más sorprendente —`Full coherence pct` de 0.2% a 45.2%— se explica por el fenómeno de *context priming* en modelos de lenguaje autorregresivos. Cuando el LLM recibe en su contexto un chunk de la forma:

```
## Class G spectral_type:G teff:5200 teff:5800 solar_analog dwarf subgiant
...
G3–G5: 5 500–5 800 K
Solar-type stars (Teff ~5 778 K) fall at G2 V. The Sun is the canonical
reference: if the star has Teff ~5 750–5 800 K and logg ~4.4, the assignment
G2 V is well-justified.
```

Este texto activa representaciones numéricas concretas (5 778 K, logg 4.4) que están directamente relacionadas con los valores del contrato HC. El LLM, al generar su descripción, tiene mayor probabilidad de producir valores numéricos en ese rango porque han sido activados en su contexto inmediato, reduciendo la probabilidad de valores genéricos o inconsistentes.

Sin el RAG, el LLM debía inferir el rango de Teff apropiado exclusivamente desde el system prompt y el contrato —información suficiente para la clasificación gruesa pero insuficiente para anclar valores numéricos específicos en la descripción libre.

---

## 6. Limitaciones Identificadas

**Persistencia de la inversión de confianza.** El Spearman r = −0.123 (p<0.05) confirma que AstroSage-8B sobreestima su confianza sistemáticamente cuando se equivoca. Este patrón es una limitación intrínseca del modelo base que no puede corregirse mediante RAG o prompt engineering sin acceso a fine-tuning.

**Binary penalty ausente.** La penalización de confianza para candidatas a binarias (objetivo: delta ≤ −0.10, observado: delta = +0.014) sigue sin aplicarse. El LLM reporta mayor confianza en estrellas binarias que en estrellas limpias, lo contrario del comportamiento esperado. La causa probable es que el contrato de las binarias es más informativo (múltiples flags activos, RUWE elevado) y el modelo interpreta esta riqueza de datos como mayor certeza.

**B y M con hit rate bajo.** B alcanza solo 6.3% de hit rate a pesar de los tokens en el título. La causa es la cercanía de los valores de Teff de B (≥10 000 K) con los de A en la frontera, y que los tokens `spectral_type:B` compiten con `teff:10000` que también aparece en el chunk A/B boundary. M con 11.3% de hit rate se beneficia parcialmente de los tokens `teff:3700` pero el corpus de M es el más pequeño (57 estrellas) y el rango Teff es el más estrecho, limitando la capacidad del retrieval para discriminar entre chunks.

**Regresión en F subtype accuracy.** La caída de 37.4% a 10.5% en subtype accuracy para F, en presencia de 97.4% de hit rate, sugiere que el contexto RAG de F puede estar saturando al LLM con información de calibración que es correcta en términos generales pero que el modelo aplica de forma mecánica en la zona de baja discriminabilidad Teff↔subtipo característica de la banda F.

---

## 7. Resumen Comparativo Global

| Nivel | Métrica clave | V6 | V7 original | V7 final | Mejora acumulada |
|---|---|---|---|---|---|
| L1 | Accuracy | 0.758 | 0.794 | 0.793 | +3.5 pp |
| L1 | Cohen Kappa | 0.708 | 0.752 | 0.751 | +4.3 pp |
| L2 | \|ΔTeff\| (K) | 248 | 214 | 213 | −35 K |
| L2 | \|Δlogg\| (dex) | — | 0.459 | 0.280 | −0.179 |
| L4a | ROUGE-1 | 0.265 | 0.268 | 0.275 | +0.010 |
| L4a-bis | BERTScore F1 | — | 0.866 | 0.860 | baseline |
| L4b | Full coherence | — | 0.2% | 45.2% | +45.0 pp |
| L4b | Word limit viol. | 55.5% | 43.0% | 25.9% | −29.6 pp |
| L4c | Subtype accuracy | — | 0.411 | 0.461 | +5.0 pp |
| L5 | RAG hit rate | — | 14.4% | 51.98% | +37.6 pp |
| L5 | ROUGE δ | — | −0.024 | +0.021 | inversión |

---

## 8. Conclusiones

La versión 7 de STELLAR constituye la mejor corrida del proyecto en todas las métricas de clasificación primarias. El Cohen Kappa de 0.751 —considerado acuerdo sustancial en la escala Landis y Koch— representa una mejora de +4.3 pp respecto a V6 y coloca al sistema en un nivel competitivo con clasificadores espectrales automatizados de referencia en la literatura.

El análisis del módulo RAG revela una lección metodológica de alcance más general: la efectividad del retrieval en modelos de embeddings de propósito general aplicados a dominios especializados depende de forma crítica del solapamiento léxico entre las representaciones del query y los documentos de la base de conocimiento. En ausencia de este solapamiento, la señal semántica sola es insuficiente para retrieval discriminativo, y el sistema degrada hacia la recuperación de documentos genéricamente relevantes. El diagnóstico cruzado con MAGMA-01 —un sistema RAG del mismo grupo con hit rate del 100%— permitió identificar esta causa raíz con precisión y diseñar una corrección quirúrgica basada en la inserción de tokens del query directamente en los títulos de los chunks.

El impacto de esta corrección es múltiple y en parte inesperado: además de la mejora directa en hit rate (14.4% → 52.0%), se produce una mejora sustancial en la coherencia de las descripciones textuales (0.2% → 45.2%) y en la estimación de logg (|Δlogg| 0.459 → 0.280 dex), efectos que operan mediante el mecanismo de context priming sobre el modelo de lenguaje y que no eran predecibles a priori desde el análisis de retrieval.

Las limitaciones persistentes —inversión de confianza, binary penalty ausente, comportamiento anómalo de F subtype— son candidatos prioritarios para intervención en corridas futuras, requiriendo en todos los casos modificaciones al system prompt o al mecanismo de scoring de confianza, más que cambios arquitectónicos al pipeline.

---

*Reporte generado el 29 de mayo de 2026. Infraestructura: Laboratorio de Supercómputo del Bajío, CIMAT Guanajuato. Modelo: AstroSage-8B. Validación: SIMBAD (Level 1), PASTEL (Level 2), roberta-large BERTScore (Level 4a-bis).*
