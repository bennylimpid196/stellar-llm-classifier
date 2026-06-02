# CLAUDE.md — Reporte Técnico de Estancia 2026: STELLAR

## 1. Propósito

Contexto maestro para escribir el reporte técnico final de la estancia.
Lee este archivo completo antes de escribir cualquier línea.
NO inventes datos — todo debe estar respaldado por los archivos fuente listados en §3.

---

## 2. Identidad del Proyecto

**Nombre:** STELLAR — *Spectral Type Estimation via Language Learning and Astronomical Reasoning*

**Descripción:** Clasificador estelar híbrido que combina Hard Computing (HC)
determinista con el LLM AstroSage-8B, operando sobre 498 estrellas de Gaia DR3.

**Arquitectura:**

- HC (pipeline_version HC-2.0): 4 agentes deterministas → contrato JSON por estrella
- SC: AstroSage-8B recibe el contrato → clasificación MK + descripción en lenguaje natural

**Resultado final validado (V6):**

- Accuracy: 0.7579 | Cohen Kappa: 0.7083 | Macro F1: 0.6710
- Near-miss (d≤1): 0.9976 | |ΔTeff| vs PASTEL: 248.0 K
- Bootstrap CI 95% sobre error: [0.135, 0.205]

---

## 3. Mapa de Archivos Fuente

### Especificación y contexto — Context/

| Archivo                                                                                                       | Alimenta      |
| ------------------------------------------------------------------------------------------------------------- | ------------- |
| `Context/01. Data Dictionary and Gaia DR3 Sample.md`                                                        | §3.2 Dataset |
| `Context/02. Coding Standards and Tech Stack Guidelines.md`                                                 | §5.1 Código |
| `Context/03. AstrometryAgent Specification (HC Module).md`                                                  | §3.1 HC      |
| `Context/04. ContinuumAgent Specification (HC Module).md`                                                   | §3.1 HC      |
| `Context/05. LineAgent Specification (HC Module).md`                                                        | §3.1 HC      |
| `Context/06. BinaryDetectorAgent Specification (HC Module).md`                                              | §3.1 HC      |
| `Context/Especificación Técnica del Proyecto - Clasificador Estelar Híbrido HC+SC (Versión Sólida).md` | §2.1, §2.2  |
| `Context/HC to SC Handoff.md`                                                                               | §3.1, §3.2  |

### Progreso y resultados — cluster/

| Archivo                                                     | Alimenta                    |
| ----------------------------------------------------------- | --------------------------- |
| `cluster/Data/hc_contracts.json`                          | §3.1 ejemplos de contratos |
| `cluster/Data/ground_truth_final.csv`                     | §3.2 ground truth          |
| `cluster/knowledge_base/*.md`                             | §2.2 Marco teórico        |
| `cluster/outputs/validation_v6/validation_report_v6.json` | §4.1 Resultados V6         |
| `cluster/outputs/validation_v7/validation_report_v7.json` | §4.1 Resultados V7         |
| `cluster/outputs/validation_v7/bertscore_v7.json`         | §4.1 BERTScore             |
| `cluster/outputs/validation_v7/rag_impact_v7.json`        | §4.1 impacto RAG           |
| `cluster/scripts/system_prompt_v7.py`                     | §3.1 SC arquitectura       |
| `cluster/scripts/validator_v7.py`                         | §3.1 métricas             |
| `cluster/README_cluster.md`                               | §5.1 infraestructura       |

---

## 4. Estructura del Reporte


* Introducción
  1.1 Antecedentes
  1.2 Planteamiento del Problema
  1.3 Justificación
* Solución Propuesta
  2.1 Objetivos
  2.2 Marco Teórico y Metodología
* Desarrollo
  3.1 Modelos y Algoritmos
  3.2 Conjuntos de Datos
* Resultados
  4.1 Resultados Experimentales
  4.2 Conclusiones y Recomendaciones
* Anexos
  5.1 Código

## 5. Checklist ML Reproducibility v2.0

Para cada sección verificar:

- [ ] Descripción matemática de cada agente HC

- [ ] Supuestos explícitos (convención EW, umbral RUWE, etc.)

- [ ] n=498 estrellas, splits de validación documentados

- [ ] Preprocesamiento: NaN, fe_h==0.0 como artefacto, convención ESP-ELS

- [ ] Dataset link: https://gea.esac.esa.int/archive/

- [ ] Hiperparámetros: versiones V1–V7 del system prompt

- [ ] Número de runs: 7 versiones × 498 estrellas

- [ ] Métricas con varianza: Bootstrap CI 95% = [0.135, 0.205]

- [ ] Infraestructura: Lab-SB CIMAT · GPU · Titan RTX 24GB ×2 · walltime 120h



---



## 6. Tecnología



- LaTeX, español, compilar con `pdflatex`

- Tablas: booktabs únicamente (`\toprule \midrule \bottomrule`)

- Archivos: `reporte/main.tex`, secciones en `reporte/secciones/`

- Marcar datos no encontrados como `% TODO:`

- Nunca usar `\hline`



---



## 7. Instrucciones Operativas



1. Lee los archivos fuente relevantes ANTES de escribir cada sección

2. Una sección a la vez — espera confirmación antes de continuar

3. Al terminar cada sección lista: ítems del checklist cubiertos y pendientes

4. Primer paso: lee `cluster/outputs/validation_v7/validation_report_v7.json`

   y `cluster/outputs/validation_v7/validation_summary_v7.txt` para tener

   las métricas más actualizadas, luego comienza con §1.1 Antecedentes



---



## 8. Datos Clave



- Corpus: 498 estrellas · Gaia DR3 · `cluster/Data/catalog.csv`

- HC: 4 agentes · HC-2.0 · 498/498 éxito

- SC: AstroSage-8B · 7 versiones de prompt (V1–V7)

- V6: Acc=0.7579 · κ=0.7083 · F1=0.6710 · near-miss=0.9976 · |ΔTeff|=248K

- Confusión principal: B↔A (frontera Teff≈10 000 K)

- Hallazgo clave: prior fuerte hacia G en LLM → anclaje HC lo corrige

- Clúster: Lab-SB CIMAT · `est_posgrado_cesar.aguirre@148.207.185.31` · puerto 2284

- Conda env: `prometheus` · Modelo: `AstroLlama/AstroSage-8B`



---



## 9. Bitácora de Progreso (última actualización: 2026-06-01)



### 9.1 Estado global

Esqueleto LaTeX montado y funcional. **§1, §2 y §3 redactadas y compilando**
(`pdflatex` + `bibtex`, ~18 páginas, sin errores). Paramos **antes de empezar §4
Resultados**.



### 9.2 Infraestructura del reporte (hecho)

- `reporte/main.tex` — preámbulo completo: babel español, `booktabs`, `siunitx`,
  `natbib` (estilo `plainnat`), `listings` (para anexo de código), `hyperref` +
  `cleveref`, macros del proyecto (`\teff`, `\logg`, `\MG`, `\STELLAR`,
  `\AstroSage`). Incluye `\input` activos de §1, §2, §3; §4 y §5 comentados.

- `reporte/referencias.bib` — 5 entradas **verificadas**: `thompson1987`,

  `lindegren2021`, `wangchen2019`, `gaiadr3`, `pastel2016`.

- Compilar con: `cd reporte && pdflatex main && bibtex main && pdflatex main && pdflatex main`.



### 9.3 Secciones redactadas (hecho)

- **§1 Introducción** (`secciones/01_introduccion.tex`): 1.1 Antecedentes,

  1.2 Planteamiento (incluye 4 preguntas de investigación), 1.3 Justificación.

- **§2 Solución** (`secciones/02_solucion.tex`): 2.1 Objetivos (general + 6

  específicos), 2.2 Marco Teórico y Metodología (fundamentos MK/Teff,

  luminosidad/logg, poblaciones, M_G/extinción/V_tan, líneas Ca II/Hα; +

  metodología: HC vs SC, arquitectura, anclaje `hc_anchor`, V1–V7, validación

  multinivel). Tablas: umbrales Teff→letra, logg→luminosidad.

- **§3 Desarrollo** (`secciones/03_desarrollo.tex`): 3.1 Modelos y Algoritmos

  (los 4 agentes con su matemática: A_G, M_G, V_tan, s=Nσ², Voigt + FWHM

  Thompson, RUWE/RV adaptativo/NSS; contrato HC-2.0 + quality_score; capa SC

  AstroSage-8B, anclaje duro, calibración de confianza, RAG; tabla evolución

  V1–V7). 3.2 Conjuntos de Datos (origen Gaia DR3, preprocesamiento/convenciones,

  distribución por letra y población, banderas, quality_score, cobertura ground

  truth SIMBAD/PASTEL). **Todas las cifras de §3.2 calculadas directo de los

  archivos fuente**, no inventadas.



### 9.4 Dónde nos quedamos / siguiente paso

**Continuar con §4.1 Resultados Experimentales.** Plan acordado: leer

`validation_report_v6.json`, `bertscore_v7.json`, `rag_impact_v7.json`; comparar

V6 ↔ V7; construir tablas de métricas por nivel (1–5), matriz de confusión,

near-miss, ΔTeff/Δlogg vs PASTEL, subtipo por clase, calibración de confianza e

impacto RAG. Métricas V7 ya leídas (en `validation_report_v7.json`):

Acc=0.7951 · κ=0.7529 · MacroF1=0.6936 · near-miss=0.998 · |ΔTeff|=212.2K ·

|Δlogg|=0.46 · BERTScore F1=0.866. OJO: el CI bootstrap reportado en el JSON

`[0.1413, 0.2049]` es sobre el **error**, no sobre accuracy.



### 9.5 Pendientes (TODO) acumulados

- **§4.1, §4.2 (Resultados y Conclusiones)** — no iniciadas.

- **§5.1 Anexos: Código** — no iniciada.

- **Resumen/abstract** — placeholder vacío en `main.tex`, redactar al final.

- **Figura del pipeline** conceptual HC→contrato→SC (`figuras/pipeline.*`).

- **Ejemplos de contratos JSON** (entrada HC y salida SC) — el usuario los pidió

  explícitamente como inserción en §3.1 (comentario en `03_desarrollo.tex` tras

  el quality_score). Fuente: `cluster/Data/hc_contracts.json` y ejemplos del

  `HC to SC Handoff.md` §6.

- **Citas faltantes en `referencias.bib`**: AstroSage-8B (clave `astrosage`,

  confirmar paper/model card), sistema MK (Morgan, Keenan & Kellman 1943).

- **Forward refs** `\ref{sec:resultados}` (5, muestran "??") se resolverán al

  crear §4 con `\label{sec:resultados}`.

- El usuario está **editando manualmente** los .tex (ajustes de estilo/prosa);

  respetar esos cambios, no revertirlos.
