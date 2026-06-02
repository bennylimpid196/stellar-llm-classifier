# Handoff Document: HC → SC Layer

## Clasificador Estelar Híbrido HC+SC — Pipeline Version HC-2.0

**Fecha de corte:** 2026-04-28 | **Preparado para:** Implementación del módulo Soft Computing (AstroSage-Llama)

---

## 1. Resumen Ejecutivo del Proyecto

Sistema de clasificación espectral estelar automático, explicable y de alta precisión sobre datos Gaia DR3. La arquitectura es de dos capas:

- **Hard Computing (HC):** algoritmos deterministas que extraen parámetros físicos limpios y generan banderas lógicas ("predigestión") a partir de datos crudos de Gaia.
- **Soft Computing (SC):** inferencia mediante **AstroSage-Llama**, un LLM especializado que recibe los datos pre-digeridos del HC y emite un diagnóstico astrofísico final en formato JSON estructurado.

El HC está **completamente implementado y validado**.

---

## 2. Estado del Dataset (DB-1-2026-04-28)

### 2.1 Catálogo Tabular

**Archivo:** `DB-1-2026-04-28/catalog.csv`  
**Estrellas:** 498  
**Columnas:** 22 (20 requeridas por especificación + 2 flags de disponibilidad espectral)

|Columna|Tipo|Unidades|Notas|
|---|---|---|---|
|`source_id`|int64|—|ID único Gaia DR3. No castear a float.|
|`ra` / `dec`|float|deg|Coordenadas ecuatoriales|
|`parallax`|float|mas|Puede ser ≤ 0 para fuentes lejanas/ruidosas|
|`parallax_error`|float|mas|SNR = parallax/parallax_error|
|`pmra` / `pmdec`|float|mas/yr|`pmra` ya corregido por cos(δ) en DR3|
|`ruwe`|float|—|> 1.4 indica mala solución astrométrica o binariedad|
|`phot_g_mean_mag`|float|mag|Magnitud aparente banda G|
|`ebpminrp_gspphot`|float|mag|Extinción E(BP-RP). Altamente propenso a NaN|
|`teff_gspphot`|float|K|Temperatura efectiva GSP-Phot|
|`logg_gspphot`|float|dex|Gravedad superficial. Pasa directo al SC|
|`mh_gspphot`|float|dex|Metalicidad global [M/H]|
|`alphafe_gspspec`|float|dex|[α/Fe]. Sparse en DR3 (~53% cobertura)|
|`fem_gspspec`|float|dex|[Fe/M] espectroscópico. Sparse (~44% cobertura)|
|`ew_espels_halpha`|float|Å|EW H-alpha ESP-ELS. **Convención: negativo = emisión, positivo = absorción**|
|`ew_espels_halpha_flag`|int|—|Usar solo si flag == 0|
|`radial_velocity_error`|float|km/s|Puede ser NaN sin datos RVS|
|`rv_nb_transits`|float|count|Llega como float, castear a int defensivamente|
|`nss_solution_type`|string|—|Puede ser `""`, `"None"`, o NaN|
|`has_rvs_spectrum`|bool|—|True si existe archivo .npz en carpeta `rvs/`|
|`has_bprp_spectrum`|bool|—|True si la estrella está en `spectra_bprp_ids.npy`|

### 2.2 Espectros RVS por estrella (Radial Velocity Spectra)

**Carpeta:** `DB-1-2026-04-28/rvs/`  
**Formato:** Un archivo `.npz` por estrella, nombrado por `source_id`  
**~430 archivos disponibles**

```python
import numpy as np
rvs = np.load("rvs/1244571953471006720.npz")
# Keys: ['wavelength_nm', 'flux', 'flux_error']
# wavelength_nm: shape=(2401,), rango 846.0–870.0 nm, dtype=float64
# flux:          shape=(2401,), dtype=float32  ← puede ser todo NaN
# flux_error:    shape=(2401,), dtype=float32
```

> **Importante:** Algunos archivos RVS tienen `flux` completamente NaN (espectros sin datos válidos). Siempre verificar `np.isnan(flux).all()` antes de procesar. Caer al BP/RP como fallback.

**Ventana espectral RVS:** 846–870 nm — cubre el **Ca II Triplete** (849.8, 854.2, 866.2 nm), altamente sensible a logg y metalicidad.

### 2.3 Espectros BP/RP (matriz global)

**Archivos:**

- `spectra_bprp.npy` — shape `(498, 343)`, flux en unidades absolutas, dtype float64
- `spectra_bprp_ids.npy` — shape `(498,)`, source_ids alineados con filas de flux
- `sampling_bprp.npy` — shape `(343,)`, longitudes de onda 336–1020 nm (paso 2 nm)

```python
ids  = np.load("spectra_bprp_ids.npy")
flux = np.load("spectra_bprp.npy")
wave = np.load("sampling_bprp.npy")

# Extraer espectro de una estrella por source_id:
idx = np.where(ids == 1244571953471006720)[0][0]
star_flux = flux[idx]  # array de 343 valores
```

> **Nota de resolución:** el paso de 2 nm/pixel del BP/RP es insuficiente para ajustes Voigt confiables de líneas individuales estrechas. Se usa únicamente para normalización del continuo (ContinuumAgent).

### 2.4 Ground Truth de Validación

**Archivo:** `DB-1-2026-04-28/validation/ground_truth_clean.csv`  
**Estrellas totales:** 498

|Parámetro|Cobertura|Fuente|
|---|---|---|
|`sp_type` (tipo MK)|491/498 (98.6%)|SIMBAD|
|`teff_pastel`|340/498 (68.3%)|PASTEL (mediana multi-estudio)|
|`logg_pastel`|194/498 (39.0%)|PASTEL|
|`feh_pastel`|183/498 (36.7%)|PASTEL|
|`n_pastel_measurements`|340/498|PASTEL (nº de estudios por estrella)|
|`source_id`|498/498|Extraído de `user_specified_id` (Gaia DR3)|

**Rango Teff validado:** 3938–17360 K (mediana 6322 K)  
**Sin outliers físicos:** Verificado con bounds [2000, 100000] K para Teff, [-2.0, 7.5] para logg, [-5.5, 2.5] para [Fe/H].

**Pipeline de construcción del ground truth:**

1. `validation_data_fetcher_v2.py` → resolución SIMBAD de source_ids a nombres canónicos
2. `join_pastel_local.py` → join local con `pastel.dat.gz` (archivo descargado manualmente de CDS), dos pasadas: nombre directo + alias HD/HIP
3. `ground_truth_cleaner.py` → recuperación de source_id desde `user_specified_id`, filtro de outliers físicos

---

## 3. Arquitectura HC — Módulos Implementados

### 3.1 AstrometryAgent

**Función:** Calcula magnitud absoluta M_G y velocidad tangencial V_tan a partir de astrometría Gaia.

**Fórmulas:**

```
A_G = 2.74 × E(BP-RP)          [extinción, Wang & Chen 2019]
M_G = G + 5 + 5·log10(π/1000) - A_G
V_tan = 4.74 × √(μ_α*² + μ_δ²) / π   [km/s]
```

**Guards implementados:**

1. Paralaje negativo/cero → retorna `status: "partial"`, suprime M_G y V_tan
2. `ebpminrp_gspphot` NaN → A_G = 0.0 con warning (no propaga NaN)
3. Paralaje SNR < 5 → M_G y V_tan suprimidos del contrato (evita valores absurdos)

**Nota:** `pmra` en Gaia DR3 ya es μ_α* (corregido por cos δ). No re-aplicar el coseno.

### 3.2 BinaryDetectorAgent

**Función:** Detecta sistemas múltiples no resueltos mediante tres indicadores independientes.

**Criterios:**

- `is_astrometric_binary`: RUWE > 1.4 (Lindegren et al. 2021)
- `is_rv_variable`: rv_nb_transits ≥ 5 AND rv_error > umbral adaptativo por Teff:
    - Teff < 4000 K: umbral 2.0 km/s (muchas líneas nítidas)
    - 4000 ≤ Teff < 7000 K: umbral 5.0 km/s (estándar)
    - Teff ≥ 7000 K: umbral 10.0 km/s (pocas líneas anchas, evita falsos positivos)
- `is_confirmed_nss`: `nss_solution_type` contiene string válido (ej. "Orbital", "AstroSpectroSB1")
- `is_binary_candidate`: OR de los tres anteriores (bandera maestra)

**Caveat documentado:** en estrellas M activas (Teff < 4000 K), el jitter cromosférico puede producir variabilidad RV de 1-3 km/s sin binariedad. `is_rv_variable` solo debe interpretarse junto con `is_astrometric_binary` para esta población.

### 3.3 ContinuumAgent

**Función:** normaliza el espectro dividiendo por el pseudo-continuo estimado mediante splines cúbicos con sigma-clipping iterativo.

**Modos de operación:**

- **Modo RVS** (`rvs_mode=True`): restringe el ajuste a ventanas seguras [847–849, 851–853.5, 856–860, 869.5–874 nm] para proteger las alas del Ca II triplete en estrellas frías K/M.
- **Modo BP/RP** (`rvs_mode=False`): qjuste global sobre toda la ventana 336–1020 nm.

**Smoothing dinámico:** s = N × σ² (adapta la rigidez del spline al SNR local)

**Universal Guard:** si la máscara final del continuo tiene < 10 puntos, el ajuste se marca como `fit_diverged=True` y se descarta.

**Flags de salida:**

- `continuum_is_stable`: std(flux normalizado en máscara) < 0.05
- `high_snr_continuum`: SNR > 20
- `fit_diverged`: ajuste inválido o insuficientes puntos

### 3.4 LineAgent

**Función:** ajusta perfiles de Voigt sobre el Ca II Triplete (ventana RVS) para extraer EW, FWHM y desplazamiento Doppler.

**Líneas objetivo:** 849.802 nm, 854.209 nm, 866.214 nm (Ca II triplete RVS)

**Prior instrumental:** FWHM_G mínimo = 0.075 nm (resolución RVS, R~11500 a 860 nm). El ajuste no puede producir líneas más estrechas que el instrumento.

**Límites físicos (bounds):**

- amplitude_L: (-0.5, 1.2)
- fwhm_G: (0.075 nm, sin límite superior)
- fwhm_L: (0.0, sin límite superior)
- x_0: (λ_rest ± 1.5 nm)

**FWHM total** (Thompson et al. 1987): `FWHM = 0.5346·f_L + √(0.2166·f_L² + f_G²)`

**Nota sobre H-alpha:** H-alpha (656.3 nm) está fuera de la ventana RVS (846–870 nm) y la resolución BP/RP (2 nm/pixel) es insuficiente para ajuste Voigt. **H-alpha se toma exclusivamente del catálogo `ew_espels_halpha` (pipeline ESP-ELS de Gaia)**, que provee un ajuste de alta calidad sobre los espectros RVS completos de Gaia. Esto es correcto por diseño — no es un TODO pendiente. La bandera `has_emission_source: "catalog"` documenta esta decisión.

**Convención de signo ESP-ELS (crítica):**

```
ew_espels_halpha > 0  →  ABSORCIÓN  (estrella normal)
ew_espels_halpha < 0  →  EMISIÓN    (actividad cromosférica, Be, etc.)
has_emission = True   ←  ew_espels_halpha < 0 AND flag == 0
```

---

## 4. Contratos HC — Formato de Output

### 4.1 Schema Completo (pipeline_version: HC-2.0)

```json
{
  "source_id": "string (int64 como string)",
  "pipeline_version": "HC-2.0",
  "astrometry_status": "success | partial | failed",
  "binary_status": "success | failed",
  "physical_vector": {
    "abs_mag": "float | null (M_G, suprimido si paralaje no confiable)",
    "teff_k": "int (K, siempre presente)",
    "metallicity": "float ([M/H] de mh_gspphot, siempre presente)",
    "fe_h": "float | null ([Fe/H] de fem_gspspec, ~44% cobertura)",
    "alpha_fe": "float | null ([α/Fe], ~53% cobertura)",
    "logg": "float (log g, siempre presente)",
    "v_tan": "float | null (km/s, suprimido si paralaje no confiable)",
    "extinction_ag": "float (A_G en mag, 0.0 si ebpminrp es NaN)"
  },
  "logical_flags": {
    "is_reliable_parallax": "boolean (parallax/parallax_error > 5)",
    "is_giant": "boolean (M_G < 3 AND Teff < 7000 K AND is_reliable_parallax)",
    "is_metal_poor": "boolean ([M/H] < -1.0)",
    "is_binary_candidate": "boolean (RUWE > 1.4 OR RV variable OR NSS confirmado)",
    "is_high_velocity": "boolean (V_tan > 200 km/s AND is_reliable_parallax)",
    "has_emission": "boolean (ew_espels_halpha < 0 AND flag == 0)",
    "has_emission_source": "string: 'catalog' | 'none'"
  },
  "binary_diagnostics": {
    "ruwe": "float",
    "rv_error_km_s": "float | null",
    "rv_nb_transits": "int",
    "nss_solution": "string | null",
    "adaptive_rv_threshold": "float (km/s, depende de Teff)"
  },
  "spectral_summary": {
    "halpha_catalog": {
      "ew_aa": "float (Å, positivo=absorción, negativo=emisión)",
      "flag": "int (0 = confiable)"
    },
    "cat_triplet": [
      {
        "line_nm": "float (849.802 | 854.209 | 866.214)",
        "ew_aa": "float | NaN",
        "fwhm_nm": "float | NaN",
        "status": "success | failed",
        "high_quality_fit": "boolean"
      }
    ],
    "bprp_continuum": {
      "status": "success | failed",
      "snr": "float",
      "stable": "boolean",
      "high_snr": "boolean",
      "fit_diverged": "boolean"
    },
    "rvs_continuum": {
      "status": "success | failed",
      "snr": "float",
      "stable": "boolean",
      "high_snr": "boolean",
      "fit_diverged": "boolean"
    }
  },
  "quality_score": "float [0.0 – 1.0]",
  "processing_timestamp": "ISO 8601"
}
```

### 4.2 Lógica del quality_score

```python
score = 1.0
# Factor 1 — Paralaje (crítico: anula todo si falla)
if not is_reliable_parallax: score *= 0.0
# Factor 2 — Continuo BP/RP
if bprp_status != "success":  score *= 0.3
elif not bprp_high_snr:       score *= 0.5
# Factor 3 — Continuo RVS
if rvs_status != "success":   score *= 0.7
elif not rvs_stable:          score *= 0.85
# Bonus — calidad espectral excepcional
if bprp_high_snr AND rvs_high_snr AND rvs_stable:
    score = min(1.0, score * 1.1)
```

**Distribución en la corrida actual:**

|Umbral|N|%|
|---|---|---|
|≥ 0.90|219|44.0%|
|≥ 0.80|394|79.1%|
|≥ 0.50|412|82.7%|
|< 0.50|86|17.3%|

---

## 5. Estadísticas de la Corrida HC (DB-1-2026-04-28)

**Contratos generados:** 498 / 498 (100%)  
**AstrometryAgent:** 498 success  
**BinaryDetectorAgent:** 498 success

### Distribución de Flags Lógicos

|Flag|True|%|
|---|---|---|
|`is_reliable_parallax`|496|99.6%|
|`is_giant`|187|37.6%|
|`is_metal_poor`|89|17.9%|
|`is_binary_candidate`|79|15.9%|
|`has_emission`|28|5.6%|
|`is_high_velocity`|1|0.2%|

### Calidad Espectral

|Métrica|Valor|
|---|---|
|CaT: líneas ajustadas exitosamente|888/1494 (59.4%)|
|BP/RP continuum exitoso|422/498|
|BP/RP SNR mediana|30.5|
|BP/RP SNR rango|3.7 – 57.5|
|RVS continuum exitoso|498/498|
|RVS SNR mediana|42.8|
|RVS SNR rango|0.0 – 416.7|

> El RVS SNR mínimo de 0.0 corresponde a espectros con flux totalmente NaN. El pipeline los procesa sin crash, y el quality_score los penaliza correctamente.

---

## 6. Ejemplos de Contratos HC Representativos

### 6.1 Estrella Normal (Main Sequence, referencia limpia)

**source_id:** `1244571953471006720` (*tau Boo — F7IV-V)

```json
{
  "physical_vector": {
    "abs_mag": 3.3885, "teff_k": 6319, "metallicity": -0.0794,
    "fe_h": 0.0, "alpha_fe": 0.3, "logg": 4.1688,
    "v_tan": 35.0205, "extinction_ag": 0.0
  },
  "logical_flags": {
    "is_reliable_parallax": true, "is_giant": false,
    "is_metal_poor": false, "is_binary_candidate": false,
    "is_high_velocity": false, "has_emission": false
  },
  "quality_score": 0.85
}
```

### 6.2 Candidata Binaria (RUWE extremo)

**source_id:** `3586141394007639424`

```json
{
  "physical_vector": {
    "abs_mag": 2.8542, "teff_k": 6300, "metallicity": -0.7214,
    "fe_h": -0.09, "alpha_fe": 0.18, "logg": 3.8933,
    "v_tan": 44.6597, "extinction_ag": 0.0052
  },
  "logical_flags": {
    "is_reliable_parallax": true, "is_giant": true,
    "is_metal_poor": false, "is_binary_candidate": true,
    "is_high_velocity": false, "has_emission": false
  },
  "binary_diagnostics": {
    "ruwe": 5.524089, "rv_nb_transits": 52, "nss_solution": null,
    "adaptive_rv_threshold": 5.0
  },
  "quality_score": 0.85
}
```

> RUWE = 5.52 es un indicador fuerte de binariedad no resuelta. El SC debe ponderar la clasificación con cautela — los parámetros físicos pueden estar contaminados por el compañero.

### 6.3 Estrella con Emisión H-alpha (actividad cromosférica)

**source_id:** `3712538811193759744`

```json
{
  "physical_vector": {
    "abs_mag": 6.0151, "teff_k": 5132, "metallicity": -0.1492,
    "fe_h": -0.02, "alpha_fe": 0.19, "logg": 4.6222,
    "v_tan": 15.8262, "extinction_ag": 0.0
  },
  "logical_flags": {
    "is_reliable_parallax": true, "is_giant": false,
    "is_metal_poor": false, "is_binary_candidate": false,
    "is_high_velocity": false, "has_emission": true,
    "has_emission_source": "catalog"
  },
  "spectral_summary": {
    "halpha_catalog": {"ew_aa": -0.209, "flag": 0}
  },
  "quality_score": 1.0
}
```

> `ew_aa = -0.209` (negativo = emisión en convención ESP-ELS). Estrella K enana con actividad cromosférica moderada. La enana K con logg = 4.62 y Teff = 5132 K es consistente con un objeto joven o activo del Disco Fino.

### 6.4 Estrella Pobre en Metales (candidata a Halo)

**source_id:** `1386158505321789696`

```json
{
  "physical_vector": {
    "abs_mag": -0.1492, "teff_k": 9997, "metallicity": -1.0036,
    "fe_h": null, "alpha_fe": 0.46, "logg": 3.5662,
    "v_tan": 15.629, "extinction_ag": 0.003
  },
  "logical_flags": {
    "is_reliable_parallax": true, "is_giant": false,
    "is_metal_poor": true, "is_binary_candidate": true,
    "is_high_velocity": false, "has_emission": false
  },
  "binary_diagnostics": {
    "ruwe": 4.7503, "nss_solution": "Orbital",
    "adaptive_rv_threshold": 10.0
  },
  "quality_score": 1.0
}
```

> `nss_solution: "Orbital"` = binaria confirmada por Gaia. `alpha_fe = 0.46` es consistente con población del Halo (enriquecimiento α típico de formación estelar rápida). `fe_h = null` porque `fem_gspspec` no está disponible para esta estrella en DR3.

---

## 7. Schema de Output SC Esperado

El SC debe producir exactamente este JSON por contrato:

```json
{
  "source_id": "string",
  "classification": {
    "spectral_type": "string (letra: O, B, A, F, G, K, M)",
    "sub_type_range": "string (ej. '1-2', '5-7')",
    "luminosity_class": "string (I, II, III, IV, V)",
    "population_group": "string (Halo | Disco Grueso | Disco Fino)"
  },
  "confidence_scores": {
    "spectral_type_confidence": "float [0.0 – 1.0]",
    "luminosity_confidence": "float [0.0 – 1.0]",
    "population_confidence": "float [0.0 – 1.0]"
  },
  "technical_reasoning": "string (justificación astrofísica detallada)"
}
```

**Incertidumbre inherente:** El `sub_type_range` (en lugar de un subtipo único) refleja la incertidumbre de 100-200 K inherente a las estimaciones Teff de GSP-Phot.

---

## 8. Estrategia de Validación SC (post-inferencia)

### Nivel 1 — Validación MK Gruesa

Cruce de `spectral_type` contra `sp_type` de SIMBAD (disponible en `ground_truth_clean.csv` para 491/498 estrellas). Métrica: accuracy de tipo espectral principal (letra).

### Nivel 2 — Validación de Parámetros Físicos

Cruce de la clasificación contra PASTEL para las 340 estrellas con parámetros de alta resolución:

|Campo SC|Campo PASTEL|Métrica sugerida|
|---|---|---|
|`spectral_type` + `sub_type_range`|`teff_pastel`|ΔTeff medio (K)|
|`luminosity_class`|`logg_pastel`|Δlogg medio (dex)|
|`population_group`|`feh_pastel`|Confusión Halo/Disco|

**Campo `n_pastel_measurements`:** Número de estudios independientes que contribuyen a la mediana PASTEL. Úsalo como peso de confianza en la validación — una estrella con 32 estudios (ej. _mu. Ara_) es ground truth mucho más sólido que una con 1.

---

## 9. Archivos del Proyecto

```
DB-1-2026-04-28/
├── catalog.csv                          # 498 estrellas, 22 columnas
├── rvs/                                 # ~430 espectros RVS (.npz por source_id)
│   └── {source_id}.npz                  # keys: wavelength_nm, flux, flux_error
├── spectra_bprp.npy                     # (498, 343) flux BP/RP
├── spectra_bprp_ids.npy                 # (498,) source_ids alineados
├── sampling_bprp.npy                    # (343,) wavelengths 336–1020 nm
├── run_manifest.json                    # Metadata de la corrida HC
└── validation/
    └── ground_truth_clean.csv           # 498 estrellas, PASTEL + SIMBAD

Scripts/
├── run_full_dataset.py                  # Pipeline HC principal (HC-2.0)
├── join_pastel_local.py                 # Join PASTEL offline
└── ground_truth_cleaner.py             # Limpieza y validación del ground truth

JSON HC output (última corrida):
└── JSON-hc-prueba-5-20260428/
    └── hc_contracts.json                # 498 contratos, pipeline_version: HC-2.0
```

---

## 10. Decisiones de Diseño y Caveats para el SC

1. **H-alpha desde catálogo, no desde espectro:** El LineAgent opera únicamente sobre el Ca II Triplete en RVS (846–870 nm). H-alpha (656.3 nm) se toma de `ew_espels_halpha` (ESP-ELS pipeline de Gaia). La resolución BP/RP (2 nm/pixel) es insuficiente para ajuste Voigt. Esta es una decisión de diseño definitiva.
    
2. **Convención de signo EW H-alpha (crítica):** En la convención ESP-ELS, `ew_espels_halpha < 0` indica emisión real. El flag `has_emission: true` ya incorpora esta lógica. El SC no necesita re-interpretar el signo.
    
3. **`fe_h` vs `metallicity`:** El contrato contiene dos descriptores de química: `metallicity` ([M/H] fotométrico de GSP-Phot, siempre presente) y `fe_h` ([Fe/H] espectroscópico de `fem_gspspec`, ~44% cobertura). Cuando ambos están disponibles, `fe_h` es más preciso. Cuando `fe_h` es null, el SC debe usar `metallicity` como proxy.
    
4. **`is_binary_candidate` y clasificación conservadora:** Cuando este flag es True, los parámetros físicos pueden estar contaminados por el compañero no resuelto. El SC debe reflejar esta incertidumbre con confidence scores más bajos y documentarlo en `technical_reasoning`.
    
5. **Estrellas con `quality_score < 0.5`:** Son 86 estrellas (17.3%). Principalmente espectros RVS con flux NaN o BP/RP de bajo SNR. El SC puede clasificarlas usando solo el vector físico tabular, pero debe documentar la limitación espectral.
    
6. **`logg` pasa directo al SC:** `logg_gspphot` no es procesado por ningún módulo HC — va directamente del catálogo al `physical_vector`. Esto es por diseño: la clase de luminosidad se estima mejor desde logg que desde M_G para estrellas con extinción incierta.
    
7. **Umbral `is_giant`:** `M_G < 3.0 AND Teff < 7000 K AND is_reliable_parallax`. El corte en 7000 K excluye estrellas A/B calientes con M_G brillante que no son gigantes evolutivas.