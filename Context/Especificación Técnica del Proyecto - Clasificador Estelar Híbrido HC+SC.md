
## 1. Resumen del Proyecto

El objetivo de este proyecto es desarrollar un sistema de clasificación espectral estelar automático, explicable y de alta precisión utilizando datos de la misión Gaia DR3/DR4. La problemática radica en que los espectros reales presentan ruidos, enrojecimiento interestelar y contaminación por binariedad que confunden a los métodos tradicionales.

Nuestra solución propone una arquitectura de dos capas:

1. **Hard Computing (HC):** Algoritmos deterministas basados en física clásica y estadística para extraer parámetros limpios y generar "banderas lógicas" (predigestión) .
    
2. **Soft Computing (SC):** Inferencia mediante el modelo de lenguaje especializado **AstroSage-Llama**, que recibe los datos pre-digeridos para emitir un diagnóstico astrofísico final .
    

## 2. Fundamentación Astronómica y Astrofísica

### 2.1 El Continuo y la Extinción

El flujo de una estrella se ve modificado por el polvo interestelar que "enrojece" la luz (`ebpminrp_gspphot`). Cabe recalcar que este parámetro es una estimación fotométrica del pipeline GSP-Phot y posee incertidumbres significativas, especialmente para estrellas frías o con gravedad superficial (logg) baja. Su uso dentro del módulo HC se maneja con márgenes de error propagados.

### 2.2 Diagnósticos de Línea y Convención de Signo

- **H-alpha (656.3 nm):** Indicador clave de actividad estelar. _Nota técnica sobre convención de signo:_ Según el diccionario de datos de nuestro catálogo (pipeline ESP-ELS), los valores negativos indican absorción y los positivos indican emisión. Nuestro sistema HC obedece estrictamente a esta convención.
    
- **Triplete de Calcio (849-866 nm):** En el espectro RVS, estas líneas son altamente sensibles a la gravedad superficial y la metalicidad.
    

### 2.3 Química Estelar y Poblaciones

La metalicidad global (`mh_gspphot`) y la abundancia de hierro (`fem_gspspec`) nos indican el nivel de enriquecimiento químico. Esto, combinado con el cociente alfa/Fe (`alphafe_gspspec`), nos permite diferenciar estrellas de poblaciones galácticas distintas (Disco Fino vs. Halo).

### 2.4 Astrometría Derivada y Diagrama HR

La combinación del paralaje y la magnitud aparente en la banda G permite calcular la Magnitud Absoluta ($M_G$). Esto, junto con la temperatura (`teff_gspphot`), determina la posición en el diagrama HR.

## 3. Configuración de Módulos de Hard Computing (HC)

| **Módulo**               | **Método Determinista**     | **Detalles y Parámetros de Salida**                                                      |
| ------------------------ | --------------------------- | ---------------------------------------------------------------------------------------- |
| **ContinuumAgent**       | Splines Cúbicos / Chebyshev | Espectro normalizado. Se estima en regiones de pseudo-continuo definidas explícitamente. |
| **LineAgent**            | Ajuste de Perfiles de Voigt | Extrae Ancho Equivalente (EW), FWHM y Desplazamiento Doppler.                            |
| **BinaryDetector**       | Análisis de ABIC y RUWE     | RUWE se utiliza como proxy de "candidato a binaria o fuente problemática".               |
| **Astrometría Derivada** | Trigonometría y Álgebra     | Calcula la Magnitud Absoluta ($M_G$) y la Velocidad Espacial Tangencial ($V_{tan}$).     |

## 4. Requerimientos de Datos (Gaia DR3)

- **Identificación:** `source_id`.
    
- **Astrometría:** `ra`, `dec`, `parallax`, `parallax_error`, `pmra`, `pmdec`, `ruwe`.
    
- **Fotometría:** `phot_g_mean_mag`, `ebpminrp_gspphot`.
    
- **Química:** `mh_gspphot`, `alphafe_gspspec`, `fem_gspspec`.
    
- **Física:** `teff_gspphot`, `logg_gspphot`.
    
- **Líneas:** `ew_espels_halpha` y `ew_espels_halpha_flag`.
    

## 5. Esquema de Interfaces JSON y Banderas Lógicas (Predigestión)

### 5.1 Output de HC (Input para el LLM)

Se incorpora `fe_h` al vector físico para aprovechar la extracción espectroscópica fina.

JSON

```
{
  "source_id": "int64",
  "physical_vector": {
    "abs_mag": "float ($M_G$)",
    "teff_k": "int (K)",
    "metallicity": "float ([M/H])",
    "fe_h": "float ([Fe/H] derivado de fem_gspspec)",
    "alpha_fe": "float ([alpha/Fe])",
    "v_tan": "float (km/s)"
  },
  "logical_flags": {
    "is_reliable_parallax": "boolean (Si parallax / parallax_error > 5)",
    "is_giant": "boolean (Si $M_G$ < 3 AND Teff < 7000 K)",
    "is_metal_poor": "boolean (Si [M/H] < -1.0)",
    "is_binary_candidate": "boolean (Si RUWE > 1.4 OR ABIC favorece modelo binario)",
    "is_high_velocity": "boolean (Si $V_{tan}$ > 200 km/s AND is_reliable_parallax == true)",
    "has_emission": "boolean (Si EW_Halpha > 0 según convención ESP-ELS)"
  }
}
```

### 5.2 Output de SC (Inferencia Final y Validación)

El campo `sub_type` se ha convertido en un rango (`sub_type_range`) para reflejar honestamente la incertidumbre de 100-200 K inherente a las estimaciones de GSP-Phot.

JSON

```
{
  "classification": {
    "spectral_type": "string (ej. K)",
    "sub_type_range": "string (ej. 1-2)",
    "luminosity_class": "string (I-V)",
    "population_group": "string (Halo, Disco Fino, Disco Grueso)"
  },
  "confidence_scores": {
    "spectral_type_confidence": "float (0.0 - 1.0)",
    "luminosity_confidence": "float (0.0 - 1.0)",
    "population_confidence": "float (0.0 - 1.0)"
  },
  "technical_reasoning": "string (Justificación detallada de la clasificación)"
}
```

**Estrategia de Validación Científica Multi-Nivel:**

Para garantizar la credibilidad del output de AstroSage-Llama, se implementará un _benchmark_ automatizado en dos frentes:

1. **Clasificación MK Gruesa:** Se cruzarán los resultados con bibliotecas espectrales estándar como Pickles y MILES para validar el tipo espectral general (G, K, M).
    
2. **Parámetros Físicos y Química Fina:** Para validar que el LLM está interpretando correctamente la metalicidad y la clase de luminosidad fina, se validará el output contra un subconjunto de estrellas del catálogo **PASTEL** (parámetros de alta resolución de la literatura) cruzado con los `source_id` de Gaia. Esto cierra el loop de manera end-to-end.