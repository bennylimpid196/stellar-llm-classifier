"""
Descarga estratificada de Gaia DR3 para el sistema multi-agente de clasificación espectral.

Estructura:
    Bloque 1 — Secuencia principal normal         1.500 objetos (30%)
    Bloque 2 — Gigantes y subgigantes               800 objetos (16%)
    Bloque 3 — Casos de conflicto entre agentes   1.000 objetos (20%)
    Bloque 4 — Alta extinción                       500 objetos (10%)
    Bloque 5 — Ground truth verificable            1.200 objetos (24%)
    ─────────────────────────────────────────────────────────────
    TOTAL                                          5.000 objetos

Estrategia de descarga:
    - Query base con columnas compartidas por todos los agentes
    - Queries de enriquecimiento opcionales para columnas de menor cobertura
      (GSP-Spec/RVS: fem_gspspec, alphafe_gspspec; ESP-ELS: ew_espels_halpha)
    - Merge final en pandas con left join sobre source_id
    - Columnas ausentes → NaN → degradación de confianza documentada por cada agente

Uso:
    # Smoke test inmediato (5 objetos por bloque, ~30 segundos)
    python gaia_queries.py --test

    # Descarga completa (~5.000 objetos)
    python gaia_queries.py

    # Descarga completa con ground truth GALAH
    python gaia_queries.py --galah ruta/a/galah_crossmatch.csv

Requisitos:
    pip install astroquery astropy pandas pyarrow
"""

import sys
import time
import argparse
import pandas as pd
from astroquery.gaia import Gaia

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1  # Sin límite; cada query tiene su propio TOP

# Tamaño de lote para queries de enriquecimiento con IN (...)
# El TAP de ESA puede rechazar queries con strings muy largos;
# 500 IDs por lote es un valor conservador y seguro.
ENRICHMENT_BATCH_SIZE = 500

# Número de objetos por bloque en modo smoke test
SMOKE_N = 5

# ── Constantes de resiliencia ──────────────────────────────────
# Segundos de espera antes del reintento síncrono tras un error 500
RETRY_DELAY = 5

# Límite de filas por llamada síncrona en el TAP de ESA
# (el servidor rechaza peticiones síncronas que superen este valor)
SYNC_ROW_LIMIT = 2000

# Pausa entre páginas en el modo sync paginado (cortesía con el servidor)
PAGE_DELAY = 2

# Columnas base: compartidas por todos los agentes
# Presentes en prácticamente todas las fuentes con GSP-Phot disponible
BASE_COLS = """
    source_id,
    ra,
    dec,
    l,
    b,
    parallax,
    parallax_error,
    parallax_over_error,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    phot_g_mean_flux_over_error,
    teff_gspphot,
    logg_gspphot,
    mh_gspphot,
    ebpminrp_gspphot,
    ruwe,
    non_single_star
"""

# Versión con alias gs. para queries con JOIN a astrophysical_parameters
# Necesario porque varias columnas existen en ambas tablas y ADQL las rechaza como ambiguas
BASE_COLS_GS = """
    gs.source_id,
    gs.ra,
    gs.dec,
    gs.l,
    gs.b,
    gs.parallax,
    gs.parallax_error,
    gs.parallax_over_error,
    gs.phot_g_mean_mag,
    gs.phot_bp_mean_mag,
    gs.phot_rp_mean_mag,
    gs.phot_g_mean_flux_over_error,
    gs.teff_gspphot,
    gs.logg_gspphot,
    gs.mh_gspphot,
    gs.ebpminrp_gspphot,
    gs.ruwe,
    gs.non_single_star
"""

# Columnas de enriquecimiento: menor cobertura, queries separadas
ENRICHMENT_COLS_GSPSPEC = """
    source_id,
    fem_gspspec,
    alphafe_gspspec,
    radial_velocity,
    radial_velocity_error
"""

ENRICHMENT_COLS_ESPELS = """
    source_id,
    ew_espels_halpha,
    ew_espels_halpha_uncertainty,
    ew_espels_halpha_flag
"""

# Filtros de calidad mínima — versión sin alias (para queries de tabla simple)
QUALITY_FLOOR = """
    parallax_over_error > 5
    AND phot_g_mean_flux_over_error > 50
    AND teff_gspphot IS NOT NULL
    AND logg_gspphot IS NOT NULL
    AND ruwe IS NOT NULL
"""

# Versión con alias gs. para queries con JOIN a astrophysical_parameters
QUALITY_FLOOR_GS = """
    gs.parallax_over_error > 5
    AND gs.phot_g_mean_flux_over_error > 50
    AND gs.teff_gspphot IS NOT NULL
    AND gs.logg_gspphot IS NOT NULL
    AND gs.ruwe IS NOT NULL
"""


# ─────────────────────────────────────────────────────────────
# BLOQUE 1 — SECUENCIA PRINCIPAL NORMAL (1.500 objetos)
# ─────────────────────────────────────────────────────────────
# Estrellas de todos los tipos espectrales en la secuencia principal.
# Baja extinción, RUWE limpio, sin emisión Hα anómala.
# Distribuidas por tipo espectral según T_eff; proporciones en comentarios.

QUERY_BLOCK1 = f"""
SELECT TOP 1500
    {BASE_COLS}
FROM gaiadr3.gaia_source
WHERE
    {QUALITY_FLOOR}
    AND logg_gspphot >= 4.0                  -- secuencia principal
    AND ruwe < 1.4                           -- fuente puntual limpia
    AND ebpminrp_gspphot < 0.3               -- baja extinción
    AND teff_gspphot BETWEEN 3000 AND 40000  -- cubre tipos M a O
ORDER BY
    source_id  -- IDs de Gaia son pseudoaleatorios por diseño; ordenar por ellos da distribución uniforme
"""

# Nota: si se quiere control fino por tipo espectral, reemplazar la query única
# por 6 sub-queries con rangos de T_eff y TOP individuales:
#   O/B: teff > 10000  → TOP 150
#   A:   teff 7500-10000 → TOP 200
#   F:   teff 6000-7500  → TOP 250
#   G:   teff 5200-6000  → TOP 350
#   K:   teff 3700-5200  → TOP 350
#   M:   teff < 3700     → TOP 200
# Ver función build_block1_by_subtype() al final del archivo.


# ─────────────────────────────────────────────────────────────
# BLOQUE 2 — GIGANTES Y SUBGIGANTES (800 objetos)
# ─────────────────────────────────────────────────────────────
# Estrellas evolucionadas. Prueba la separación enana/gigante,
# el error más costoso según el documento.
# log g < 3.5 → gigantes RGB y AGB
# log g 3.5–4.0 → subgigantes (zona ambigua)

QUERY_BLOCK2 = f"""
SELECT TOP 800
    {BASE_COLS}
FROM gaiadr3.gaia_source
WHERE
    {QUALITY_FLOOR}
    AND logg_gspphot < 4.0                   -- gigantes y subgigantes
    AND logg_gspphot > 0.0                   -- excluye soluciones espurias
    AND ruwe < 1.4
    AND teff_gspphot BETWEEN 3500 AND 8000   -- rango óptimo GSP-Phot
ORDER BY
    source_id
"""

# Para separar explícitamente gigantes (log g < 3.5) de subgigantes (3.5–4.0),
# usar build_block2_by_luminosity_class() al final del archivo.


# ─────────────────────────────────────────────────────────────
# BLOQUE 3 — CASOS DE CONFLICTO ENTRE AGENTES (1.000 objetos)
# ─────────────────────────────────────────────────────────────
# Cuatro sub-bloques, 250 objetos cada uno.
# Cada sub-bloque activa un caso de conflicto documentado en la sección 4.1.

# 3A: Emisión Hα + tipo B/A → hipótesis estrellas Be y Herbig Ae/Be
QUERY_BLOCK3A = f"""
SELECT TOP 250
    {BASE_COLS_GS}
FROM gaiadr3.gaia_source AS gs
JOIN gaiadr3.astrophysical_parameters AS ap
    ON gs.source_id = ap.source_id
WHERE
    {QUALITY_FLOOR_GS}
    AND gs.teff_gspphot > 7500               -- tipos A y B
    AND ap.ew_espels_halpha IS NOT NULL
    AND ap.ew_espels_halpha > 1.0            -- CORREGIDO: > 0 indica emisión
    AND ap.ew_espels_halpha_flag = '0'        -- medición de calidad
ORDER BY
    ap.ew_espels_halpha DESC                 -- CORREGIDO: primero las emisiones más intensas
"""

# ... (El bloque 3B y 3C se quedan igual) ...

# 3D: Estrellas M activas / T Tauri tardías → emisión Hα + T_eff < 4000 K
QUERY_BLOCK3D = f"""
SELECT TOP 250
    {BASE_COLS_GS}
FROM gaiadr3.gaia_source AS gs
JOIN gaiadr3.astrophysical_parameters AS ap
    ON gs.source_id = ap.source_id
WHERE
    {QUALITY_FLOOR_GS}
    AND gs.teff_gspphot < 4000               -- tipos M
    AND ap.ew_espels_halpha IS NOT NULL
    AND ap.ew_espels_halpha > 1.0            -- CORREGIDO: emisión neta positiva
    AND ap.ew_espels_halpha_flag = '0'
ORDER BY
    gs.teff_gspphot ASC                      -- primero las más frías
"""

# 3C: Metal-pobres del halo → [Fe/H] < -1.0, [α/Fe] > +0.3
# Usa columnas GSP-Spec (RVS), solo disponibles para G < ~16
QUERY_BLOCK3C = f"""
SELECT TOP 250
    {BASE_COLS}
FROM gaiadr3.gaia_source
WHERE
    {QUALITY_FLOOR}
    AND mh_gspphot < -1.0                   -- proxy fotométrico de baja metalicidad
    AND phot_g_mean_mag < 16                -- rango de disponibilidad del RVS
    AND ruwe < 1.4
ORDER BY
    source_id                               -- CORREGIDO: Evita el timeout por ordenamiento masivo
"""

# 3D: Estrellas M activas / T Tauri tardías → emisión Hα + T_eff < 4000 K
QUERY_BLOCK3D = f"""
SELECT TOP 250
    {BASE_COLS_GS}
FROM gaiadr3.gaia_source AS gs
JOIN gaiadr3.astrophysical_parameters AS ap
    ON gs.source_id = ap.source_id
WHERE
    {QUALITY_FLOOR_GS}
    AND gs.teff_gspphot < 4000               -- tipos M
    AND ap.ew_espels_halpha IS NOT NULL
    AND ap.ew_espels_halpha < -1.0           -- emisión neta
    AND ap.ew_espels_halpha_flag = '0'
ORDER BY
    gs.teff_gspphot ASC                      -- primero las más frías
"""


# ─────────────────────────────────────────────────────────────
# BLOQUE 4 — ALTA EXTINCIÓN (500 objetos)
# ─────────────────────────────────────────────────────────────
# Estrellas con E(BP-RP) alto; prueba si ContinuumPhotometryAgent
# detecta incertidumbre en la corrección y si el integrador la propaga.
# Se excluye el plano galáctico más denso (|b| < 5°) para evitar
# regiones donde GSP-Phot colapsa completamente.

QUERY_BLOCK4 = f"""
SELECT TOP 500
    {BASE_COLS}
FROM gaiadr3.gaia_source
WHERE
    {QUALITY_FLOOR}
    AND ebpminrp_gspphot > 0.5               -- extinción significativa
    AND ABS(b) > 5                           -- evita el plano más denso
    AND ruwe < 1.4
    AND teff_gspphot BETWEEN 4000 AND 15000  -- rango donde la corrección importa más
ORDER BY
    ebpminrp_gspphot DESC                    -- primero las más enrojecidas
"""


# ─────────────────────────────────────────────────────────────
# BLOQUE 5 — GROUND TRUTH VERIFICABLE (1.200 objetos)
# ─────────────────────────────────────────────────────────────
# Estrellas con clasificación MK independiente por espectroscopía
# de alta resolución. Cruce con GALAH DR3 vía source_id de Gaia.
#
# GALAH DR3 está disponible en VizieR (catálogo J/MNRAS/506/150).
# La tabla galah_dr3.main_star en el TAP de GALAH tiene la columna
# "gaiadr2_source_id"; para DR3 se requiere un crossmatch adicional
# usando gaiadr3.dr2_neighbourhood.
#
# Opción A (recomendada para empezar): usar la tabla de crossmatch
# interna de Gaia entre DR2 y DR3 para recuperar los source_id DR3
# correspondientes a las estrellas GALAH.
#
# La query siguiente asume que ya tienes una lista de source_ids DR3
# de GALAH pre-computada (ver función load_galah_sourceids() abajo).
# Si no, usar la query de crossmatch comentada al final.

def build_block5_query(galah_source_ids: list) -> str:
    """
    Construye la query del Bloque 5 dado una lista de source_id DR3
    provenientes del cruce con GALAH DR3.

    Args:
        galah_source_ids: Lista de integers con los source_id de Gaia DR3.

    Returns:
        String ADQL listo para ejecutar con Gaia.launch_job().
    """
    ids_str = ", ".join(str(sid) for sid in galah_source_ids[:1200])
    return f"""
    SELECT
        {BASE_COLS}
    FROM gaiadr3.gaia_source
    WHERE
        source_id IN ({ids_str})
        AND teff_gspphot IS NOT NULL
        AND logg_gspphot IS NOT NULL
        AND ruwe IS NOT NULL
    """

# Query de crossmatch DR2 → DR3 para obtener los source_ids DR3 de GALAH
# (ejecutar una sola vez y guardar el resultado)
QUERY_GALAH_CROSSMATCH = """
SELECT
    n.dr3_source_id,
    n.dr2_source_id,
    n.angular_distance
FROM gaiadr3.dr2_neighbourhood AS n
WHERE
    n.dr2_source_id IN (
        -- Sustituir por la lista de gaiadr2_source_id extraída de GALAH DR3
        -- Ejemplo: 1234567890, 9876543210, ...
        SELECT source_id_dr2 FROM TAP_UPLOAD.galah_ids
    )
    AND n.angular_distance < 0.5   -- arcsec, para crossmatch seguro
ORDER BY
    n.angular_distance ASC
"""


# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES DE DESCARGA
# ─────────────────────────────────────────────────────────────

def _launch_with_fallback(adql: str) -> pd.DataFrame:
    """
    Intenta ejecutar una query ADQL con la siguiente estrategia de resiliencia:

      1. launch_job_async  — modo normal, sin límite de filas.
      2. launch_job        — síncrono, máx. 2.000 filas por llamada.
         Se usa como fallback cuando el servidor devuelve 500/null,
         que es el síntoma documentado de los problemas de infraestructura
         del archivo de ESA (aviso en https://www.cosmos.esa.int/web/gaia/news).
      3. Si ambos fallan, re-lanza la excepción original para que el
         caller la gestione.

    Entre el intento async y el sync espera RETRY_DELAY segundos para
    no saturar el servidor durante episodios de inestabilidad.
    """
    try:
        job = Gaia.launch_job_async(adql, verbose=False)
        return job.get_results().to_pandas()

    except Exception as e_async:
        print(f"  ⚠ launch_job_async falló ({type(e_async).__name__}: {e_async})")
        print(f"  → Reintentando en {RETRY_DELAY}s con launch_job (síncrono, máx. 2.000 filas)...")
        time.sleep(RETRY_DELAY)

        try:
            # launch_job síncrono: el servidor lo procesa de inmediato y
            # es más estable durante episodios de degradación del TAP async.
            job = Gaia.launch_job(adql, verbose=False)
            return job.get_results().to_pandas()
        except Exception as e_sync:
            print(f"  ✗ launch_job también falló ({type(e_sync).__name__}: {e_sync})")
            raise e_sync


def _inject_top(adql: str, n: int) -> str:
    """
    Inserta o reemplaza la cláusula TOP en una query ADQL.
    Necesario para trocear bloques grandes en llamadas síncronas de ≤2.000 filas.
    """
    import re
    # Reemplaza TOP existente
    adql_mod = re.sub(r"SELECT\s+TOP\s+\d+", f"SELECT TOP {n}", adql,
                      flags=re.IGNORECASE)
    # Si no había TOP, lo inserta tras SELECT
    if not re.search(r"SELECT\s+TOP\s+\d+", adql_mod, re.IGNORECASE):
        adql_mod = re.sub(r"SELECT\s+", f"SELECT TOP {n} ", adql_mod,
                          count=1, flags=re.IGNORECASE)
    return adql_mod


def _extract_top(adql: str) -> int:
    """Extrae el valor de TOP de una query ADQL. Devuelve 0 si no hay TOP."""
    import re
    m = re.search(r"SELECT\s+TOP\s+(\d+)", adql, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _spinner(label: str, stop_event):
    """
    Hilo que imprime un spinner animado con tiempo transcurrido
    mientras se espera respuesta del servidor.
    Se detiene cuando stop_event se activa.
    """
    import itertools
    frames = itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"])
    t0 = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - t0
        print(f"\r  {next(frames)} {label}  {elapsed:.0f}s", end="", flush=True)
        time.sleep(0.12)
    print(f"\r  {chr(32)*70}", end="\r", flush=True)


def run_query(adql: str, description: str) -> pd.DataFrame:
    """
    Ejecuta una query ADQL contra el TAP de Gaia ESA con resiliencia completa.
    Muestra spinner animado con tiempo transcurrido mientras espera.
    """
    import threading

    print(f"\n{'─'*60}")
    print(f"Ejecutando: {description}")
    print(f"{'─'*60}")
    t0 = time.time()

    # ── Intento 1: async (el camino normal) ─────────────────────
    stop = threading.Event()
    spin = threading.Thread(
        target=_spinner,
        args=("Esperando servidor ESA (async)...", stop),
        daemon=True
    )
    spin.start()
    try:
        job = Gaia.launch_job_async(adql, verbose=False)
        df = job.get_results().to_pandas()
        stop.set(); spin.join()
        elapsed = time.time() - t0
        print(f"  → {len(df):,} filas  ({elapsed:.1f}s)  [async]")
        return df
    except Exception as e_async:
        stop.set(); spin.join()
        print(f"  ⚠ Async falló ({type(e_async).__name__}). Activando fallback síncrono...")
        time.sleep(RETRY_DELAY)

    # ── Intento 2: sync simple (funciona para ≤ SYNC_ROW_LIMIT filas) ──
    n_requested = _extract_top(adql)
    if n_requested <= SYNC_ROW_LIMIT:
        stop2 = threading.Event()
        spin2 = threading.Thread(
            target=_spinner,
            args=("Esperando servidor ESA (sync)...", stop2),
            daemon=True
        )
        spin2.start()
        try:
            job = Gaia.launch_job(adql, verbose=False)
            df = job.get_results().to_pandas()
            stop2.set(); spin2.join()
            elapsed = time.time() - t0
            print(f"  → {len(df):,} filas  ({elapsed:.1f}s)  [sync]")
            return df
        except Exception as e_sync:
            stop2.set(); spin2.join()
            print(f"  ✗ Sync también falló: {e_sync}")
            raise e_sync

    # ── Intento 3: sync paginado (para bloques grandes) ─────────
    # El servidor síncrono tiene un límite de SYNC_ROW_LIMIT filas por llamada.
    # Se simula paginación con OFFSET. Nota: OFFSET en ADQL/TAP no es estándar
    # en todas las implementaciones; el TAP de ESA lo soporta.
    print(f"  → Paginando {n_requested:,} filas en chunks de {SYNC_ROW_LIMIT}...")
    parts = []
    offset = 0
    while offset < n_requested:
        chunk = min(SYNC_ROW_LIMIT, n_requested - offset)

        # Construir query con TOP y OFFSET
        import re
        adql_chunk = _inject_top(adql, chunk)
        # Insertar OFFSET antes del ORDER BY si existe, o al final
        if re.search(r"ORDER\s+BY", adql_chunk, re.IGNORECASE):
            adql_chunk = re.sub(
                r"(ORDER\s+BY.*?)$",
                f"OFFSET {offset} \\1",
                adql_chunk,
                flags=re.IGNORECASE | re.DOTALL
            )
        else:
            adql_chunk = adql_chunk.rstrip() + f"\nOFFSET {offset}"

        stop_p = threading.Event()
        spin_p = threading.Thread(
            target=_spinner,
            args=(f"Página {offset//SYNC_ROW_LIMIT + 1} (offset={offset})...", stop_p),
            daemon=True
        )
        spin_p.start()
        try:
            job = Gaia.launch_job(adql_chunk, verbose=False)
            df_chunk = job.get_results().to_pandas()
            stop_p.set(); spin_p.join()
        except Exception as e:
            stop_p.set(); spin_p.join()
            print(f"  ✗ Chunk offset={offset} falló: {e}")
            raise

        if df_chunk.empty:
            break  # No hay más filas

        parts.append(df_chunk)
        offset += len(df_chunk)
        print(f"  Página offset={offset - len(df_chunk)}: {len(df_chunk):,} filas")

        if len(df_chunk) < chunk:
            break  # El servidor devolvió menos de lo pedido → fin de datos

        time.sleep(PAGE_DELAY)  # Pausa cortés entre páginas

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    elapsed = time.time() - t0
    print(f"  → {len(df):,} filas totales  ({elapsed:.1f}s)  [sync paginado]")
    return df


def merge_with_enrichment(base_df: pd.DataFrame,
                          enrichment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left join del DataFrame base con uno de enriquecimiento.
    Las columnas ausentes quedan como NaN: comportamiento deseado,
    el agente correspondiente degradará su confianza explícitamente.
    """
    if enrichment_df.empty:
        return base_df
    return base_df.merge(enrichment_df, on="source_id", how="left")


def _batch_ids(source_ids: list, batch_size: int = ENRICHMENT_BATCH_SIZE):
    """Parte una lista de IDs en sublistas de tamaño batch_size."""
    for i in range(0, len(source_ids), batch_size):
        yield source_ids[i : i + batch_size]


def run_enrichment_batched(source_ids: list, cols: str, table: str,
                           extra_filter: str, description: str) -> pd.DataFrame:
    """
    Ejecuta queries de enriquecimiento en lotes de ENRICHMENT_BATCH_SIZE
    para evitar que el TAP de ESA rechace strings demasiado largos.

    Args:
        source_ids   : Lista completa de source_id sobre los que enriquecer.
        cols         : String ADQL con las columnas a seleccionar.
        table        : Tabla ADQL fuente (ej. gaiadr3.gaia_source).
        extra_filter : Condición WHERE adicional (ej. 'AND fem_gspspec IS NOT NULL').
        description  : Etiqueta para logging.

    Returns:
        DataFrame concatenado de todos los lotes.
    """
    n_batches = max(1, (len(source_ids) + ENRICHMENT_BATCH_SIZE - 1) // ENRICHMENT_BATCH_SIZE)
    print(f"\n{'─'*60}")
    print(f"Enriquecimiento — {description}")
    print(f"  {len(source_ids):,} IDs en {n_batches} lote(s) de {ENRICHMENT_BATCH_SIZE}")

    parts = []
    for i, batch in enumerate(_batch_ids(source_ids), start=1):
        ids_str = ", ".join(str(sid) for sid in batch)
        adql = f"""
        SELECT {cols}
        FROM {table}
        WHERE source_id IN ({ids_str})
        {extra_filter}
        """
        t0 = time.time()
        df = _launch_with_fallback(adql)
        elapsed = time.time() - t0
        parts.append(df)
        print(f"  Lote {i}/{n_batches}: {len(df):,} filas  ({elapsed:.1f}s)")

    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"  → Total con datos: {len(result):,} filas")
    return result


# ─────────────────────────────────────────────────────────────
# FUNCIONES DE MUESTREO FINO (opcionales)
# ─────────────────────────────────────────────────────────────

def build_block1_by_subtype(n_scale: float = 1.0) -> list:
    """
    Genera 6 sub-queries del Bloque 1 con control fino por tipo espectral.

    Args:
        n_scale: Factor de escala sobre los conteos nominales.
                 Usar 1/300 para smoke test (5 objetos por sub-tipo).

    Returns:
        Lista de tuplas (adql, descripción).
    """
    subtypes = [
        ("O/B", 10000, 40000, 150),
        ("A",    7500, 10000, 200),
        ("F",    6000,  7500, 250),
        ("G",    5200,  6000, 350),
        ("K",    3700,  5200, 350),
        ("M",    3000,  3700, 200),
    ]
    queries = []
    for name, tmin, tmax, n_base in subtypes:
        n = max(1, int(n_base * n_scale))
        adql = f"""
        SELECT TOP {n}
            {BASE_COLS}
        FROM gaiadr3.gaia_source
        WHERE
            {QUALITY_FLOOR}
            AND logg_gspphot >= 4.0
            AND ruwe < 1.4
            AND ebpminrp_gspphot < 0.3
            AND teff_gspphot BETWEEN {tmin} AND {tmax}
        ORDER BY source_id
        """
        queries.append((adql, f"Bloque 1 — Tipo {name}"))
    return queries


def build_block2_by_luminosity_class(n_scale: float = 1.0) -> list:
    """
    Genera 2 sub-queries del Bloque 2: gigantes y subgigantes por separado.
    """
    return [
        (f"""
        SELECT TOP {max(1, int(550 * n_scale))}
            {BASE_COLS}
        FROM gaiadr3.gaia_source
        WHERE
            {QUALITY_FLOOR}
            AND logg_gspphot < 3.5
            AND logg_gspphot > 0.0
            AND ruwe < 1.4
            AND teff_gspphot BETWEEN 3500 AND 8000
        ORDER BY source_id
        """, "Bloque 2 — Gigantes (log g < 3.5)"),

        (f"""
        SELECT TOP {max(1, int(250 * n_scale))}
            {BASE_COLS}
        FROM gaiadr3.gaia_source
        WHERE
            {QUALITY_FLOOR}
            AND logg_gspphot BETWEEN 3.5 AND 4.0
            AND ruwe < 1.4
            AND teff_gspphot BETWEEN 3500 AND 8000
        ORDER BY source_id
        """, "Bloque 2 — Subgigantes (log g 3.5–4.0)"),
    ]


# ─────────────────────────────────────────────────────────────
# SMOKE TEST — verificación rápida (~30 segundos)
# ─────────────────────────────────────────────────────────────

def smoke_test() -> pd.DataFrame:
    """
    Descarga SMOKE_N objetos por bloque para verificar que:
      1. La conexión al TAP de ESA funciona.
      2. Todas las columnas base existen y se deserializan sin errores.
      3. Las queries de enriquecimiento devuelven filas (pueden ser 0, es válido).
      4. El merge left join no introduce duplicados ni columnas fantasma.

    No requiere nada externo. No guarda archivos a menos que pase todos los checks.

    Returns:
        DataFrame con ~35–50 objetos si el test pasa; lanza AssertionError si falla.
    """
    print("=" * 60)
    print(f"  SMOKE TEST — {SMOKE_N} objetos por bloque")
    print("=" * 60)

    N = SMOKE_N
    blocks = []

    # ── Queries base reducidas ──────────────────────────────────
    smoke_queries = [
        (f"""
        SELECT TOP {N} {BASE_COLS}
        FROM gaiadr3.gaia_source
        WHERE {QUALITY_FLOOR}
          AND logg_gspphot >= 4.0 AND ruwe < 1.4 AND ebpminrp_gspphot < 0.3
          AND teff_gspphot BETWEEN 3000 AND 40000
        ORDER BY source_id
        """, "SMOKE Bloque 1 — Secuencia principal"),

        (f"""
        SELECT TOP {N} {BASE_COLS}
        FROM gaiadr3.gaia_source
        WHERE {QUALITY_FLOOR}
          AND logg_gspphot < 4.0 AND logg_gspphot > 0.0
          AND ruwe < 1.4 AND teff_gspphot BETWEEN 3500 AND 8000
        ORDER BY source_id
        """, "SMOKE Bloque 2 — Gigantes/subgigantes"),

        (f"""
        SELECT TOP {N} {BASE_COLS_GS}
        FROM gaiadr3.gaia_source AS gs
        JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id
        WHERE {QUALITY_FLOOR_GS}
          AND gs.teff_gspphot > 7500
          AND ap.ew_espels_halpha IS NOT NULL
          AND ap.ew_espels_halpha > 1.0           -- CORREGIDO: > 1.0 para emisión neta
          AND ap.ew_espels_halpha_flag = '0'
        ORDER BY ap.ew_espels_halpha DESC         -- CORREGIDO: DESC para ver las más intensas
        """, "SMOKE Bloque 3A — Emisión Hα + tipo B/A"),

        (f"""
        SELECT TOP {N} {BASE_COLS}
        FROM gaiadr3.gaia_source
        WHERE {QUALITY_FLOOR}
          AND ruwe > 1.4 AND teff_gspphot BETWEEN 4000 AND 8000
        ORDER BY ruwe DESC
        """, "SMOKE Bloque 3B — RUWE elevado"),

        (f"""
        SELECT TOP {N} {BASE_COLS}
        FROM gaiadr3.gaia_source
        WHERE {QUALITY_FLOOR}
          AND mh_gspphot < -1.0 AND phot_g_mean_mag < 16 AND ruwe < 1.4
        ORDER BY source_id                        -- CORREGIDO: Evita el timeout
        """, "SMOKE Bloque 3C — Metal-pobres del halo"),

        (f"""
        SELECT TOP {N} {BASE_COLS_GS}
        FROM gaiadr3.gaia_source AS gs
        JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id
        WHERE {QUALITY_FLOOR_GS}
          AND gs.teff_gspphot < 4000
          AND ap.ew_espels_halpha IS NOT NULL
          AND ap.ew_espels_halpha > 1.0           -- CORREGIDO: > 1.0 para emisión neta
          AND ap.ew_espels_halpha_flag = '0'
        ORDER BY gs.teff_gspphot ASC
        """, "SMOKE Bloque 3D — Estrellas M activas"),

        (f"""
        SELECT TOP {N} {BASE_COLS}
        FROM gaiadr3.gaia_source
        WHERE {QUALITY_FLOOR}
          AND ebpminrp_gspphot > 0.5 AND ABS(b) > 5
          AND ruwe < 1.4 AND teff_gspphot BETWEEN 4000 AND 15000
        ORDER BY ebpminrp_gspphot DESC
        """, "SMOKE Bloque 4 — Alta extinción"),
    ]

    errors = []
    for adql, desc in smoke_queries:
        try:
            df = run_query(adql, desc)
            df["bloque"] = desc
            blocks.append(df)
        except Exception as e:
            errors.append(f"  ✗ {desc}: {e}")

    if errors:
        print("\n⚠️  Errores en queries base:")
        for err in errors:
            print(err)
        raise AssertionError("Smoke test fallido en queries base. Ver errores arriba.")

    # ── Merge de bloques ────────────────────────────────────────
    base_df = pd.concat(blocks, ignore_index=True).drop_duplicates(subset="source_id")
    print(f"\n✓ Bloques concatenados: {len(base_df):,} objetos únicos")

    # ── Verificar columnas base esperadas ───────────────────────
    expected_base_cols = [
        "source_id", "ra", "dec", "l", "b",
        "parallax", "parallax_error", "parallax_over_error",
        "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
        "phot_g_mean_flux_over_error",
        "teff_gspphot", "logg_gspphot", "mh_gspphot", "ebpminrp_gspphot",
        "ruwe", "non_single_star",
    ]
    missing_cols = [c for c in expected_base_cols if c not in base_df.columns]
    if missing_cols:
        raise AssertionError(f"Columnas base ausentes en el DataFrame: {missing_cols}")
    print(f"✓ Todas las columnas base presentes ({len(expected_base_cols)})")

    # ── Enriquecimiento GSP-Spec (lote pequeño) ─────────────────
    all_ids = base_df["source_id"].tolist()
    try:
        gspspec_df = run_enrichment_batched(
            all_ids,
            cols=ENRICHMENT_COLS_GSPSPEC,
            table="gaiadr3.gaia_source",
            extra_filter="AND fem_gspspec IS NOT NULL",
            description="GSP-Spec (fem, alphafe, vrad)"
        )
        base_df = merge_with_enrichment(base_df, gspspec_df)
        print(f"✓ Merge GSP-Spec OK  — {base_df['fem_gspspec'].notna().sum()} objetos con fem_gspspec")
    except Exception as e:
        print(f"⚠️  Enriquecimiento GSP-Spec falló (no bloqueante): {e}")

    # ── Enriquecimiento ESP-ELS (lote pequeño) ──────────────────
    try:
        espels_df = run_enrichment_batched(
            all_ids,
            cols=ENRICHMENT_COLS_ESPELS,
            table="gaiadr3.astrophysical_parameters",
            extra_filter="AND ew_espels_halpha IS NOT NULL",
            description="ESP-ELS (EW Hα)"
        )
        base_df = merge_with_enrichment(base_df, espels_df)
        print(f"✓ Merge ESP-ELS OK   — {base_df['ew_espels_halpha'].notna().sum()} objetos con EW Hα")
    except Exception as e:
        print(f"⚠️  Enriquecimiento ESP-ELS falló (no bloqueante): {e}")

    # ── Verificar que no hay duplicados ─────────────────────────
    n_dupes = base_df.duplicated(subset="source_id").sum()
    assert n_dupes == 0, f"¡Hay {n_dupes} source_id duplicados tras el merge!"
    print(f"✓ Sin duplicados en source_id")

    # ── Reporte final del smoke test ────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SMOKE TEST PASADO")
    print(f"{'═'*60}")
    print(f"  Objetos descargados  : {len(base_df):,}")
    print(f"  Columnas totales     : {len(base_df.columns)}")
    _print_coverage(base_df)

    # Guardar resultado del smoke test para inspección manual
    out = "gaia_smoke_test_result.csv"
    base_df.to_csv(out, index=False)
    print(f"\n  Resultado guardado en: {out}")
    print(f"  Abre el CSV y verifica que los valores tienen sentido físico:")
    print(f"    · teff_gspphot entre 3000 y 40000 K")
    print(f"    · logg_gspphot entre 0 y 5.5")
    print(f"    · ruwe > 0")
    print(f"    · parallax_over_error > 5")

    return base_df


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL — descarga completa
# ─────────────────────────────────────────────────────────────

def _print_coverage(df: pd.DataFrame) -> None:
    """Imprime un resumen de cobertura de columnas clave."""
    cols_to_check = [
        ("fem_gspspec",      "Con fem_gspspec (RVS)"),
        ("alphafe_gspspec",  "Con alphafe_gspspec (RVS)"),
        ("radial_velocity",  "Con velocidad radial"),
        ("ew_espels_halpha", "Con EW Hα (ESP-ELS)"),
    ]
    for col, label in cols_to_check:
        if col in df.columns:
            n = df[col].notna().sum()
            pct = 100 * n / len(df) if len(df) > 0 else 0
            print(f"  {label:<30}: {n:>5,}  ({pct:.1f}%)")
    if "ruwe" in df.columns:
        n_ruwe = (df["ruwe"] > 1.4).sum()
        pct = 100 * n_ruwe / len(df) if len(df) > 0 else 0
        print(f"  {'RUWE > 1.4':<30}: {n_ruwe:>5,}  ({pct:.1f}%)")


def download_stratified_sample(galah_source_ids: list = None) -> pd.DataFrame:
    """
    Ejecuta la descarga completa de los 5 bloques y retorna un
    DataFrame unificado con todas las columnas.

    Args:
        galah_source_ids: Lista de source_id DR3 del catálogo GALAH.
                          Si es None, el Bloque 5 se omite con advertencia.

    Returns:
        DataFrame con ~5.000 filas y columnas de todos los agentes.
        Las columnas de enriquecimiento (GSP-Spec, ESP-ELS) pueden ser NaN.
    """
    print("=" * 60)
    print("  DESCARGA COMPLETA — muestra estratificada 5.000 objetos")
    print("=" * 60)

    blocks = []

    block_queries = [
        (QUERY_BLOCK1,  "Bloque 1 — Secuencia principal normal"),
        (QUERY_BLOCK2,  "Bloque 2 — Gigantes y subgigantes"),
        (QUERY_BLOCK3A, "Bloque 3A — Emisión Hα + tipo B/A"),
        (QUERY_BLOCK3B, "Bloque 3B — RUWE elevado"),
        (QUERY_BLOCK3C, "Bloque 3C — Metal-pobres del halo"),
        (QUERY_BLOCK3D, "Bloque 3D — Estrellas M activas"),
        (QUERY_BLOCK4,  "Bloque 4 — Alta extinción"),
    ]

    for adql, desc in block_queries:
        df = run_query(adql, desc)
        df["bloque"] = desc
        blocks.append(df)

    # ── Bloque 5: ground truth GALAH ───────────────────────────
    if galah_source_ids:
        q5 = build_block5_query(galah_source_ids)
        df5 = run_query(q5, "Bloque 5 — Ground truth GALAH")
        df5["bloque"] = "Bloque 5 — Ground truth GALAH"
        blocks.append(df5)
    else:
        print("\n⚠️  Bloque 5 omitido: no se proporcionaron source_ids de GALAH.")
        print("   Para incluirlo ejecuta con: --galah ruta/a/galah_crossmatch.csv")

    # ── Concatenar y deduplicar ─────────────────────────────────
    base_df = pd.concat(blocks, ignore_index=True)
    base_df = base_df.drop_duplicates(subset="source_id")
    print(f"\n✓ Muestra base unificada: {len(base_df):,} objetos únicos")

    # ── Enriquecimiento GSP-Spec (RVS) — en lotes ───────────────
    all_ids = base_df["source_id"].tolist()
    gspspec_df = run_enrichment_batched(
        all_ids,
        cols=ENRICHMENT_COLS_GSPSPEC,
        table="gaiadr3.gaia_source",
        extra_filter="AND fem_gspspec IS NOT NULL",
        description="GSP-Spec (fem, alphafe, vrad)"
    )
    base_df = merge_with_enrichment(base_df, gspspec_df)

    # ── Enriquecimiento ESP-ELS (Hα) — en lotes ─────────────────
    espels_df = run_enrichment_batched(
        all_ids,
        cols=ENRICHMENT_COLS_ESPELS,
        table="gaiadr3.astrophysical_parameters",
        extra_filter="AND ew_espels_halpha IS NOT NULL",
        description="ESP-ELS (EW Hα)"
    )
    base_df = merge_with_enrichment(base_df, espels_df)

    # ── Reporte final ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  DESCARGA COMPLETADA")
    print(f"{'═'*60}")
    print(f"  Total objetos únicos : {len(base_df):,}")
    print(f"  Columnas totales     : {len(base_df.columns)}")
    _print_coverage(base_df)
    print(f"\n  Distribución por bloque:")
    print(base_df["bloque"].value_counts().to_string())

    return base_df


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA — CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Descarga estratificada de Gaia DR3 para clasificación espectral MK."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Smoke test: descarga {SMOKE_N} objetos por bloque (~30s). "
             "No requiere nada externo."
    )
    parser.add_argument(
        "--galah",
        type=str,
        default=None,
        metavar="RUTA_CSV",
        help="CSV con columna 'source_id_dr3' para el Bloque 5 (ground truth GALAH). "
             "Si se omite, el Bloque 5 se salta."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="gaia_sample_stratified_5000",
        metavar="PREFIJO",
        help="Prefijo para los archivos de salida (default: gaia_sample_stratified_5000)."
    )
    args = parser.parse_args()

    if args.test:
        # ── MODO SMOKE TEST ─────────────────────────────────────
        df = smoke_test()

    else:
        # ── MODO DESCARGA COMPLETA ──────────────────────────────
        galah_ids = None
        if args.galah:
            print(f"Cargando source_ids GALAH desde: {args.galah}")
            galah_df = pd.read_csv(args.galah)
            galah_ids = galah_df["source_id_dr3"].dropna().astype(int).tolist()
            print(f"  → {len(galah_ids):,} IDs cargados")

        df = download_stratified_sample(galah_source_ids=galah_ids)

        # Guardar en CSV y Parquet
        csv_path     = f"{args.out}.csv"
        parquet_path = f"{args.out}.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)

        print(f"\n✓ Archivos guardados:")
        print(f"   {csv_path}")
        print(f"   {parquet_path}")

