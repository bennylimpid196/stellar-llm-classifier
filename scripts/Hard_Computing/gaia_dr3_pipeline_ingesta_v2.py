# -*- coding: utf-8 -*-
"""
Gaia DR3 — Pipeline de Ingesta de Datos (v2 — Corregido)
==========================================================
Corrección crítica respecto a v1:
    - nss_solution_type NO existe en gaiadr3.gaia_source.
      Se obtiene mediante LEFT JOIN con gaiadr3.nss_two_body_orbit.
      El LEFT JOIN preserva estrellas simples (sin solución NSS) como NULL.

Confirmación de convención de signo (TEST 7 del diagnóstico):
    - ew_espels_halpha = +0.022 → absorción (positivo = absorción en DR3)
    - El BinaryDetectorAgent debe usar: has_emission = (ew_espels_halpha < 0)
"""

import warnings
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Desactivar LaTeX renderer

from astroquery.gaia import Gaia
from gaiaxpy import calibrate

warnings.filterwarnings("ignore", module='astropy.io.votable')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# QUERY MAESTRA CORREGIDA
# Cambios respecto a v1:
#   1. nss_solution_type: eliminado de gaia_source, obtenido via LEFT JOIN
#      con gaiadr3.nss_two_body_orbit (tabla oficial NSS de Gaia DR3).
#   2. launch_job_async en lugar de launch_job (confirmado funcional en TEST 12).
#   3. ORDER BY mantenido (confirmado funcional en TEST 10).
# =============================================================================

QUERY_MAESTRA = """
SELECT TOP 500
    -- Identificación
    s.source_id,
    s.ra,
    s.dec,

    -- Astrometría (AstrometryAgent)
    -- ADVERTENCIA: parallax puede ser <= 0 para fuentes lejanas/ruidosas.
    s.parallax,
    s.parallax_error,
    s.pmra,
    s.pmdec,

    -- Fotometría (AstrometryAgent — necesaria para calcular M_G)
    s.phot_g_mean_mag,

    -- Extinción (AstrometryAgent — ADVERTENCIA: prone a NaN para estrellas frías)
    s.ebpminrp_gspphot,

    -- Calidad astrométrica (BinaryDetectorAgent)
    s.ruwe,

    -- Variabilidad RV espectroscópica (BinaryDetectorAgent)
    s.radial_velocity_error,
    s.rv_nb_transits,

    -- Binariedad oficial NSS (BinaryDetectorAgent)
    -- Obtenido via LEFT JOIN con gaiadr3.nss_two_body_orbit.
    -- NULL si la estrella NO tiene solución NSS (estrella simple).
    -- Valores posibles: 'Orbital', 'AstroSpectroSB1', 'SB2', etc.
    nss.nss_solution_type AS nss_solution_type,

    -- Parámetros físicos GSP-Phot (AstrometryAgent + BinaryDetectorAgent)
    a.teff_gspphot,
    a.logg_gspphot,
    a.mh_gspphot,

    -- Química GSP-Spec (passthrough directo al contrato SC)
    -- Disponible solo para estrellas con RVS de suficiente SNR (~3M fuentes).
    a.alphafe_gspspec,
    a.fem_gspspec,

    -- H-alpha ESP-ELS (LineAgent)
    -- CONVENCIÓN DE SIGNO CONFIRMADA en Gaia DR3:
    --   Valores POSITIVOS -> absorción (mayoría de estrellas SP)
    --   Valores NEGATIVOS -> emisión real (estrellas activas, Be, jóvenes)
    -- Usar solo si ew_espels_halpha_flag = '0' (calidad OK)
    a.ew_espels_halpha,
    a.ew_espels_halpha_flag

FROM gaiadr3.gaia_source AS s

-- JOIN interno: solo estrellas con parámetros astrofísicos
JOIN gaiadr3.astrophysical_parameters AS a
    ON s.source_id = a.source_id

-- LEFT JOIN: preserva estrellas simples (sin NSS) como NULL en nss_solution_type
LEFT JOIN gaiadr3.nss_two_body_orbit AS nss
    ON s.source_id = nss.source_id

WHERE
    s.has_rvs = 'True'
    AND s.parallax IS NOT NULL
    AND s.parallax_error IS NOT NULL
    AND s.ruwe IS NOT NULL
    AND a.teff_gspphot IS NOT NULL

ORDER BY s.phot_g_mean_mag ASC
"""

# Versión alternativa si nss_two_body_orbit no tiene la columna solution_id
# (usar si la query principal falla con error de columna desconocida)
QUERY_MAESTRA_SIN_NSS = """
SELECT TOP 500
    s.source_id, s.ra, s.dec,
    s.parallax, s.parallax_error,
    s.pmra, s.pmdec,
    s.phot_g_mean_mag,
    s.ebpminrp_gspphot,
    s.ruwe,
    s.radial_velocity_error,
    s.rv_nb_transits,
    a.teff_gspphot,
    a.logg_gspphot,
    a.mh_gspphot,
    a.alphafe_gspspec,
    a.fem_gspspec,
    a.ew_espels_halpha,
    a.ew_espels_halpha_flag
FROM gaiadr3.gaia_source AS s
JOIN gaiadr3.astrophysical_parameters AS a
    ON s.source_id = a.source_id
WHERE s.has_rvs = 'True'
  AND s.parallax IS NOT NULL
  AND s.parallax_error IS NOT NULL
  AND s.ruwe IS NOT NULL
  AND a.teff_gspphot IS NOT NULL
ORDER BY s.phot_g_mean_mag ASC
"""


def verificar_tabla_nss():
    """
    Verifica qué columnas exactas tiene gaiadr3.nss_two_body_orbit.
    Ejecutar si la query maestra falla para confirmar el nombre de columna correcto.
    """
    print("Verificando columnas de gaiadr3.nss_two_body_orbit...")
    try:
        job = Gaia.launch_job("""
            SELECT column_name, datatype
            FROM tap_schema.columns
            WHERE table_name = 'gaiadr3.nss_two_body_orbit'
            ORDER BY column_name
        """)
        df = job.get_results().to_pandas()
        print(df.to_string(index=False))
        return df
    except Exception as e:
        logger.error(f"No se pudo consultar tap_schema: {e}")
        return None


# =============================================================================
# DESCARGA DEL CATÁLOGO
# =============================================================================

def descargar_catalogo(query: str = None,
                       output_csv: str = "catalogo_maestro_gaia.csv") -> pd.DataFrame:
    """
    Lanza la query maestra al archivo Gaia DR3 en modo asíncrono y guarda el resultado.
    Si la query con NSS falla, intenta automáticamente la versión sin NSS.

    Returns:
        DataFrame con los metadatos tabulares listos para los módulos HC.
    """
    if query is None:
        query = QUERY_MAESTRA

    print("=" * 60)
    print("PASO 1: Descargando Catálogo Maestro Gaia DR3")
    print("=" * 60)

    # Intento 1: Query con NSS
    df = _ejecutar_query(query, "Query maestra (con NSS)")

    # Fallback: Query sin NSS si la primera falla
    if df is None:
        logger.warning(
            "Query con NSS falló. Intentando versión sin nss_solution_type. "
            "El BinaryDetectorAgent usará solo RUWE y RV para binariedad."
        )
        df = _ejecutar_query(QUERY_MAESTRA_SIN_NSS, "Query maestra (sin NSS — fallback)")
        if df is not None:
            # Añadir columna vacía para mantener el contrato de datos consistente
            df['nss_solution_type'] = None

    if df is None or len(df) == 0:
        logger.error("No se obtuvieron datos. Revisa la conexión o relaja los filtros.")
        return pd.DataFrame()

    # Casting defensivo de rv_nb_transits (llega como float, el agente necesita int)
    if 'rv_nb_transits' in df.columns:
        df['rv_nb_transits'] = pd.to_numeric(df['rv_nb_transits'], errors='coerce')

    df.to_csv(output_csv, index=False)
    logger.info(f"Catálogo guardado: {output_csv} ({len(df)} estrellas)")

    _imprimir_resumen(df)
    return df


def _ejecutar_query(query: str, descripcion: str) -> pd.DataFrame:
    """Ejecuta una query ADQL en modo async y retorna DataFrame o None si falla."""
    try:
        logger.info(f"Ejecutando: {descripcion}")
        job = Gaia.launch_job_async(query)
        df = job.get_results().to_pandas()
        logger.info(f"OK: {len(df)} filas descargadas.")
        return df
    except Exception as e:
        logger.error(f"Fallo en '{descripcion}': {e}")
        return None


def _imprimir_resumen(df: pd.DataFrame):
    """Imprime vista previa y estadísticas de NaN del catálogo descargado."""
    cols_preview = [
        'source_id', 'phot_g_mean_mag', 'parallax', 'ruwe',
        'teff_gspphot', 'mh_gspphot',
        'ew_espels_halpha', 'ew_espels_halpha_flag', 'nss_solution_type'
    ]
    cols_disponibles = [c for c in cols_preview if c in df.columns]

    print(f"\nVista previa (5 filas):")
    print(df[cols_disponibles].head(5).to_string(index=False))

    print(f"\nPorcentaje de NaN por columna (solo las que tienen NaN):")
    nan_pct = (df.isna().sum() / len(df) * 100).round(1)
    nan_pct = nan_pct[nan_pct > 0]
    if len(nan_pct) > 0:
        print(nan_pct.to_string())
    else:
        print("  Ninguna columna tiene NaN.")

    # Reporte de binarias NSS
    if 'nss_solution_type' in df.columns:
        n_nss = df['nss_solution_type'].notna().sum()
        print(f"\nEstrellas con solución NSS (binarias confirmadas): {n_nss} / {len(df)}")
        if n_nss > 0:
            print(df['nss_solution_type'].value_counts().to_string())


# =============================================================================
# DESCARGA DE ESPECTROS BP/RP
# =============================================================================

def descargar_espectros_bprp(source_ids: list,
                              max_batch: int = 50) -> tuple:
    """
    Descarga y calibra espectros BP/RP en flujo absoluto [W/m²/nm].
    Procesa en lotes para evitar timeouts del servidor.

    Returns:
        (spectra_df, sampling_nm): DataFrame con flujos y array de longitudes
        de onda en nm. (None, None) si falla completamente.
    """
    print("=" * 60)
    print("PASO 2: Descargando Espectros BP/RP (GaiaXPy)")
    print("=" * 60)

    todos = []
    sampling_ref = None

    for i in range(0, len(source_ids), max_batch):
        lote = source_ids[i:i + max_batch]
        n_lote = i // max_batch + 1
        print(f"  Lote {n_lote} ({len(lote)} estrellas)...", end=" ", flush=True)
        try:
            spectra, sampling = calibrate(lote, save_file=False)
            if sampling_ref is None:
                sampling_ref = sampling
            todos.append(spectra)
            print(f"OK — {len(spectra)} espectros, "
                  f"{sampling[0]:.0f}–{sampling[-1]:.0f} nm, "
                  f"{len(sampling)} puntos.")
        except ValueError as e:
            print(f"sin espectros: {e}")
        except Exception as e:
            print(f"error: {e}")

    if not todos:
        logger.error("No se descargó ningún espectro BP/RP.")
        return None, None

    spectra_df = pd.concat(todos, ignore_index=True)
    logger.info(f"Total BP/RP: {len(spectra_df)} espectros.")
    return spectra_df, sampling_ref


# =============================================================================
# DESCARGA DE ESPECTROS RVS
# =============================================================================

def descargar_espectros_rvs(source_ids: list) -> dict:
    """
    Descarga espectros RVS vía DataLink para el LineAgent (Ca II Triplete).
    Aplica fallback estrella a estrella porque no todos los source_id
    tienen RVS individual disponible vía DataLink público.

    Returns:
        Dict {source_id: {'wavelength_nm', 'flux', 'flux_error'}}
    """
    print("=" * 60)
    print("PASO 3: Descargando Espectros RVS (DataLink)")
    print("=" * 60)

    espectros = {}

    for i, sid in enumerate(source_ids):
        print(f"  [{i+1}/{len(source_ids)}] {sid}...", end=" ", flush=True)
        try:
            datalink = Gaia.load_data(
                ids=[sid],
                data_release='Gaia DR3',
                retrieval_type='RVS',
                data_structure='INDIVIDUAL',
                format='votable'
            )

            if not datalink:
                print("sin datos RVS.")
                continue

            clave = list(datalink.keys())[0]
            tabla = datalink[clave][0].to_table()

            if 'wavelength' not in tabla.columns or 'flux' not in tabla.columns:
                print(f"columnas inesperadas: {list(tabla.columns)}")
                continue

            wave = np.array(tabla['wavelength'])
            flux = np.array(tabla['flux'])
            flux_err = np.array(tabla['flux_error']) if 'flux_error' in tabla.columns else None

            espectros[sid] = {
                'wavelength_nm': wave,
                'flux':          flux,
                'flux_error':    flux_err
            }
            print(f"OK — {len(tabla)} pts, "
                  f"{wave.min():.1f}–{wave.max():.1f} nm")

        except Exception as e:
            print(f"error: {e}")

    logger.info(f"RVS descargados: {len(espectros)} / {len(source_ids)}")
    return espectros


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def visualizar_espectro_bprp(spectra_df: pd.DataFrame,
                              sampling: np.ndarray,
                              idx: int = 0):
    """Grafica un espectro BP/RP calibrado con líneas relevantes marcadas."""
    if spectra_df is None or len(spectra_df) == 0:
        logger.warning("No hay espectros BP/RP para visualizar.")
        return

    sid  = spectra_df.iloc[idx]['source_id']
    flux = np.array(spectra_df.iloc[idx]['flux'])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sampling, flux, color='black', linewidth=0.8, label=f'ID: {sid}')
    ax.axvspan(330,  680,  color='royalblue', alpha=0.08, label='BP (330–680 nm)')
    ax.axvspan(640, 1050,  color='tomato',    alpha=0.08, label='RP (640–1050 nm)')

    lineas = {
        'H-α 656.3':  656.3,
        'Ca II 849.8': 849.8,
        'Ca II 854.2': 854.2,
        'Ca II 866.2': 866.2,
    }
    for nombre, lam in lineas.items():
        if sampling[0] <= lam <= sampling[-1]:
            ax.axvline(lam, color='green', linestyle='--',
                       linewidth=0.8, alpha=0.8, label=nombre)

    ax.set_title(f'Espectro BP/RP Gaia DR3 — ID: {sid}')
    ax.set_xlabel('Longitud de Onda (nm)')
    ax.set_ylabel('Flujo [W/m^2/nm]')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualizar_espectro_rvs(espectros_rvs: dict, source_id: int):
    """Grafica un espectro RVS con el Ca II triplete marcado."""
    if source_id not in espectros_rvs:
        logger.warning(f"source_id {source_id} sin espectro RVS.")
        return

    datos = espectros_rvs[source_id]
    wave  = datos['wavelength_nm']
    flux  = datos['flux']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave, flux, color='navy', linewidth=0.9)

    for lam, label in [(849.8, 'Ca II 849.8'),
                       (854.2, 'Ca II 854.2'),
                       (866.2, 'Ca II 866.2')]:
        ax.axvline(lam, color='red', linestyle='--', linewidth=0.8, label=label)

    ax.set_title(f'Espectro RVS Gaia DR3 — ID: {source_id}')
    ax.set_xlabel('Longitud de Onda (nm)')
    ax.set_ylabel('Flujo')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# EMPAQUETADO PARA EL ORQUESTADOR HC
# =============================================================================

def preparar_paquete_hc(fila_catalogo: pd.Series,
                         wavelength_bprp: np.ndarray,
                         flux_bprp: np.ndarray,
                         wavelength_rvs: np.ndarray = None,
                         flux_rvs: np.ndarray = None) -> dict:
    """
    Convierte una fila del catálogo y sus espectros en el paquete
    estándar que espera HC_PipelineOrchestrator.process_source().
    """
    metadata = fila_catalogo.to_dict()

    # source_id puede llegar como float desde el DataFrame — el orquestador lo necesita int
    if 'source_id' in metadata and metadata['source_id'] is not None:
        metadata['source_id'] = int(metadata['source_id'])

    # Normalizar nss_solution_type: strings vacíos o 'nan' → None
    nss = metadata.get('nss_solution_type')
    if nss is not None and str(nss).strip().lower() in ('', 'nan', 'none', 'null'):
        metadata['nss_solution_type'] = None

    return {
        'metadata':        metadata,
        'wavelength_bprp': wavelength_bprp,
        'flux_bprp':       flux_bprp,
        'wavelength_rvs':  wavelength_rvs,
        'flux_rvs':        flux_rvs,
    }


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # --- PASO 1: Catálogo ---
    catalogo_df = descargar_catalogo(output_csv="catalogo_maestro_gaia.csv")

    if len(catalogo_df) == 0:
        logger.error("Sin datos. Abortando.")
        exit(1)

    source_ids = catalogo_df['source_id'].tolist()

    # --- PASO 2: Espectros BP/RP ---
    spectra_bprp, sampling_bprp = descargar_espectros_bprp(
        source_ids=source_ids,
        max_batch=50
    )

    # --- PASO 3: Espectros RVS (limitado a las primeras 20 para no saturar DataLink) ---
    espectros_rvs = descargar_espectros_rvs(source_ids=source_ids[:20])

    # --- PASO 4: Visualización ---
    if spectra_bprp is not None and len(spectra_bprp) > 0:
        visualizar_espectro_bprp(spectra_bprp, sampling_bprp, idx=0)

    if espectros_rvs:
        visualizar_espectro_rvs(espectros_rvs, list(espectros_rvs.keys())[0])

    # --- PASO 5: Reporte final y ejemplo de empaquetado ---
    ids_bprp = set(spectra_bprp['source_id'].tolist()) if spectra_bprp is not None else set()
    ids_rvs  = set(espectros_rvs.keys())
    ids_completos = ids_bprp & ids_rvs

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"  Estrellas en catálogo:          {len(catalogo_df)}")
    print(f"  Con espectro BP/RP:             {len(ids_bprp)}")
    print(f"  Con espectro RVS:               {len(ids_rvs)}")
    print(f"  Con ambos (pipeline completo):  {len(ids_completos)}")

    if ids_completos:
        sid_ejemplo = list(ids_completos)[0]
        fila = catalogo_df[catalogo_df['source_id'] == sid_ejemplo].iloc[0]
        idx_bprp = spectra_bprp[spectra_bprp['source_id'] == sid_ejemplo].index[0]

        paquete = preparar_paquete_hc(
            fila_catalogo=fila,
            wavelength_bprp=np.array(sampling_bprp),
            flux_bprp=np.array(spectra_bprp.iloc[idx_bprp]['flux']),
            wavelength_rvs=espectros_rvs[sid_ejemplo]['wavelength_nm'],
            flux_rvs=espectros_rvs[sid_ejemplo]['flux'],
        )
        print(f"\n  Paquete HC listo para source_id: {sid_ejemplo}")
        print(f"  Campos metadata: {list(paquete['metadata'].keys())}")
        print(f"  Puntos BP/RP:    {len(paquete['flux_bprp'])}")
        print(f"  Puntos RVS:      {len(paquete['flux_rvs'])}")
        print("\n  Listo para HC_PipelineOrchestrator.process_source()")
