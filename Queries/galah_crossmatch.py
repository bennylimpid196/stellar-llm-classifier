"""
Descarga GALAH DR3 desde VizieR y genera el crossmatch con Gaia DR3.

Produce un CSV con columna 'source_id_dr3' listo para pasarse a gaia_queries.py:

    python gaia_queries.py --galah galah_gaia_dr3_ids.csv

Pasos:
    1. Descarga GALAH DR3 de VizieR (catálogo J/MNRAS/506/150)
    2. Extrae los gaiadr2_source_id de las estrellas con clasificación fiable
    3. Hace el crossmatch DR2 → DR3 vía gaiadr3.dr2_neighbourhood en el TAP de ESA
    4. Guarda galah_gaia_dr3_ids.csv con los source_id DR3 resultantes

Requisitos:
    pip install astroquery astropy pandas pyarrow

Tiempo estimado: 5-15 minutos dependiendo del servidor de ESA.
"""

import time
import pandas as pd
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

# Catálogo GALAH DR3 en VizieR
GALAH_CATALOG   = "J/MNRAS/506/150"
GALAH_TABLE     = "J/MNRAS/506/150/table1"  # tabla principal

# Columnas de GALAH que necesitamos
# gaiadr2_source_id : ID de Gaia DR2 (para el crossmatch)
# flag_sp           : flag de calidad de parámetros estelares (0 = fiable)
# teff, logg, fe_h  : parámetros estelares para validación cruzada posterior
GALAH_COLS = [
    "sobject_id",
    "star_id",
    "gaiadr2_source_id",
    "flag_sp",
    "teff",
    "logg",
    "fe_h",
    "alpha_fe",
    "snr_c3",            # S/N en el canal 3 (zona del Hα)
]

# Filtros de calidad sobre GALAH
# flag_sp = 0  : parámetros estelares sin problemas conocidos
# snr_c3 > 30  : S/N suficiente para parámetros fiables
# fe_h no nulo : metalicidad medida (descartar espectros sin solución)
GALAH_QUALITY_FLAGS = {
    "flag_sp": 0,
    "snr_c3_min": 30,
}

# Número máximo de estrellas GALAH a descargar de VizieR
# GALAH DR3 tiene ~600.000 estrellas; nos quedamos con las de mayor calidad
GALAH_MAX_ROWS = 50000   # más que suficiente para obtener 1.200 tras filtros

# Número objetivo de source_ids DR3 para el Bloque 5
BLOCK5_TARGET = 1500     # pedimos 1.500 para tener margen; gaia_queries.py usa TOP 1200

# ─────────────────────────────────────────────────────────────
# PASO 1: DESCARGA GALAH DR3 DESDE VIZIER
# ─────────────────────────────────────────────────────────────

def download_galah_dr3() -> pd.DataFrame:
    """
    Descarga las estrellas de GALAH DR3 con flag_sp=0 y S/N > 30 desde VizieR.
    """
    print("=" * 60)
    print("  PASO 1 — Descargando GALAH DR3 desde VizieR")
    print("=" * 60)

    # Al poner columns=["**"] obligamos a VizieR a traer TODO sin filtrar,
    # lo que evita que falle si nos equivocamos en una letra del nombre.
    v = Vizier(columns=["**"], row_limit=GALAH_MAX_ROWS)

    print(f"  Conectando a VizieR — catálogo {GALAH_CATALOG}...")
    t0 = time.time()

    # Usamos el nombre del catálogo general, VizieR buscará la tabla principal
    result = v.get_catalogs(GALAH_CATALOG)

    if not result:
        raise RuntimeError("VizieR no devolvió resultados. El catálogo no existe o está caído.")

    # Tomamos la tabla principal (la primera que nos devuelva)
    main_table_name = list(result.keys())[0]
    df = result[main_table_name].to_pandas()
    elapsed = time.time() - t0
    print(f"  ✓ Tabla {main_table_name} descargada: {len(df):,} filas ({elapsed:.1f}s)")

    # VizieR altera los nombres de las columnas. Vamos a estandarizarlos:
    df.columns = df.columns.str.lower()

    # Diccionario de traducción de la nomenclatura de VizieR -> Nuestro código
    # Diccionario de traducción de la nomenclatura exacta de VizieR -> Nuestro código
    traduccion = {
        "gaiadr2": "gaiadr2_source_id",    
        "galah": "sobject_id",             
        "flagsp": "flag_sp",               
        "[fe/h]": "fe_h",                  
        "[alpha/fe]": "alpha_fe",          
        "snrc3iraf": "snr_c3",             
    }
    df = df.rename(columns=traduccion)

    # Validar que las columnas que necesitamos realmente existan ahora
    cols_requeridas = ["gaiadr2_source_id", "flag_sp", "teff", "logg", "fe_h", "snr_c3"]
    faltantes = [c for c in cols_requeridas if c not in df.columns]
    if faltantes:
        print("\n⚠️ ERROR: Faltan columnas. VizieR las llamó de otra forma.")
        print("Columnas que llegaron:", df.columns.tolist())
        raise KeyError(f"Faltan las columnas: {faltantes}")

    # Aplicar filtros de calidad
    print(f"\n  Aplicando filtros de calidad:")
    print(f"    Antes: {len(df):,} estrellas")

    df = df[df["flag_sp"] == GALAH_QUALITY_FLAGS["flag_sp"]]
    print(f"    Tras flag_sp=0: {len(df):,} estrellas")

    df = df[df["snr_c3"] >= GALAH_QUALITY_FLAGS["snr_c3_min"]]
    print(f"    Tras snr_c3>={GALAH_QUALITY_FLAGS['snr_c3_min']}: {len(df):,} estrellas")

    df = df[df["fe_h"].notna()]
    print(f"    Tras fe_h IS NOT NULL: {len(df):,} estrellas")

    df = df[df["gaiadr2_source_id"].notna()]
    print(f"    Tras gaiadr2_source_id IS NOT NULL: {len(df):,} estrellas")

    df["gaiadr2_source_id"] = df["gaiadr2_source_id"].astype("int64")

    print(f"\n  ✓ Muestra GALAH de calidad: {len(df):,} estrellas")
    return df

# ─────────────────────────────────────────────────────────────
# PASO 2: CROSSMATCH DR2 → DR3 VÍA TAP DE ESA
# ─────────────────────────────────────────────────────────────

def crossmatch_dr2_to_dr3(dr2_ids: list, batch_size: int = 500) -> pd.DataFrame:
    """
    Convierte una lista de source_id de Gaia DR2 a source_id de Gaia DR3
    usando la tabla gaiadr3.dr2_neighbourhood del TAP de ESA.

    La tabla dr2_neighbourhood contiene el crossmatch oficial entre DR2 y DR3
    calculado por el equipo de Gaia. Para cada DR2 puede haber varios candidatos
    DR3; nos quedamos con el más cercano angularmente (angular_distance mínima).

    Args:
        dr2_ids   : Lista de integers con los gaiadr2_source_id.
        batch_size: Tamaño de lote para las queries TAP (evita strings muy largos).

    Returns:
        DataFrame con columnas [dr2_source_id, dr3_source_id, angular_distance].
    """
    print("\n" + "=" * 60)
    print("  PASO 2 — Crossmatch DR2 → DR3 vía TAP de ESA")
    print("=" * 60)

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1

    n_batches = max(1, (len(dr2_ids) + batch_size - 1) // batch_size)
    print(f"  {len(dr2_ids):,} IDs DR2 en {n_batches} lotes de {batch_size}")

    parts = []
    for i in range(0, len(dr2_ids), batch_size):
        batch = dr2_ids[i : i + batch_size]
        ids_str = ", ".join(str(x) for x in batch)

        adql = f"""
        SELECT
            n.dr3_source_id,
            n.dr2_source_id,
            n.angular_distance
        FROM gaiadr3.dr2_neighbourhood AS n
        WHERE
            n.dr2_source_id IN ({ids_str})
            AND n.angular_distance < 0.5
        ORDER BY
            n.dr2_source_id, n.angular_distance ASC
        """

        lote_num = i // batch_size + 1
        print(f"  Lote {lote_num}/{n_batches}...", end="", flush=True)
        t0 = time.time()

        try:
            job = Gaia.launch_job_async(adql, verbose=False)
            df_batch = job.get_results().to_pandas()
        except Exception as e:
            print(f" ⚠ Async falló: {e}. Reintentando síncrono...")
            time.sleep(5)
            job = Gaia.launch_job(adql, verbose=False)
            df_batch = job.get_results().to_pandas()

        elapsed = time.time() - t0
        parts.append(df_batch)
        print(f" {len(df_batch):,} matches  ({elapsed:.1f}s)")

    all_matches = pd.concat(parts, ignore_index=True)

    # Quedarse con el match más cercano por DR2 ID (mínima angular_distance)
    best_matches = (
        all_matches
        .sort_values("angular_distance")
        .drop_duplicates(subset="dr2_source_id", keep="first")
        .rename(columns={
            "dr3_source_id": "source_id_dr3",
            "dr2_source_id": "source_id_dr2"
        })
    )

    print(f"\n  ✓ Crossmatch completo: {len(best_matches):,} pares DR2→DR3 únicos")
    print(f"    (de {len(dr2_ids):,} IDs DR2 de entrada)")
    lost = len(dr2_ids) - len(best_matches)
    if lost > 0:
        print(f"    ⚠ {lost:,} IDs DR2 sin match en DR3 "
              f"(fuentes no re-observadas o fuera del footprint DR3)")

    return best_matches


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def build_galah_id_list(output_csv: str = "galah_gaia_dr3_ids.csv") -> pd.DataFrame:
    """
    Ejecuta los dos pasos y guarda el resultado.

    Args:
        output_csv: Ruta del archivo de salida.

    Returns:
        DataFrame con columnas [source_id_dr3, source_id_dr2, angular_distance,
        sobject_id, teff, logg, fe_h, alpha_fe, snr_c3].
        Listo para usar con: python gaia_queries.py --galah galah_gaia_dr3_ids.csv
    """
    # ── Paso 1: GALAH DR3 ───────────────────────────────────────
    galah_df = download_galah_dr3()

    # Tomar una muestra representativa si hay más de BLOCK5_TARGET
    # Estratificar por tipo espectral usando teff como proxy
    if len(galah_df) > BLOCK5_TARGET:
        print(f"\n  Muestreando {BLOCK5_TARGET:,} estrellas de {len(galah_df):,} disponibles...")

        # Bins de temperatura aproximados a los tipos espectrales
        bins = [0, 3700, 5200, 6000, 7500, 10000, 100000]
        labels = ["M", "K", "G", "F", "A", "OB"]
        galah_df["tipo_approx"] = pd.cut(galah_df["teff"], bins=bins, labels=labels)

        # Muestreo proporcional por tipo
        sample = galah_df.groupby("tipo_approx", observed=True).apply(
            lambda g: g.sample(
                n=min(len(g), max(1, int(BLOCK5_TARGET * len(g) / len(galah_df)))),
                random_state=42
            )
        ).reset_index(drop=True)

        # Completar hasta BLOCK5_TARGET si el muestreo quedó corto
        if len(sample) < BLOCK5_TARGET:
            remaining = galah_df[~galah_df["sobject_id"].isin(sample["sobject_id"])]
            extra = remaining.sample(
                n=min(BLOCK5_TARGET - len(sample), len(remaining)),
                random_state=42
            )
            sample = pd.concat([sample, extra], ignore_index=True)

        galah_df = sample.head(BLOCK5_TARGET)
        print(f"  ✓ Muestra final: {len(galah_df):,} estrellas")
        print(f"  Distribución por tipo aproximado:")
        print(galah_df["tipo_approx"].value_counts().sort_index().to_string())

    # ── Paso 2: Crossmatch DR2 → DR3 ───────────────────────────
    dr2_ids = galah_df["gaiadr2_source_id"].tolist()
    crossmatch_df = crossmatch_dr2_to_dr3(dr2_ids)

    # ── Merge con parámetros GALAH ──────────────────────────────
    final_df = crossmatch_df.merge(
        galah_df[["gaiadr2_source_id", "sobject_id", "teff", "logg",
                  "fe_h", "alpha_fe", "snr_c3"]],
        left_on="source_id_dr2",
        right_on="gaiadr2_source_id",
        how="left"
    ).drop(columns=["gaiadr2_source_id"])

    # ── Guardar ─────────────────────────────────────────────────
    final_df.to_csv(output_csv, index=False)
    print(f"\n{'═'*60}")
    print(f"  GALAH CROSSMATCH COMPLETADO")
    print(f"{'═'*60}")
    print(f"  Archivo guardado: {output_csv}")
    print(f"  Filas            : {len(final_df):,}")
    print(f"  Columnas         : {list(final_df.columns)}")
    print(f"\n  Siguiente paso:")
    print(f"  python gaia_queries.py --galah {output_csv}")

    return final_df


if __name__ == "__main__":
    build_galah_id_list()
