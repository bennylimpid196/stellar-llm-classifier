# -*- coding: utf-8 -*-
"""
Gaia DR3 — Unified Dataset Builder
====================================
Single-script pipeline that performs, in order:
    1. Downloads a balanced stellar catalog from the Gaia DR3 archive.
    2. Rebalances the catalog by spectral category to user-defined targets.
    3. Downloads BP/RP spectra (via GaiaXPy) for all final stars.
    4. Downloads RVS spectra (via DataLink) for all final stars.
    5. Saves everything to a versioned output folder.

Output structure:
    /home/cesar/Documentos/Tesis-cimat/Estancia/data/raw/
    └── DB-{N}-{YYYY-MM-DD}/
        ├── catalog.csv               — final balanced catalog (tabular metadata)
        ├── spectra_bprp.npy          — BP/RP flux matrix  (shape: n_stars × n_wavelengths)
        ├── spectra_bprp_ids.npy      — source_id array aligned with BP/RP matrix rows
        ├── sampling_bprp.npy         — shared wavelength grid in nm (shape: n_wavelengths,)
        ├── run_manifest.json         — run metadata: date, counts, parameters
        └── rvs/
            └── {source_id}.npz       — per-star RVS: wavelength_nm, flux, flux_error

Each run always starts from scratch (no dependency on previous DB-X folders).

Usage:
    python gaia_dataset_builder.py --total 500

    Optional flags:
        --total        Total number of stars in the final dataset  [default: 500]
        --rvs-batch    Max stars per DataLink RVS session           [default: 20]
        --bprp-batch   Max stars per GaiaXPy BP/RP batch           [default: 50]
        --output-root  Override the default output root directory
"""

import argparse
import json
import logging
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from gaiaxpy import calibrate

warnings.filterwarnings("ignore", module="astropy.io.votable")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_OUTPUT_ROOT = Path("/home/cesar/Documentos/Tesis-cimat/Estancia/data/raw")

# Gaia G-band extinction coefficient (Wang & Chen 2019)
K_G_EXTINCTION = 2.74

# Shared ADQL field list used by all category queries
_ADQL_FIELDS = """
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
    a.ew_espels_halpha_flag,
    nss.nss_solution_type AS nss_solution_type
"""

_ADQL_JOINS = """
FROM gaiadr3.gaia_source AS s
JOIN gaiadr3.astrophysical_parameters AS a
    ON s.source_id = a.source_id
LEFT JOIN gaiadr3.nss_two_body_orbit AS nss
    ON s.source_id = nss.source_id
"""

# Per-category ADQL WHERE clauses (appended to the shared base conditions)
_CATEGORY_FILTERS: Dict[str, str] = {
    "kg_enanas": """
        AND a.teff_gspphot BETWEEN 4000 AND 6500
        AND a.logg_gspphot >= 3.5
        AND a.mh_gspphot >= -1.0
    """,
    "af_sp": """
        AND a.teff_gspphot BETWEEN 6500 AND 10000
        AND a.logg_gspphot >= 3.5
    """,
    "kg_gigantes": """
        AND a.teff_gspphot BETWEEN 4000 AND 6500
        AND a.logg_gspphot < 3.5
        AND a.mh_gspphot >= -1.0
    """,
    "b_calientes": """
        AND a.teff_gspphot > 10000
    """,
    "m_frias": """
        AND a.teff_gspphot < 4000
    """,
    "halo": """
        AND a.mh_gspphot < -1.0
    """,
}


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gaia DR3 Unified Dataset Builder"
    )
    parser.add_argument(
        "--total", type=int, default=500,
        help="Total number of stars in the final balanced dataset (default: 500)."
    )
    parser.add_argument(
        "--rvs-batch", type=int, default=20,
        help="Max stars per DataLink RVS download session (default: 20)."
    )
    parser.add_argument(
        "--bprp-batch", type=int, default=50,
        help="Max stars per GaiaXPy BP/RP calibration batch (default: 50)."
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for output folders (default: {DEFAULT_OUTPUT_ROOT})."
    )
    return parser.parse_args()


# =============================================================================
# OUTPUT FOLDER MANAGEMENT
# =============================================================================

def resolve_output_folder(root: Path) -> Path:
    """
    Creates and returns the next versioned DB folder.
    Scans existing DB-N-* directories to determine N+1.
    """
    root.mkdir(parents=True, exist_ok=True)

    existing = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("DB-")]
    max_n = 0
    for d in existing:
        parts = d.name.split("-")
        if len(parts) >= 2 and parts[1].isdigit():
            max_n = max(max_n, int(parts[1]))

    run_date = date.today().isoformat()
    folder_name = f"DB-{max_n + 1}-{run_date}"
    output_folder = root / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    rvs_folder = output_folder / "rvs"
    rvs_folder.mkdir(exist_ok=True)

    logger.info(f"Output folder: {output_folder}")
    return output_folder


# =============================================================================
# CATEGORY DISTRIBUTION
# =============================================================================

def compute_target_distribution(total: int) -> Dict[str, int]:
    """
    Computes per-category star counts that sum to `total`.
    Proportions are fixed; the remainder is added to the largest category.

    Base proportions (from scientific justification):
        kg_enanas   30%   — dominant main-sequence population in Gaia RVS
        af_sp       20%   — intermediate-temperature coverage
        kg_gigantes 16%   — important for luminosity class diagnostics
        b_calientes 12%   — hot star coverage
        m_frias     12%   — cool star / M-dwarf coverage
        halo        10%   — metal-poor / kinematic outlier population
    """
    proportions = {
        "kg_enanas":   0.30,
        "af_sp":       0.20,
        "kg_gigantes": 0.16,
        "b_calientes": 0.12,
        "m_frias":     0.12,
        "halo":        0.10,
    }
    targets = {cat: int(total * p) for cat, p in proportions.items()}

    # Distribute rounding remainder to the largest category
    remainder = total - sum(targets.values())
    targets["kg_enanas"] += remainder

    logger.info("Target distribution:")
    for cat, n in targets.items():
        logger.info(f"  {cat:15s}: {n}")
    logger.info(f"  {'TOTAL':15s}: {sum(targets.values())}")

    return targets


def classify_stars(df: pd.DataFrame) -> pd.Series:
    """
    Assigns a single category label to each row using priority ordering.
    Priority (highest wins): halo > m_frias > kg_gigantes > kg_enanas > af_sp > b_calientes
    """
    t = df["teff_gspphot"]
    g = df["logg_gspphot"]
    m = df["mh_gspphot"]

    cats = pd.Series("otro", index=df.index)
    cats[t >= 10000]                               = "b_calientes"
    cats[(t >= 6500) & (t < 10000) & (g >= 3.5)]  = "af_sp"
    cats[(t >= 4000) & (t < 6500)  & (g >= 3.5)]  = "kg_enanas"
    cats[(t >= 4000) & (t < 6500)  & (g < 3.5)]   = "kg_gigantes"
    cats[t < 4000]                                  = "m_frias"
    cats[m < -1.0]                                  = "halo"  # Overrides all
    return cats


# =============================================================================
# ADQL QUERY CONSTRUCTION AND EXECUTION
# =============================================================================

def _build_exclusion_clause(ids: List[int]) -> str:
    """Builds a safe SQL NOT IN clause, capped at 500 IDs to avoid query limits."""
    if not ids:
        return ""
    ids_str = ", ".join(str(i) for i in ids[:500])
    return f"AND s.source_id NOT IN ({ids_str})"


def _build_category_query(category: str, n: int, ids_to_exclude: List[int]) -> str:
    exclusion = _build_exclusion_clause(ids_to_exclude)
    base_conditions = f"""
    WHERE s.has_rvs = 'True'
      AND s.has_xp_continuous = 'True'
      AND s.parallax IS NOT NULL
      AND s.parallax_error IS NOT NULL
      AND s.ruwe IS NOT NULL
      AND a.teff_gspphot IS NOT NULL
      AND a.logg_gspphot IS NOT NULL
      {exclusion}
    """
    return (
        f"SELECT TOP {n}\n{_ADQL_FIELDS}\n"
        f"{_ADQL_JOINS}\n"
        f"{base_conditions}\n"
        f"{_CATEGORY_FILTERS[category]}\n"
        f"ORDER BY s.phot_g_mean_mag ASC"
    )


def _execute_query(query: str, description: str) -> Optional[pd.DataFrame]:
    """Submits an async ADQL job to the Gaia archive and returns a DataFrame."""
    try:
        logger.info(f"Submitting query: {description}")
        job = Gaia.launch_job_async(query)
        df = job.get_results().to_pandas()
        logger.info(f"  -> {len(df)} rows returned.")
        return df
    except Exception as e:
        logger.error(f"Query failed [{description}]: {e}")
        return None


# =============================================================================
# STEP 1 — CATALOG DOWNLOAD AND REBALANCING
# =============================================================================

def build_balanced_catalog(targets: Dict[str, int]) -> pd.DataFrame:
    """
    Downloads and assembles a balanced catalog from scratch.

    Strategy per category:
        - Requests (target * 1.2) candidates to absorb any post-download
          classification drift (Teff boundary stars may shift category).
        - Trims to exactly `target` stars sorted by brightness (best SNR first).
        - Uses a cumulative exclusion list so no source_id appears twice.
    """
    accumulated_ids: List[int] = []
    category_frames: Dict[str, pd.DataFrame] = {}

    print("\n" + "=" * 60)
    print("STEP 1: Downloading balanced catalog from Gaia DR3")
    print("=" * 60)

    for category, target in targets.items():
        # Request 20% extra to account for boundary reclassification
        n_request = int(target * 1.2)
        logger.info(f"Category '{category}': requesting {n_request} candidates for {target} target.")

        query = _build_category_query(category, n_request, accumulated_ids)
        df_cat = _execute_query(query, f"category={category}")

        if df_cat is None or len(df_cat) == 0:
            logger.warning(f"No data returned for category '{category}'. Skipping.")
            category_frames[category] = pd.DataFrame()
            continue

        # Defensive casting
        df_cat["source_id"] = df_cat["source_id"].astype("int64")
        if "rv_nb_transits" in df_cat.columns:
            df_cat["rv_nb_transits"] = pd.to_numeric(
                df_cat["rv_nb_transits"], errors="coerce"
            )

        # Normalize nss_solution_type: empty strings / 'nan' strings → None
        if "nss_solution_type" in df_cat.columns:
            df_cat["nss_solution_type"] = df_cat["nss_solution_type"].apply(
                lambda v: None if pd.isna(v) or str(v).strip().lower() in
                          ("", "nan", "none", "null") else v
            )

        # Trim to target (brightest first — already sorted by phot_g_mean_mag ASC)
        df_cat = df_cat.head(target)

        category_frames[category] = df_cat
        accumulated_ids += df_cat["source_id"].tolist()
        logger.info(f"  -> Kept {len(df_cat)} stars for '{category}'.")

    # Concatenate all categories
    all_frames = [f for f in category_frames.values() if len(f) > 0]
    if not all_frames:
        logger.error("No categories downloaded successfully. Aborting.")
        return pd.DataFrame()

    catalog = pd.concat(all_frames, ignore_index=True)
    catalog = catalog.drop_duplicates(subset="source_id")

    # Attach category labels for diagnostics (dropped before final save)
    catalog["_category"] = classify_stars(catalog)

    logger.info(f"\nCatalog assembled: {len(catalog)} stars total.")
    _print_distribution(catalog, targets)

    return catalog


def _print_distribution(df: pd.DataFrame, targets: Dict[str, int]):
    counts = df["_category"].value_counts()
    print("\nDistribution summary:")
    for cat, target in targets.items():
        actual = counts.get(cat, 0)
        status = "OK" if actual >= int(target * 0.9) else "LOW"
        print(f"  [{status}] {cat:15s}: {actual:4d} / {target:4d}")
    print(f"  {'TOTAL':20s}: {len(df):4d}")


# =============================================================================
# STEP 2 — BP/RP SPECTRA DOWNLOAD
# =============================================================================

def download_bprp_spectra(
    source_ids: List[int],
    batch_size: int,
    output_folder: Path
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Downloads and calibrates BP/RP spectra for all source_ids via GaiaXPy.
    Processes in batches to avoid server timeouts.

    Returns:
        flux_matrix   : np.ndarray shape (n_stars, n_wavelengths), NaN if missing
        ids_array     : np.ndarray shape (n_stars,) — aligned with flux_matrix rows
        sampling_nm   : np.ndarray shape (n_wavelengths,) — shared wavelength grid
    """
    print("\n" + "=" * 60)
    print("STEP 2: Downloading BP/RP spectra (GaiaXPy)")
    print("=" * 60)

    all_spectra: List[pd.DataFrame] = []
    sampling_ref: Optional[np.ndarray] = None

    for i in range(0, len(source_ids), batch_size):
        batch = source_ids[i: i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(source_ids) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} stars)...", end=" ", flush=True)

        try:
            spectra_df, sampling = calibrate(batch, save_file=False)
            if sampling_ref is None:
                sampling_ref = np.array(sampling)
            all_spectra.append(spectra_df)
            print(f"OK — {len(spectra_df)} spectra, "
                  f"{sampling[0]:.0f}–{sampling[-1]:.0f} nm")
        except ValueError as e:
            print(f"no spectra available: {e}")
        except Exception as e:
            print(f"error: {e}")

    if not all_spectra or sampling_ref is None:
        logger.error("No BP/RP spectra downloaded.")
        return None, None, None

    combined = pd.concat(all_spectra, ignore_index=True)
    combined["source_id"] = combined["source_id"].astype("int64")

    # Build aligned matrix: one row per source_id in the input list
    n_wave = len(sampling_ref)
    flux_matrix = np.full((len(source_ids), n_wave), np.nan)
    ids_array = np.array(source_ids, dtype=np.int64)

    id_to_idx = {sid: idx for idx, sid in enumerate(source_ids)}
    for _, row in combined.iterrows():
        sid = int(row["source_id"])
        if sid in id_to_idx:
            flux_matrix[id_to_idx[sid]] = np.array(row["flux"])

    n_found = int(np.sum(~np.all(np.isnan(flux_matrix), axis=1)))
    logger.info(f"BP/RP spectra obtained: {n_found} / {len(source_ids)}")

    # Persist to disk
    np.save(output_folder / "spectra_bprp.npy", flux_matrix)
    np.save(output_folder / "spectra_bprp_ids.npy", ids_array)
    np.save(output_folder / "sampling_bprp.npy", sampling_ref)
    logger.info("BP/RP arrays saved.")

    return flux_matrix, ids_array, sampling_ref


# =============================================================================
# STEP 3 — RVS SPECTRA DOWNLOAD
# =============================================================================

def download_rvs_spectra(
    source_ids: List[int],
    rvs_folder: Path
) -> Dict[int, bool]:
    """
    Downloads individual RVS spectra via Gaia DataLink for each source_id.
    Each spectrum is saved as {source_id}.npz containing:
        wavelength_nm, flux, flux_error (NaN array if not available).

    Returns:
        availability: Dict mapping source_id -> True (downloaded) / False (not available)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Downloading RVS spectra (DataLink)")
    print("=" * 60)

    availability: Dict[int, bool] = {}

    for i, sid in enumerate(source_ids):
        print(f"  [{i + 1}/{len(source_ids)}] {sid}...", end=" ", flush=True)
        try:
            datalink = Gaia.load_data(
                ids=[sid],
                data_release="Gaia DR3",
                retrieval_type="RVS",
                data_structure="INDIVIDUAL",
                format="votable"
            )

            if not datalink:
                print("no RVS data.")
                availability[sid] = False
                continue

            key = list(datalink.keys())[0]
            table = datalink[key][0].to_table()

            if "wavelength" not in table.columns or "flux" not in table.columns:
                print(f"unexpected columns: {list(table.columns)}")
                availability[sid] = False
                continue

            wave = np.array(table["wavelength"])
            flux = np.array(table["flux"])
            flux_err = (
                np.array(table["flux_error"])
                if "flux_error" in table.columns
                else np.full_like(flux, np.nan)
            )

            np.savez_compressed(
                rvs_folder / f"{sid}.npz",
                wavelength_nm=wave,
                flux=flux,
                flux_error=flux_err
            )
            availability[sid] = True
            print(f"OK — {len(wave)} pts, {wave.min():.1f}–{wave.max():.1f} nm")

        except Exception as e:
            print(f"error: {e}")
            availability[sid] = False

    n_ok = sum(availability.values())
    logger.info(f"RVS spectra saved: {n_ok} / {len(source_ids)}")
    return availability


# =============================================================================
# STEP 4 — FINALIZE CATALOG AND WRITE MANIFEST
# =============================================================================

def finalize_and_save(
    catalog: pd.DataFrame,
    bprp_availability: Optional[np.ndarray],
    rvs_availability: Dict[int, bool],
    ids_array: Optional[np.ndarray],
    output_folder: Path,
    args: argparse.Namespace
):
    """
    Attaches spectrum availability flags to the catalog and saves:
        - catalog.csv          (final tabular data)
        - run_manifest.json    (run metadata for reproducibility)
    """
    df = catalog.copy()

    # Attach RVS availability flag
    df["has_rvs_spectrum"] = df["source_id"].apply(
        lambda sid: rvs_availability.get(int(sid), False)
    )

    # Attach BP/RP availability flag
    if ids_array is not None:
        bprp_set = set(
            int(ids_array[i])
            for i in range(len(ids_array))
            if not np.all(np.isnan(
                # Check if the row in the matrix is not all NaN
                # We re-load from disk to avoid passing the full matrix here
                np.load(output_folder / "spectra_bprp.npy", mmap_mode="r")[i]
            ))
        )
        df["has_bprp_spectrum"] = df["source_id"].apply(
            lambda sid: int(sid) in bprp_set
        )
    else:
        df["has_bprp_spectrum"] = False

    # Drop internal classification column
    df.drop(columns=["_category"], errors="ignore", inplace=True)

    # Save catalog
    catalog_path = output_folder / "catalog.csv"
    df.to_csv(catalog_path, index=False)
    logger.info(f"Catalog saved: {catalog_path} ({len(df)} stars)")

    # Build and save manifest
    manifest = {
        "run_date": date.today().isoformat(),
        "output_folder": str(output_folder),
        "parameters": {
            "total_requested": args.total,
            "bprp_batch_size": args.bprp_batch,
            "rvs_batch_size": args.rvs_batch,
        },
        "results": {
            "total_stars": len(df),
            "stars_with_bprp": int(df["has_bprp_spectrum"].sum()),
            "stars_with_rvs": int(df["has_rvs_spectrum"].sum()),
            "stars_with_both": int(
                (df["has_bprp_spectrum"] & df["has_rvs_spectrum"]).sum()
            ),
        },
        "category_distribution": classify_stars(df).value_counts().to_dict(),
        "nan_percentages": {
            col: round(float(df[col].isna().mean() * 100), 1)
            for col in df.columns
            if df[col].isna().any()
        },
    }

    manifest_path = output_folder / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved: {manifest_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    for key, value in manifest["results"].items():
        print(f"  {key:30s}: {value}")
    print(f"\n  Output folder: {output_folder}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Resolve versioned output folder
    output_folder = resolve_output_folder(args.output_root)
    rvs_folder = output_folder / "rvs"

    # Compute per-category targets from total
    targets = compute_target_distribution(args.total)

    # Step 1: Build balanced catalog
    catalog = build_balanced_catalog(targets)
    if catalog.empty:
        logger.error("Catalog is empty. Aborting.")
        return

    source_ids = catalog["source_id"].astype(int).tolist()

    # Step 2: BP/RP spectra for all final stars
    flux_matrix, ids_array, sampling = download_bprp_spectra(
        source_ids=source_ids,
        batch_size=args.bprp_batch,
        output_folder=output_folder
    )

    # Step 3: RVS spectra for all final stars
    rvs_availability = download_rvs_spectra(
        source_ids=source_ids,
        rvs_folder=rvs_folder
    )

    # Step 4: Finalize catalog with availability flags and write manifest
    finalize_and_save(
        catalog=catalog,
        bprp_availability=flux_matrix,
        rvs_availability=rvs_availability,
        ids_array=ids_array,
        output_folder=output_folder,
        args=args
    )


if __name__ == "__main__":
    main()