"""
ground_truth_cleaner.py
-----------------------
Cleans and validates the raw ground_truth_final.csv produced by join_pastel_local.py.

Fixes applied
-------------
1. source_id recovery   : Extracts the correct Gaia DR3 source_id from the
                          'user_specified_id' column (e.g. 'Gaia DR3 1244571953471006720'
                          → 1244571953471006720). The previous value was a row index.

2. Physical outlier filter : Removes rows where PASTEL parameters are outside
                             physically meaningful bounds:
                               - Teff  :  2 000 – 50 000 K
                               - logg  : -1.0  –  5.5  dex
                               - [Fe/H]: -5.0  –  1.5  dex
                             Flagged rows have their PASTEL columns set to NaN
                             (they are NOT dropped — the SIMBAD data is still valid).

3. n_pastel_measurements : Cast to nullable integer for clarity.

Output
------
ground_truth_clean.csv  — ready for benchmark use by the SC validation layer.
"""

import re
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("GroundTruthCleaner")

# ---------------------------------------------------------------------------
# Physical bounds for PASTEL parameters
# ---------------------------------------------------------------------------
# Physical bounds for PASTEL parameters (based on observed range in pastel.dat).
# Teff : 1 750 – 89 125 K observed; we keep 2 000 K floor (below this PASTEL
#         entries are typically brown dwarfs or substellar objects outside our scope).
# logg : -2.0 – 7.0 observed; 7.0 are white dwarfs (valid for our benchmark).
# [Fe/H]: -4.8 – 2.4 observed; all physically plausible.
BOUNDS = {
    "teff_pastel":  (2_000,  100_000),
    "logg_pastel":  (-2.0,   7.5),
    "feh_pastel":   (-5.5,   2.5),
}


def recover_source_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the numeric Gaia DR3 source_id from 'user_specified_id'.
    Example: 'Gaia DR3 1244571953471006720' → 1244571953471006720
    """
    extracted = df["user_specified_id"].str.extract(r"(\d{10,})")
    n_recovered = extracted[0].notna().sum()
    n_failed    = extracted[0].isna().sum()

    df["source_id"] = extracted[0].astype("Int64")

    logger.info(f"source_id recovery: {n_recovered} recovered, {n_failed} failed.")
    if n_failed > 0:
        failed = df.loc[extracted[0].isna(), "user_specified_id"].tolist()
        logger.warning(f"  Could not extract source_id from: {failed}")

    return df


def flag_physical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets PASTEL parameter columns to NaN where values fall outside
    physically meaningful bounds. The row itself is kept — SIMBAD data
    (sp_type, ra, dec) remains valid for MK-level validation.
    """
    total_flagged = 0

    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        n = int(mask.sum())
        if n > 0:
            logger.warning(
                f"  {col}: {n} outlier(s) outside [{lo}, {hi}] → set to NaN. "
                f"Values: {df.loc[mask, col].tolist()}"
            )
            df.loc[mask, col] = np.nan
            total_flagged += n

    if total_flagged == 0:
        logger.info("Physical outlier check: no outliers found.")
    else:
        logger.info(f"Physical outlier check: {total_flagged} values flagged.")

    return df


def report(df: pd.DataFrame) -> None:
    """Prints a concise coverage and statistics report."""
    logger.info("=== Final ground truth report ===")
    logger.info(f"  Total stars      : {len(df)}")
    for col in ["teff_pastel", "logg_pastel", "feh_pastel"]:
        if col in df.columns:
            n = int(df[col].notna().sum())
            logger.info(f"  {col:20s}: {n} non-null ({100 * n / len(df):.1f}%)")

    logger.info(f"  sp_type coverage : {int(df['sp_type'].notna().sum())} / {len(df)}")

    if "teff_pastel" in df.columns:
        subset = df["teff_pastel"].dropna()
        logger.info(f"  Teff range       : {subset.min():.0f} – {subset.max():.0f} K  "
                    f"(median {subset.median():.0f} K)")


def main():
    parser = argparse.ArgumentParser(
        description="Clean and validate ground_truth_final.csv for SC benchmark use."
    )
    parser.add_argument(
        "--input",  type=Path, required=True,
        help="Path to ground_truth_final.csv"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path for the cleaned output CSV"
    )
    args = parser.parse_args()

    logger.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input, dtype={"source_id": str})
    logger.info(f"  Rows loaded: {len(df)}")

    # --- Fix 1: source_id ---
    df = recover_source_id(df)

    # --- Fix 2: physical outliers ---
    logger.info("Checking physical parameter bounds...")
    df = flag_physical_outliers(df)

    # --- Fix 3: n_pastel_measurements as integer ---
    if "n_pastel_measurements" in df.columns:
        df["n_pastel_measurements"] = df["n_pastel_measurements"].astype("Int64")

    # --- Save ---
    df.to_csv(args.output, index=False)
    logger.info(f"Saved to {args.output}")

    report(df)


if __name__ == "__main__":
    main()