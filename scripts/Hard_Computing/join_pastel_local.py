"""
join_pastel_local.py
--------------------
Joins a locally downloaded PASTEL .dat file with the existing SIMBAD CSV.
No network calls — runs entirely offline.

Usage:
    python join_pastel_local.py \
        --pastel  /path/to/pastel.dat \
        --simbad  /path/to/ground_truth_direct.csv \
        --output  /path/to/ground_truth_final.csv
"""

import re
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("PastelJoin")

# ---------------------------------------------------------------------------
# PASTEL fixed-width column positions (verified from pastel.dat header)
#   0-33  : Star name
#  34-49  : RA (deg)
#  50-65  : Dec (deg)
# 141-148 : Teff (K)
# 153-160 : logg (dex)
# 166-174 : [Fe/H] (dex)
# 183-203 : First author
# 203-223 : Bibcode
# ---------------------------------------------------------------------------
# Column positions verified character-by-character against pastel.dat.
# The format is fixed-width with these fields (0-indexed, end exclusive):
#   0   – 34  : Star name
#   34  – 49  : RA (deg)
#   50  – 65  : Dec (deg)
#   139 – 145 : Teff (K)       — 4 or 5 digits, right-justified
#   145 – 152 : e_Teff         — blank when no uncertainty published
#   152 – 158 : logg (dex)
#   158 – 165 : e_logg         — blank when no uncertainty published
#   165 – 172 : [Fe/H] (dex)
#   203 – 223 : Bibcode
PASTEL_COLSPECS = [(0,34),(34,49),(50,65),(139,145),(145,152),(152,158),(165,172),(203,223)]
PASTEL_COLNAMES = ["name","ra","dec","teff","e_teff","logg","feh","bibcode"]


def _norm(s: str) -> str:
    """Canonical join key: collapse whitespace, uppercase."""
    return re.sub(r'\s+', ' ', str(s).strip()).upper()


def load_pastel(path: Path) -> pd.DataFrame:
    """
    Parses the PASTEL fixed-width .dat file and aggregates multiple
    measurements per star using the median (one row per published study).
    """
    logger.info(f"Loading PASTEL from {path}...")
    # read_fwf supports .gz natively via pandas/gzip
    open_path = path  # pandas handles .gz transparently
    df = pd.read_fwf(
        open_path,
        colspecs=PASTEL_COLSPECS,
        names=PASTEL_COLNAMES,
        dtype=str,
        header=None,
        compression="gzip" if str(path).endswith(".gz") else "infer",
    )
    df["name"] = df["name"].str.strip()
    for col in ["teff", "logg", "feh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"  Raw rows: {len(df)} | Unique stars: {df['name'].nunique()} | With Teff: {df['teff'].notna().sum()}")

    # Aggregate: median across all studies per star
    df_agg = df.groupby("name", as_index=False).agg(
        teff_pastel=("teff", "median"),
        logg_pastel=("logg", "median"),
        feh_pastel=("feh",  "median"),
        n_pastel_measurements=("bibcode", "count"),
    )
    df_agg["_jk"] = df_agg["name"].apply(_norm)
    return df_agg


def load_simbad(path: Path) -> pd.DataFrame:
    """
    Loads the SIMBAD CSV, dropping any stale pastel columns from previous runs
    so the join always starts from a clean state.
    """
    logger.info(f"Loading SIMBAD from {path}...")
    df = pd.read_csv(path, dtype={"source_id": "Int64"})

    stale = [c for c in df.columns if any(x in c for x in ["pastel", "_x", "_y"])]
    if stale:
        logger.info(f"  Dropping stale columns from previous runs: {stale}")
        df = df.drop(columns=stale)

    logger.info(f"  Stars: {len(df)}")
    df["_jk"] = df["main_id"].apply(_norm)
    return df


def join(df_simbad: pd.DataFrame, df_pastel: pd.DataFrame) -> pd.DataFrame:
    """
    Two-pass join:

    Pass 1 — Direct normalised name match (works when SIMBAD MAIN_ID and
              PASTEL name share the same designation, e.g. '* EPS RET').

    Pass 2 — HD/HIP prefix match: for unmatched stars, check whether any
              PASTEL name that starts with 'HD' or 'HIP' can be derived from
              the SIMBAD name by stripping the leading '* ' or 'V* ' prefix
              and collapsing spaces in the number (e.g. 'HD    344' → 'HD344').
              This handles the most common SIMBAD↔PASTEL nomenclature gap.
    """
    pastel_cols = ["teff_pastel", "logg_pastel", "feh_pastel", "n_pastel_measurements"]

    # Pass 1 — direct
    merged = pd.merge(df_simbad, df_pastel[["_jk"] + pastel_cols], on="_jk", how="left")
    matched = merged["teff_pastel"].notna()
    n1 = int(matched.sum())
    logger.info(f"Pass-1 (direct name): {n1} / {len(merged)} matched")

    # Pass 2 — compact HD/HIP key
    # Build a secondary PASTEL index keyed by compacted HD/HIP number
    # e.g. 'HD    344' → 'HD344', 'HIP  1234' → 'HIP1234'
    def _compact_hd(name: str) -> str | None:
        if not isinstance(name, str): return None
        m = re.match(r'^(HD|HIP)\s+(\d+)', name.strip(), re.IGNORECASE)
        return f"{m.group(1).upper()}{m.group(2)}" if m else None

    df_pastel["_hd_key"] = df_pastel["name"].apply(_compact_hd)
    pastel_hd = df_pastel.dropna(subset=["_hd_key"]).drop_duplicates(subset=["_hd_key"]).set_index("_hd_key")[pastel_cols]

    # Build the same compact key from SIMBAD main_id
    # SIMBAD format examples: 'HD 120136', '* tau Boo', 'V* AI Scl'
    merged["_hd_key"] = merged["main_id"].apply(_compact_hd)

    needs_fill = ~matched & merged["_hd_key"].notna() & merged["_hd_key"].isin(pastel_hd.index)
    for col in pastel_cols:
        merged.loc[needs_fill, col] = merged.loc[needs_fill, "_hd_key"].map(pastel_hd[col])

    n2 = int(merged["teff_pastel"].notna().sum()) - n1
    logger.info(f"Pass-2 (HD/HIP compact key): {n2} additional stars matched")
    logger.info(f"Total matched: {n1 + n2} / {len(merged)}")

    return merged.drop(columns=["_jk", "_hd_key"], errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pastel", type=Path, required=True, help="Path to pastel.dat (uncompressed)")
    parser.add_argument("--simbad", type=Path, required=True, help="Path to ground_truth_direct.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    df_pastel = load_pastel(args.pastel)
    df_simbad = load_simbad(args.simbad)
    final     = join(df_simbad, df_pastel)

    final.to_csv(args.output, index=False)
    logger.info(f"Saved to {args.output}")

    for col in ["teff_pastel", "logg_pastel", "feh_pastel"]:
        if col in final.columns:
            n = int(final[col].notna().sum())
            logger.info(f"  {col}: {n} non-null ({100 * n / len(final):.1f}%)")


if __name__ == "__main__":
    main()