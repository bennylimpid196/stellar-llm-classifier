import pandas as pd
import logging
import time
from pathlib import Path
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("DirectValidator")


class DirectValidationFetcher:
    def __init__(self, db_folder: Path):
        self.db_folder    = db_folder
        self.catalog_path = db_folder / "catalog.csv"
        self.output_dir   = db_folder / "validation"
        self.output_dir.mkdir(exist_ok=True)
        self.output_path  = self.output_dir / "ground_truth_direct.csv"

    # ------------------------------------------------------------------
    # Retry helper (fixes the 4 timeout-lost blocks from v1)
    # ------------------------------------------------------------------

    @staticmethod
    def _retry(fn, retries: int = 4, backoff: float = 15.0):
        """Exponential-backoff retry. Returns None after all attempts fail."""
        wait = backoff
        for attempt in range(1, retries + 1):
            try:
                return fn()
            except Exception as e:
                if attempt == retries:
                    logger.error(f"All {retries} attempts failed: {e}")
                    return None
                logger.warning(f"Attempt {attempt} failed ({e}). Retrying in {wait:.0f}s...")
                time.sleep(wait)
                wait *= 2

    # ------------------------------------------------------------------
    # PASTEL batch query (same working method as v1)
    # ------------------------------------------------------------------

    def _fetch_pastel_batch(self, names: list) -> pd.DataFrame:
        """
        Queries PASTEL for a batch of star names via query_constraints().
        Identical to v1 — this is the confirmed-working method for B/pastel.
        Fix applied: wrapped in _retry() so timeouts don't silently drop blocks.
        """
        if not names:
            return pd.DataFrame()

        clean_names = [str(n).strip() for n in names if str(n).strip()]
        if not clean_names:
            return pd.DataFrame()

        def _query():
            v = Vizier(columns=["**", "Teff", "logg", "[Fe/H]"], row_limit=-1)
            return v.query_constraints(
                catalog="B/pastel/pastel",
                Source=",".join(clean_names)
            )

        result = self._retry(_query)
        if result is None or len(result) == 0:
            return pd.DataFrame()

        try:
            df = result[0].to_pandas()
            id_cols = [c for c in df.columns if c.lower() in ["source", "_source", "id", "main_id"]]
            if not id_cols:
                logger.warning(f"No identifier column found. Columns: {df.columns.tolist()}")
                return pd.DataFrame()

            df = df.rename(columns={
                id_cols[0]:  "main_id",
                "Teff":      "teff_pastel",
                "logg":      "logg_pastel",
                "[Fe/H]":    "feh_pastel",
                "__Fe_H_":   "feh_pastel",
            })

            keep = [c for c in ["main_id", "teff_pastel", "logg_pastel", "feh_pastel"] if c in df.columns]
            return df[keep]

        except Exception as e:
            logger.warning(f"Error parsing PASTEL batch result: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, chunk_size: int = 50):

        # --- STEP 1: Load or fetch SIMBAD data ---
        df_simbad = pd.DataFrame()

        if self.output_path.exists():
            logger.info("Loading existing SIMBAD data from CSV...")
            df_simbad = pd.read_csv(self.output_path)

        if df_simbad.empty or "main_id" not in df_simbad.columns:
            logger.info("Starting SIMBAD phase (no existing data)...")
            df_local   = pd.read_csv(self.catalog_path)
            source_ids = df_local["source_id"].dropna().unique().tolist()
            all_simbad = []

            total_blocks = int(np.ceil(len(source_ids) / chunk_size))
            for i in range(0, len(source_ids), chunk_size):
                chunk = source_ids[i : i + chunk_size]
                logger.info(f"SIMBAD block {i // chunk_size + 1} / {total_blocks}...")

                def _simbad_query(c=chunk):
                    Simbad.add_votable_fields("sp_type")
                    return Simbad.query_objects([f"Gaia DR3 {sid}" for sid in c])

                res = self._retry(_simbad_query)
                if res is not None:
                    temp_df = res.to_pandas()
                    temp_df["source_id"] = (
                        temp_df["user_specified_id"].str.extract(r"(\d+)").astype(np.int64)
                    )
                    all_simbad.append(temp_df)
                time.sleep(0.5)

            df_simbad = pd.concat(all_simbad, ignore_index=True)
            df_simbad = df_simbad.rename(columns={
                "MAIN_ID":  "main_id",
                "SP_TYPE":  "spectral_type_simbad",
            })
            df_simbad.to_csv(self.output_path, index=False)

        # --- STEP 2: PASTEL phase ---
        logger.info("Querying PASTEL in batches using SIMBAD names...")
        star_names = df_simbad["main_id"].dropna().unique().tolist()
        all_pastel = []

        total_blocks = int(np.ceil(len(star_names) / chunk_size))
        for i in range(0, len(star_names), chunk_size):
            name_chunk = star_names[i : i + chunk_size]
            logger.info(f"PASTEL block {i // chunk_size + 1} / {total_blocks}...")
            p_df = self._fetch_pastel_batch(name_chunk)
            if not p_df.empty:
                all_pastel.append(p_df)
            time.sleep(1.0)

        # --- STEP 3: Aggregate + merge ---
        if not all_pastel:
            logger.warning("PASTEL returned no data. Output contains only SIMBAD data.")
            return

        df_pastel = pd.concat(all_pastel, ignore_index=True)

        # Normalised join key
        df_simbad["_jk"] = df_simbad["main_id"].astype(str).str.replace(" ", "").str.upper()
        df_pastel["_jk"] = df_pastel["main_id"].astype(str).str.replace(" ", "").str.upper()

        # FIX (v1 bug): aggregate multiple PASTEL measurements per star using
        # the median instead of drop_duplicates (which discarded all but the
        # first row, losing information from other published studies).
        numeric_pastel = [c for c in ["teff_pastel", "logg_pastel", "feh_pastel"] if c in df_pastel.columns]
        df_pastel_agg  = df_pastel.groupby("_jk", as_index=False)[numeric_pastel].median()

        n_before = df_simbad["_jk"].isin(df_pastel_agg["_jk"]).sum()
        logger.info(f"PASTEL unique stars after aggregation: {len(df_pastel_agg)} "
                    f"(direct match with SIMBAD: {n_before})")

        final = pd.merge(df_simbad, df_pastel_agg, on="_jk", how="left").drop(columns=["_jk"])
        final.to_csv(self.output_path, index=False)

        n_matched = int(final["teff_pastel"].notna().sum()) if "teff_pastel" in final.columns else 0
        logger.info(f"Done. {n_matched} / {len(final)} stars have PASTEL parameters.")
        for col in ["teff_pastel", "logg_pastel", "feh_pastel"]:
            if col in final.columns:
                k = int(final[col].notna().sum())
                logger.info(f"  {col}: {k} non-null ({100 * k / len(final):.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-folder", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    DirectValidationFetcher(args.db_folder).run(chunk_size=args.chunk_size)