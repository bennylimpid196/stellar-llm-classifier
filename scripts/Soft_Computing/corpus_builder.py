"""
corpus_builder.py — SC Corpus Builder
======================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0

Reads the 498 HC contracts (hc_contracts.json) and the ground truth table
(ground_truth_final.csv), merges them by source_id, and writes a single
stellar_corpus.json ready to be consumed by inference_manager.py.

Each element in the output corpus is a self-contained inference unit:
  - The full HC contract (physical_vector, logical_flags, spectral_summary, etc.)
  - Ground truth fields attached under the key "ground_truth" (for post-hoc
    validation by validator.py — the LLM does NOT see these fields).

Design decisions:
  - source_id is always treated as string to avoid int64 overflow in JSON.
  - ground_truth.source_id is extracted from user_specified_id via regex,
    because the source_id column in ground_truth_final.csv is a pipeline
    artifact (constant value = 3).
  - NaN values in numeric ground truth fields are serialized as null (JSON).
  - Contracts with quality_score == 0.0 are included but flagged; the
    inference_manager decides whether to skip them.
  - No contract is silently dropped. A per-contract validation report is
    written alongside the corpus.

Outputs:
  <output_dir>/stellar_corpus.json        — Main corpus (list of dicts)
  <output_dir>/corpus_build_report.json   — Validation report

Usage:
    python3 corpus_builder.py \\
        --contracts   /path/to/hc_contracts.json \\
        --ground-truth /path/to/ground_truth_final.csv \\
        --output      /path/to/outputs/

Author: Hybrid Stellar Classifier Project / CIMAT
Version: 1.0
"""

import json
import re
import math
import logging
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="corpus_builder.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Also stream to console so the user sees progress without tailing the log.
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_console)


# ---------------------------------------------------------------------------
# Ground truth parser
# ---------------------------------------------------------------------------

_GAIA_ID_PATTERN = re.compile(r"(\d{15,20})")


def _extract_source_id(user_specified_id: str) -> Optional[str]:
    """
    Extracts the numeric Gaia DR3 source_id from strings of the form
    'Gaia DR3 1244571953471006720'.

    Returns the ID as a string, or None if no match is found.
    """
    if not isinstance(user_specified_id, str):
        return None
    match = _GAIA_ID_PATTERN.search(user_specified_id)
    return match.group(1) if match else None


def _safe_float(value) -> Optional[float]:
    """Converts a value to float, returning None for NaN or non-finite."""
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    """Converts a value to int, returning None for NaN."""
    try:
        f = float(value)
        return None if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CorpusBuilder class
# ---------------------------------------------------------------------------

class CorpusBuilder:
    """
    Merges HC contracts with ground truth metadata and produces the
    stellar_corpus.json consumed by inference_manager.py.

    Attributes:
        contracts_path  (Path): Path to hc_contracts.json.
        ground_truth_path (Path): Path to ground_truth_final.csv.
        output_dir      (Path): Directory for output files.
    """

    def __init__(
        self,
        contracts_path: Path,
        ground_truth_path: Path,
        output_dir: Path,
    ):
        self.contracts_path    = contracts_path
        self.ground_truth_path = ground_truth_path
        self.output_dir        = output_dir

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_contracts(self) -> list[dict]:
        log.info(f"Loading HC contracts from: {self.contracts_path}")
        with open(self.contracts_path, "r", encoding="utf-8") as f:
            contracts = json.load(f)
        if not isinstance(contracts, list):
            raise ValueError("hc_contracts.json must be a JSON array at the root level.")
        log.info(f"  Loaded {len(contracts)} contracts.")
        return contracts

    def _load_ground_truth(self) -> dict[str, dict]:
        """
        Loads ground_truth_final.csv and returns a dict keyed by source_id
        (string). Extracts the numeric source_id from the user_specified_id
        column because the source_id column itself is a pipeline artifact.
        """
        log.info(f"Loading ground truth from: {self.ground_truth_path}")
        gt_df = pd.read_csv(self.ground_truth_path)
        log.info(f"  Raw rows: {len(gt_df)}")

        gt_df["_sid"] = gt_df["user_specified_id"].apply(_extract_source_id)

        missing = gt_df["_sid"].isna().sum()
        if missing > 0:
            log.warning(f"  {missing} rows could not extract a source_id from user_specified_id.")

        gt_df = gt_df.dropna(subset=["_sid"])

        gt_map: dict[str, dict] = {}
        for _, row in gt_df.iterrows():
            sid = row["_sid"]
            gt_map[sid] = {
                "main_id":              row.get("main_id"),
                "sp_type":              row.get("sp_type") if pd.notna(row.get("sp_type")) else None,
                "teff_pastel":          _safe_float(row.get("teff_pastel")),
                "logg_pastel":          _safe_float(row.get("logg_pastel")),
                "feh_pastel":           _safe_float(row.get("feh_pastel")),
                "n_pastel_measurements": _safe_int(row.get("n_pastel_measurements")),
            }

        log.info(f"  Ground truth entries indexed: {len(gt_map)}")
        return gt_map

    # ------------------------------------------------------------------
    # Contract validation
    # ------------------------------------------------------------------

    REQUIRED_TOP_LEVEL = {
        "source_id", "pipeline_version", "physical_vector",
        "logical_flags", "spectral_summary", "quality_score",
    }

    REQUIRED_PHYSICAL_VECTOR = {
        "abs_mag", "teff_k", "metallicity", "fe_h",
        "alpha_fe", "logg", "v_tan", "extinction_ag",
    }

    REQUIRED_LOGICAL_FLAGS = {
        "is_reliable_parallax", "is_giant", "is_metal_poor",
        "is_binary_candidate", "is_high_velocity", "has_emission",
    }

    def _validate_contract(self, contract: dict) -> list[str]:
        """
        Validates a single HC contract for structural completeness.

        Returns a list of warning strings (empty list = contract is valid).
        """
        warnings: list[str] = []

        missing_top = self.REQUIRED_TOP_LEVEL - set(contract.keys())
        if missing_top:
            warnings.append(f"Missing top-level keys: {sorted(missing_top)}")

        pv = contract.get("physical_vector", {})
        missing_pv = self.REQUIRED_PHYSICAL_VECTOR - set(pv.keys())
        if missing_pv:
            warnings.append(f"Missing physical_vector keys: {sorted(missing_pv)}")

        # NaN check inside physical_vector
        nan_pv = [k for k, v in pv.items() if isinstance(v, float) and math.isnan(v)]
        if nan_pv:
            warnings.append(f"NaN values in physical_vector: {nan_pv}")

        lf = contract.get("logical_flags", {})
        missing_lf = self.REQUIRED_LOGICAL_FLAGS - set(lf.keys())
        if missing_lf:
            warnings.append(f"Missing logical_flags keys: {sorted(missing_lf)}")

        qs = contract.get("quality_score")
        if qs is not None and qs == 0.0:
            warnings.append("quality_score == 0.0 (parallax unreliable — SC inference unreliable).")

        return warnings

    # ------------------------------------------------------------------
    # Corpus assembly
    # ------------------------------------------------------------------

    def _assemble_entry(
        self,
        contract: dict,
        gt_map: dict[str, dict],
    ) -> dict:
        """
        Builds a single corpus entry by attaching the ground truth block
        to the HC contract. The ground truth block is keyed separately so
        inference_manager.py can strip it before building the LLM prompt.
        """
        sid = str(contract["source_id"])

        gt_block = gt_map.get(sid, {
            "main_id":              None,
            "sp_type":              None,
            "teff_pastel":          None,
            "logg_pastel":          None,
            "feh_pastel":           None,
            "n_pastel_measurements": None,
        })

        entry = dict(contract)          # shallow copy of the full HC contract
        entry["ground_truth"] = gt_block
        return entry

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def build(self) -> Path:
        """
        Orchestrates the full corpus build.

        Returns the path to the written stellar_corpus.json.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        contracts = self._load_contracts()
        gt_map    = self._load_ground_truth()

        corpus:  list[dict] = []
        report_entries: list[dict] = []

        n_valid          = 0
        n_gt_matched     = 0
        n_zero_qs        = 0
        n_with_warnings  = 0

        for contract in contracts:
            sid = str(contract.get("source_id", "UNKNOWN"))
            warnings = self._validate_contract(contract)

            has_gt  = sid in gt_map
            zero_qs = contract.get("quality_score", -1) == 0.0

            if warnings:
                n_with_warnings += 1
                for w in warnings:
                    log.warning(f"  [{sid}] {w}")
            else:
                n_valid += 1

            if has_gt:
                n_gt_matched += 1

            if zero_qs:
                n_zero_qs += 1

            entry = self._assemble_entry(contract, gt_map)
            corpus.append(entry)

            report_entries.append({
                "source_id":   sid,
                "quality_score": contract.get("quality_score"),
                "has_gt":      has_gt,
                "warnings":    warnings,
            })

        # --- Write corpus ---
        corpus_path = self.output_dir / "stellar_corpus.json"
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        log.info(f"stellar_corpus.json written -> {corpus_path}")

        # --- Write validation report ---
        report = {
            "total_contracts":       len(contracts),
            "valid_contracts":       n_valid,
            "contracts_with_warnings": n_with_warnings,
            "gt_matched":            n_gt_matched,
            "gt_unmatched":          len(contracts) - n_gt_matched,
            "zero_quality_score":    n_zero_qs,
            "entries":               report_entries,
        }
        report_path = self.output_dir / "corpus_build_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info(f"corpus_build_report.json written -> {report_path}")

        # --- Summary ---
        log.info("=" * 60)
        log.info("CORPUS BUILD SUMMARY")
        log.info(f"  Total contracts    : {len(contracts)}")
        log.info(f"  Valid (no warnings): {n_valid}")
        log.info(f"  With warnings      : {n_with_warnings}")
        log.info(f"  GT matched         : {n_gt_matched} / {len(contracts)}")
        log.info(f"  quality_score = 0.0: {n_zero_qs}")
        log.info("=" * 60)

        return corpus_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HC+SC Stellar Classifier — Corpus Builder v1.0"
    )
    parser.add_argument(
        "--contracts",
        type=Path,
        required=True,
        help="Path to hc_contracts.json",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground_truth_final.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sc"),
        help="Output directory (default: outputs/sc)",
    )
    args = parser.parse_args()

    if not args.contracts.exists():
        log.error(f"HC contracts file not found: {args.contracts}")
        raise FileNotFoundError(f"HC contracts file not found: {args.contracts}")

    if not args.ground_truth.exists():
        log.error(f"Ground truth file not found: {args.ground_truth}")
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")

    builder = CorpusBuilder(
        contracts_path    = args.contracts,
        ground_truth_path = args.ground_truth,
        output_dir        = args.output,
    )
    corpus_path = builder.build()
    log.info(f"Done. Corpus ready at: {corpus_path}")


if __name__ == "__main__":
    main()
