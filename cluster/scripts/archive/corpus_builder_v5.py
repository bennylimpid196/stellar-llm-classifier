"""
corpus_builder_v5.py — HC Anchor Enrichment
=============================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
STELLAR Version: 5.0

Reads stellar_corpus.json and enriches each entry with a pre-computed
hc_anchor block containing the HC-layer decisions:
  - mk_letter:        MK spectral type letter (deterministic from teff_k)
  - population_group: Galactic population (deterministic from chemistry/kinematics)
  - chemistry_source: "fe_h" or "metallicity_fallback"
  - Proximity flags:  near_teff_boundary, near_logg_boundary, near_chemistry_boundary

This creates stellar_corpus_v5.json — the input to inference_manager_v5.py.
Having a separate enriched corpus file provides full auditability of HC decisions
independent of the LLM inference results.

HC Decision Rules
-----------------
MK Letter (from teff_k):
  O: >= 30000K
  B: >= 10000K
  A: >= 7500K
  F: >= 6000K
  G: >= 5200K
  K: >= 3700K
  M: < 3700K

Population Group (from chemistry + kinematics):
  Priority 1 — Kinematics override:
    is_high_velocity = True AND is_reliable_parallax = True → Halo
  Priority 2 — Alpha-fe rule (when alpha_fe is non-null):
    alpha_fe >= 0.2 AND -1.0 <= chemistry < -0.2 → Disco Grueso (confirmed)
  Priority 3 — Chemistry thresholds:
    chemistry < -1.0  → Halo
    chemistry < -0.2  → Disco Grueso
    else              → Disco Fino
  Chemistry source:
    fe_h (spectroscopic) takes priority over metallicity (photometric)
    fe_h == 0.0 is treated as artifact (null) — 3 stars in this corpus

Outputs:
  <output_dir>/stellar_corpus_v5.json     — Enriched corpus
  <output_dir>/hc_anchor_report.json      — Per-star HC decisions for audit
  <output_dir>/hc_anchor_summary.json     — Aggregate statistics

Usage:
    python3 corpus_builder_v5.py \\
        --corpus  /path/to/stellar_corpus.json \\
        --output  /path/to/Data/

Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 1.0
"""

import json
import math
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="corpus_builder_v5.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_console)

# ---------------------------------------------------------------------------
# HC Decision Functions
# ---------------------------------------------------------------------------

# MK class boundaries (teff_k in Kelvin)
MK_BOUNDARIES = [
    (30000, "O"),
    (10000, "B"),   # boundary guard: 9800K instead of 10000K
    (7500,  "A"),
    (6000,  "F"),
    (5200,  "G"),
    (3700,  "K"),
    (0,     "M"),
]

# Teff boundaries used to flag proximity (within 200K)
TEFF_BOUNDARY_VALUES = [30000, 10000, 7500, 6000, 5200, 3700]

# logg class boundaries
LOGG_BOUNDARIES = [1.5, 2.5, 3.5, 3.9]

# Chemistry thresholds
CHEM_BOUNDARY_HALO       = -1.0
CHEM_BOUNDARY_THICK_DISK = -0.2
CHEM_PROXIMITY_WINDOW    = 0.1


def compute_mk_letter(teff_k: float) -> str:
    """
    Deterministic MK letter assignment from teff_k.

    Standard MK boundary at 10000K.
    In V5 the LLM does not assign the letter, so no guard is needed.

    Args:
        teff_k (float): Effective temperature in Kelvin.

    Returns:
        str: MK letter {O, B, A, F, G, K, M}
    """
    for threshold, letter in MK_BOUNDARIES:
        if teff_k >= threshold:
            return letter
    return "M"


def compute_population(
    fe_h: Optional[float],
    metallicity: float,
    alpha_fe: Optional[float],
    v_tan: Optional[float],
    is_high_velocity: bool,
    is_reliable_parallax: bool,
) -> tuple[str, str, str]:
    """
    Deterministic population group assignment.

    Priority order:
      1. Kinematics override (is_high_velocity + reliable parallax)
      2. Alpha-fe rule (when alpha_fe >= 0.2)
      3. Chemistry thresholds

    Args:
        fe_h:                 Spectroscopic [Fe/H] (None if unavailable)
        metallicity:          Photometric [M/H] from GSP-Phot
        alpha_fe:             Alpha-to-iron ratio (None if unavailable)
        v_tan:                Tangential velocity in km/s (None if unavailable)
        is_high_velocity:     HC flag for high tangential velocity
        is_reliable_parallax: HC flag for reliable Gaia parallax

    Returns:
        tuple: (population_group, chemistry_source, decision_rule)
    """
    # Priority 1 — Kinematics override
    if is_high_velocity and is_reliable_parallax and v_tan is not None:
        return "Halo", "kinematics_override", "is_high_velocity=True"

    # Select best chemistry indicator
    if fe_h is not None and not math.isnan(fe_h) and fe_h != 0.0:
        chemistry = fe_h
        chem_source = "fe_h"
    else:
        chemistry = metallicity
        chem_source = "metallicity_fallback"

    # Priority 2 — Alpha-fe rule
    if (alpha_fe is not None and
            not math.isnan(alpha_fe) and
            alpha_fe >= 0.2 and
            CHEM_BOUNDARY_HALO <= chemistry < CHEM_BOUNDARY_THICK_DISK):
        return "Disco Grueso", chem_source, f"alpha_fe={alpha_fe:.2f}>=0.2 AND chemistry={chemistry:.3f} in [-1.0,-0.2)"

    # Priority 3 — Chemistry thresholds
    if chemistry < CHEM_BOUNDARY_HALO:
        return "Halo", chem_source, f"chemistry={chemistry:.3f} < {CHEM_BOUNDARY_HALO}"
    if chemistry < CHEM_BOUNDARY_THICK_DISK:
        return "Disco Grueso", chem_source, f"chemistry={chemistry:.3f} in [{CHEM_BOUNDARY_HALO},{CHEM_BOUNDARY_THICK_DISK})"
    return "Disco Fino", chem_source, f"chemistry={chemistry:.3f} >= {CHEM_BOUNDARY_THICK_DISK}"


def compute_hc_anchor(star: dict) -> dict:
    """
    Computes the full hc_anchor block for a single corpus entry.

    Args:
        star (dict): A single entry from stellar_corpus.json.

    Returns:
        dict: hc_anchor block ready to be injected into the corpus entry.
    """
    pv  = star.get("physical_vector", {})
    lf  = star.get("logical_flags", {})
    qs  = star.get("quality_score", 1.0)

    teff_k      = pv.get("teff_k", 5000.0)
    logg        = pv.get("logg")
    fe_h        = pv.get("fe_h")
    metallicity = pv.get("metallicity", 0.0)
    alpha_fe    = pv.get("alpha_fe")
    v_tan       = pv.get("v_tan")
    abs_mag     = pv.get("abs_mag")

    # Sanitize fe_h artifact (0.0 = HC artifact for 3 stars)
    if isinstance(fe_h, float) and (math.isnan(fe_h) or fe_h == 0.0):
        fe_h = None

    # HC decisions
    mk_letter = compute_mk_letter(teff_k)

    pop_group, chem_source, decision_rule = compute_population(
        fe_h,
        metallicity,
        alpha_fe,
        v_tan,
        lf.get("is_high_velocity", False),
        lf.get("is_reliable_parallax", True),
    )

    # Chemistry value used for population
    chemistry_value = (
        fe_h if (fe_h is not None and not math.isnan(fe_h))
        else metallicity
    )

    # Proximity flags for LLM confidence calibration
    near_teff_boundary = any(
        abs(teff_k - b) <= 200 for b in TEFF_BOUNDARY_VALUES
    )

    near_logg_boundary = (
        logg is not None and
        any(abs(logg - b) <= 0.2 for b in LOGG_BOUNDARIES)
    )

    near_chemistry_boundary = (
        chemistry_value is not None and (
            abs(chemistry_value - CHEM_BOUNDARY_THICK_DISK) <= CHEM_PROXIMITY_WINDOW or
            abs(chemistry_value - CHEM_BOUNDARY_HALO) <= CHEM_PROXIMITY_WINDOW
        )
    )

    return {
        # HC decisions (fixed — LLM must not change these)
        "mk_letter":          mk_letter,
        "population_group":   pop_group,
        "chemistry_source":   chem_source,
        "decision_rule":      decision_rule,

        # Values for LLM context
        "chemistry_value":    round(float(chemistry_value), 4) if chemistry_value is not None else None,
        "alpha_fe":           round(float(alpha_fe), 4) if alpha_fe is not None and not math.isnan(alpha_fe) else None,
        "teff_k":             teff_k,
        "logg":               logg,
        "abs_mag":            abs_mag,
        "v_tan":              round(float(v_tan), 4) if v_tan is not None else None,

        # Proximity flags for confidence calibration
        "near_teff_boundary":      near_teff_boundary,
        "near_logg_boundary":      near_logg_boundary,
        "near_chemistry_boundary": near_chemistry_boundary,

        # Flags passthrough for stellar_description
        "is_binary_candidate": lf.get("is_binary_candidate", False),
        "has_emission":        lf.get("has_emission", False),
        "is_metal_poor":       lf.get("is_metal_poor", False),
        "is_high_velocity":    lf.get("is_high_velocity", False),
        "quality_score":       qs,
    }


# ---------------------------------------------------------------------------
# Corpus Enricher
# ---------------------------------------------------------------------------

class CorpusBuilderV5:
    """
    Enriches stellar_corpus.json with pre-computed hc_anchor blocks.

    The hc_anchor contains all HC-layer decisions for each star.
    This creates a fully auditable record of what the HC computed
    before the LLM inference step.
    """

    def __init__(self, corpus_path: Path, output_dir: Path):
        self.corpus_path = corpus_path
        self.output_dir  = output_dir

    def build(self) -> Path:
        """
        Runs the enrichment pipeline.

        Returns:
            Path: Path to the written stellar_corpus_v5.json
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Loading corpus: {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        log.info(f"  {len(corpus)} entries loaded")

        enriched_corpus = []
        report_entries  = []

        # Aggregate counters
        letter_counts  = Counter()
        pop_counts     = Counter()
        source_counts  = Counter()
        near_teff_n    = 0
        near_logg_n    = 0
        near_chem_n    = 0

        for star in corpus:
            sid = str(star.get("source_id", "UNKNOWN"))

            hc_anchor = compute_hc_anchor(star)

            # Attach hc_anchor to the corpus entry
            enriched = dict(star)
            enriched["hc_anchor"] = hc_anchor
            enriched_corpus.append(enriched)

            # Counters
            letter_counts[hc_anchor["mk_letter"]] += 1
            pop_counts[hc_anchor["population_group"]] += 1
            source_counts[hc_anchor["chemistry_source"]] += 1
            if hc_anchor["near_teff_boundary"]:    near_teff_n += 1
            if hc_anchor["near_logg_boundary"]:    near_logg_n += 1
            if hc_anchor["near_chemistry_boundary"]: near_chem_n += 1

            # Per-star report entry
            report_entries.append({
                "source_id":        sid,
                "mk_letter":        hc_anchor["mk_letter"],
                "population_group": hc_anchor["population_group"],
                "chemistry_source": hc_anchor["chemistry_source"],
                "decision_rule":    hc_anchor["decision_rule"],
                "chemistry_value":  hc_anchor["chemistry_value"],
                "teff_k":           hc_anchor["teff_k"],
                "near_teff_boundary":      hc_anchor["near_teff_boundary"],
                "near_logg_boundary":      hc_anchor["near_logg_boundary"],
                "near_chemistry_boundary": hc_anchor["near_chemistry_boundary"],
            })

        # Write enriched corpus
        corpus_v5_path = self.output_dir / "stellar_corpus_v5.json"
        with open(corpus_v5_path, "w", encoding="utf-8") as f:
            json.dump(enriched_corpus, f, indent=2, ensure_ascii=False)
        log.info(f"stellar_corpus_v5.json written -> {corpus_v5_path}")

        # Write per-star audit report
        report_path = self.output_dir / "hc_anchor_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_entries, f, indent=2, ensure_ascii=False)
        log.info(f"hc_anchor_report.json written -> {report_path}")

        # Write summary
        summary = {
            "total_stars":          len(corpus),
            "mk_letter_distribution": dict(sorted(letter_counts.items())),
            "population_distribution": dict(pop_counts.most_common()),
            "chemistry_source": dict(source_counts.most_common()),
            "proximity_flags": {
                "near_teff_boundary":      near_teff_n,
                "near_logg_boundary":      near_logg_n,
                "near_chemistry_boundary": near_chem_n,
            },
        }
        summary_path = self.output_dir / "hc_anchor_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Log summary
        log.info("=" * 60)
        log.info("HC ANCHOR ENRICHMENT SUMMARY")
        log.info(f"  Total stars      : {len(corpus)}")
        log.info(f"  MK letters       : {dict(sorted(letter_counts.items()))}")
        log.info(f"  Population       : {dict(pop_counts.most_common())}")
        log.info(f"  Chemistry source : {dict(source_counts.most_common())}")
        log.info(f"  Near Teff bound  : {near_teff_n}")
        log.info(f"  Near logg bound  : {near_logg_n}")
        log.info(f"  Near chem bound  : {near_chem_n}")
        log.info("=" * 60)

        return corpus_v5_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="STELLAR V5 — HC Anchor Corpus Enrichment"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to stellar_corpus.json (output of corpus_builder.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/"),
        help="Output directory (default: Data/)",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")

    builder = CorpusBuilderV5(
        corpus_path = args.corpus,
        output_dir  = args.output,
    )
    out = builder.build()
    log.info(f"Done. Enriched corpus ready at: {out}")


if __name__ == "__main__":
    main()
