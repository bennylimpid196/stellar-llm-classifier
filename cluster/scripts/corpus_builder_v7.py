"""
corpus_builder_v7.py — STELLAR V7
===================================
Generates stellar_corpus_v7.json from stellar_corpus_v5.json.

Three changes over V5 — everything else is inherited unchanged:

  1. A/B boundary rule (new in V7)
     Stars with Teff in [9700, 10100) K AND logg >= 3.8 are reassigned
     from B to A. This deterministic logg rule achieved 76.5% accuracy
     on boundary stars in V6 analysis vs 43.1% for the strict Teff threshold.
     The hc_anchor fields affected: mk_letter, near_teff_boundary, and a
     new field ab_boundary_logg_corrected (bool).

  2. Population group translation (Spanish → English)
     V5 emits Spanish labels ("Disco Fino", "Disco Grueso", "Halo").
     V7 normalizes to English ("Thin Disk", "Thick Disk", "Halo") to match
     the SC output schema and the knowledge base documents.

  3. NaN → null serialization
     V5 contains literal NaN values in spectral_summary.cat_triplet fields.
     V7 serializes these as JSON null (standard-compliant), which is safe for
     all downstream consumers (validator_v7.py, bert-score, pandas, etc.).

Usage:
    python3 corpus_builder_v7.py
    python3 corpus_builder_v7.py --input Data/stellar_corpus_v5.json
                                 --output Data/stellar_corpus_v7.json
    python3 corpus_builder_v7.py --dry-run   # process but do not write

Output:
    Data/stellar_corpus_v7.json — same schema as V5 plus:
      hc_anchor.mk_letter          may change B→A for boundary stars
      hc_anchor.population_group   now in English
      hc_anchor.ab_boundary_logg_corrected  bool (new field)
      hc_anchor.pipeline_version   "HC-2.0-V7"

Dependencies: none (stdlib only)

Version: 1.0 — STELLAR V7
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = BASE_DIR / "Data" / "stellar_corpus_v5.json"
DEFAULT_OUTPUT = BASE_DIR / "Data" / "stellar_corpus_v7.json"

# ── A/B boundary rule constants ───────────────────────────────────────────────
AB_TEFF_LOW  = 9700    # inclusive
AB_TEFF_HIGH = 10100   # exclusive  (>= 10100 → B unconditionally)
AB_LOGG_THR  = 3.8     # logg >= this → assign A within the boundary window

# ── Population translation map ────────────────────────────────────────────────
POPULATION_ES_TO_EN = {
    "Disco Fino":   "Thin Disk",
    "Disco Grueso": "Thick Disk",
    "Halo":         "Halo",
}


# ── NaN-safe JSON serialization ───────────────────────────────────────────────

def _nan_to_none(obj):
    """
    Recursively replace float NaN and Inf with None so the output is
    standard JSON (null instead of NaN, which is not valid JSON).
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


# ── A/B boundary rule ─────────────────────────────────────────────────────────

def _apply_ab_boundary_rule(anchor: dict) -> tuple[str, bool]:
    """
    Apply the V7 logg-based A/B boundary correction.

    Returns
    -------
    mk_letter : str   — corrected letter (may change B→A)
    corrected : bool  — True if the rule fired and changed the letter
    """
    letter = anchor.get("mk_letter", "")
    teff   = anchor.get("teff_k")
    logg   = anchor.get("logg")

    if teff is None or logg is None:
        return letter, False

    in_boundary_window = AB_TEFF_LOW <= teff < AB_TEFF_HIGH

    if not in_boundary_window:
        return letter, False

    # Rule fires only when the raw Teff would assign B (teff >= 10000)
    # but the logg criterion overrides to A.
    # Stars with teff < 10000 are already A — no change needed.
    if letter == "B" and logg >= AB_LOGG_THR:
        return "A", True

    return letter, False


# ── Population translation ────────────────────────────────────────────────────

def _translate_population(population_es: str) -> str:
    """
    Translate Spanish population label to English.
    Unknown values are returned unchanged with a warning.
    """
    translated = POPULATION_ES_TO_EN.get(population_es)
    if translated is None:
        log.warning(
            f"Unknown population label '{population_es}' — keeping as-is. "
            "Add it to POPULATION_ES_TO_EN if intentional."
        )
        return population_es
    return translated


# ── Per-star transformation ───────────────────────────────────────────────────

def _transform_star(star: dict, idx: int) -> tuple[dict, dict]:
    """
    Apply all V7 transformations to a single star contract.

    Returns
    -------
    star_v7  : dict   — transformed contract
    audit    : dict   — per-star audit record for the build report
    """
    sid   = star.get("source_id", f"IDX_{idx}")
    audit = {"source_id": sid, "ab_corrected": False, "population_translated": False}

    # Deep copy to avoid mutating the input — NaN→None is applied at the end
    import copy
    star_v7 = copy.deepcopy(star)

    anchor = star_v7.get("hc_anchor", {})
    if not anchor:
        log.warning(f"[{sid}] No hc_anchor found — star passed through unchanged.")
        return star_v7, audit

    # ── 1. A/B boundary rule ──────────────────────────────────────────────────
    new_letter, corrected = _apply_ab_boundary_rule(anchor)
    if corrected:
        old_letter = anchor["mk_letter"]
        anchor["mk_letter"] = new_letter
        audit["ab_corrected"] = True
        audit["ab_old_letter"] = old_letter
        audit["ab_new_letter"] = new_letter
        audit["ab_teff"] = anchor.get("teff_k")
        audit["ab_logg"] = anchor.get("logg")
        log.info(
            f"[{sid}] A/B correction: {old_letter}→{new_letter} "
            f"(Teff={anchor.get('teff_k')} K, logg={anchor.get('logg'):.3f})"
        )

    # Mark the boundary flag regardless of whether correction fired
    anchor["ab_boundary_logg_corrected"] = corrected

    # ── 2. Population translation ─────────────────────────────────────────────
    pop_es = anchor.get("population_group", "")
    pop_en = _translate_population(pop_es)
    if pop_en != pop_es:
        anchor["population_group"] = pop_en
        audit["population_translated"] = True
        audit["population_old"] = pop_es
        audit["population_new"] = pop_en

    # ── 3. Pipeline version bump ──────────────────────────────────────────────
    anchor["pipeline_version"] = "HC-2.0-V7"

    star_v7["hc_anchor"] = anchor

    return star_v7, audit


# ── Build report ──────────────────────────────────────────────────────────────

def _print_summary(audits: list[dict], n_total: int, elapsed: float) -> None:
    n_ab    = sum(1 for a in audits if a["ab_corrected"])
    n_pop   = sum(1 for a in audits if a["population_translated"])
    n_no_anchor = sum(1 for a in audits if not a.get("ab_corrected") and not a.get("population_translated") and "ab_old_letter" not in a and "population_old" not in a)

    log.info("=" * 62)
    log.info("  STELLAR V7 — corpus_builder_v7.py SUMMARY")
    log.info("=" * 62)
    log.info(f"  Stars processed          : {n_total}")
    log.info(f"  A/B boundary corrections : {n_ab}")
    log.info(f"  Population translations  : {n_pop}")
    log.info(f"  Elapsed                  : {elapsed:.2f}s")
    log.info("=" * 62)

    if n_ab > 0:
        log.info("  Stars with B→A correction:")
        for a in audits:
            if a["ab_corrected"]:
                log.info(
                    f"    {a['source_id']}  "
                    f"Teff={a['ab_teff']} K  logg={a['ab_logg']:.3f}  "
                    f"{a['ab_old_letter']}→{a['ab_new_letter']}"
                )
    log.info("=" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STELLAR V7 — Build stellar_corpus_v7.json from V5."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input corpus (V5). Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output corpus (V7). Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process and report but do not write the output file.",
    )
    args = parser.parse_args()

    log.info("=== corpus_builder_v7.py — STELLAR V7 — start ===")
    log.info(f"  Input  : {args.input}")
    log.info(f"  Output : {args.output}")
    log.info(f"  Dry run: {args.dry_run}")

    if not args.input.exists():
        log.error(f"Input corpus not found: {args.input}")
        sys.exit(1)

    # ── Load V5 corpus ────────────────────────────────────────────────────────
    log.info("Loading V5 corpus...")
    t0 = time.perf_counter()
    with open(args.input, encoding="utf-8") as f:
        corpus_v5 = json.load(f)   # allow_nan=True is Python default

    if not isinstance(corpus_v5, list):
        log.error("Expected a JSON array at the top level of the corpus.")
        sys.exit(1)

    log.info(f"Loaded {len(corpus_v5)} star contracts.")

    # ── Transform ─────────────────────────────────────────────────────────────
    corpus_v7 = []
    audits    = []

    for idx, star in enumerate(corpus_v5):
        star_v7, audit = _transform_star(star, idx)
        # Apply NaN→null globally on this star
        star_v7 = _nan_to_none(star_v7)
        corpus_v7.append(star_v7)
        audits.append(audit)

    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(audits, len(corpus_v5), elapsed)

    # ── Write output ──────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("Dry run — output file NOT written.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing {args.output} ...")
        with open(args.output, "w", encoding="utf-8") as f:
            # allow_nan=False enforces that no NaN slipped through
            json.dump(corpus_v7, f, ensure_ascii=False, indent=2, allow_nan=False)
        size_mb = args.output.stat().st_size / 1024 / 1024
        log.info(f"Written: {args.output}  ({size_mb:.2f} MB)")

    log.info("=== corpus_builder_v7.py — end ===")


if __name__ == "__main__":
    main()
