"""
validator.py — SC Output Validator
====================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0

Reads the consolidated SC results (sc_results_*.json) and the
stellar_corpus.json (which carries the ground truth block), then computes
a multi-level validation report.

Validation levels
-----------------
Level 1 — MK Coarse Validation (vs SIMBAD)
    Compares predicted spectral_type (letter) against sp_type from SIMBAD.
    Available for 491/498 stars. Metrics:
      - Per-class accuracy
      - Macro F1, Weighted F1
      - Cohen's Kappa
      - 7x7 confusion matrix (O/B/A/F/G/K/M)
      - Bootstrap 95% CI on overall accuracy (n=1000 resamples)
      - Weighted sub-type distance error

Level 2 — Physical Parameter Validation (vs PASTEL)
    Available for 340/498 stars. Metrics:
      - Mean |ΔTeff| and mean |ΔTeff|/Teff_PASTEL (weighted by n_pastel)
      - Mean |Δlogg| (weighted by n_pastel)
      - Population group confusion matrix (3x3: Halo / Disco Grueso / Disco Fino)
      - [Fe/H]-based population assignment accuracy

Level 3 — Confidence Calibration
    Verifies that confidence_scores are informative:
      - Mean confidence for correct vs incorrect spectral_type predictions
      - Mean confidence for is_binary_candidate=True vs False stars
      - Spearman correlation between spectral_type_confidence and correctness

Outputs
-------
  <output_dir>/validation_report.json     — Full structured metrics
  <output_dir>/confusion_spectral.csv     — 7x7 spectral type confusion matrix
  <output_dir>/confusion_luminosity.csv   — 5x5 luminosity class confusion matrix
  <output_dir>/confusion_population.csv   — 3x3 population group confusion matrix
  <output_dir>/validation_summary.txt     — Human-readable summary for thesis

Usage
-----
    python3 validator.py \\
        --results  /path/to/sc_results.json \\
        --corpus   /path/to/stellar_corpus.json \\
        --output   /path/to/outputs/validation

Author: Hybrid Stellar Classifier Project / CIMAT
Version: 1.0
"""

import json
import logging
import argparse
import re
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional

from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    accuracy_score,
)
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="validator.log",
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
# Constants
# ---------------------------------------------------------------------------

# MK sequence order — used for sub-type distance calculation
MK_ORDER = ["O", "B", "A", "F", "G", "K", "M"]

# Sub-type range per class (number of subtypes spanning each letter)
# Used to compute intra-class distance in the weighted error metric.
MK_SUBTYPES = {"O": 9, "B": 9, "A": 9, "F": 9, "G": 9, "K": 9, "M": 9}

# SIMBAD sp_type prefix extraction — captures the leading letter(s)
_SP_TYPE_PATTERN = re.compile(r"^([OBAFGKM])")

# Population group order for confusion matrix
POP_ORDER = ["Halo", "Disco Grueso", "Disco Fino"]

# Luminosity class order for confusion matrix
LUM_ORDER = ["I", "II", "III", "IV", "V"]

# Teff midpoints per MK class (K) — used for sub-type distance estimation
TEFF_MIDPOINTS = {
    "O": 40000, "B": 20000, "A": 8750,
    "F": 6750,  "G": 5600,  "K": 4450, "M": 3200,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_simbad_letter(sp_type: Optional[str]) -> Optional[str]:
    """
    Extracts the MK letter from a SIMBAD sp_type string.
    Examples: 'F7IV-V' -> 'F', 'K2III' -> 'K', 'G3IV-V' -> 'G'
    Returns None if no match.
    """
    if not isinstance(sp_type, str):
        return None
    match = _SP_TYPE_PATTERN.match(sp_type.strip())
    return match.group(1) if match else None


def _mk_letter_distance(a: str, b: str) -> int:
    """
    Returns the distance between two MK letters in the O→M sequence.
    Example: distance(F, K) = 3.
    """
    try:
        return abs(MK_ORDER.index(a) - MK_ORDER.index(b))
    except ValueError:
        return -1  # Unknown letter


def _teff_from_subtype_range(letter: str, sub_range: str) -> Optional[float]:
    """
    Estimates a representative Teff from a predicted spectral type + sub-type
    range string (e.g. 'F', '6-7'). Used for ΔTeff proxy when PASTEL is
    unavailable.

    Returns None if the mapping cannot be computed.
    """
    # Sub-type range midpoint
    parts = str(sub_range).split("-")
    if len(parts) != 2:
        return None
    try:
        mid_subtype = (int(parts[0]) + int(parts[1])) / 2.0
    except ValueError:
        return None

    # Linear interpolation within the class thermal window
    boundaries = {
        "O": (30000, 100000),
        "B": (10000, 30000),
        "A": (7500,  10000),
        "F": (6000,  7500),
        "G": (5200,  6000),
        "K": (3700,  5200),
        "M": (2400,  3700),
    }
    if letter not in boundaries:
        return None
    t_low, t_high = boundaries[letter]
    # Subtype 0 -> hot end, subtype 9 -> cool end
    teff = t_high - (mid_subtype / 9.0) * (t_high - t_low)
    return round(teff, 1)


def _bootstrap_accuracy_ci(
    y_true: list,
    y_pred: list,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Computes a bootstrap confidence interval for overall accuracy.

    Args:
        y_true      : True labels.
        y_pred      : Predicted labels.
        n_resamples : Number of bootstrap samples.
        ci          : Confidence level (default 0.95).
        seed        : Random seed for reproducibility.

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)

    scores = []
    for _ in range(n_resamples):
        idx     = rng.integers(0, n, size=n)
        scores.append(accuracy_score(arr_true[idx], arr_pred[idx]))

    alpha = 1.0 - ci
    lo    = float(np.percentile(scores, 100 * alpha / 2))
    hi    = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class SCValidator:
    """
    Validates SC classification outputs against SIMBAD and PASTEL ground truth.

    Attributes:
        results_path (Path): Path to sc_results_*.json.
        corpus_path  (Path): Path to stellar_corpus.json (carries ground truth).
        output_dir   (Path): Directory for validation outputs.
    """

    def __init__(
        self,
        results_path: Path,
        corpus_path: Path,
        output_dir: Path,
    ):
        self.results_path = results_path
        self.corpus_path  = corpus_path
        self.output_dir   = output_dir

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_results(self) -> dict[str, dict]:
        """Loads SC results indexed by source_id."""
        log.info(f"Loading SC results from: {self.results_path}")
        with open(self.results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        log.info(f"  SC results loaded: {len(results)}")
        return {str(r["source_id"]): r for r in results}

    def _load_corpus(self) -> dict[str, dict]:
        """Loads corpus indexed by source_id (carries ground_truth block)."""
        log.info(f"Loading corpus from: {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        log.info(f"  Corpus loaded: {len(corpus)} entries")
        return {str(s["source_id"]): s for s in corpus}

    # ------------------------------------------------------------------
    # Level 1 — MK Coarse Validation
    # ------------------------------------------------------------------

    def _level1_mk(
        self,
        matched: list[dict],
    ) -> dict:
        """
        Computes all Level 1 metrics (spectral type, letter only).

        Args:
            matched: List of dicts with keys:
                source_id, pred_letter, true_letter, pred_subrange,
                spectral_type_confidence, is_binary_candidate.
        """
        rows = [r for r in matched if r["true_letter"] is not None
                                   and r["pred_letter"] is not None]

        if not rows:
            log.warning("Level 1: no rows with both true and predicted letters.")
            return {}

        y_true = [r["true_letter"] for r in rows]
        y_pred = [r["pred_letter"] for r in rows]

        # Overall accuracy
        acc = accuracy_score(y_true, y_pred)

        # Bootstrap CI
        ci_lo, ci_hi = _bootstrap_accuracy_ci(y_true, y_pred)

        # F1 scores
        labels = MK_ORDER
        macro_f1    = f1_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None,      zero_division=0)

        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Per-class accuracy
        per_class_acc = {}
        for i, letter in enumerate(labels):
            row_sum = cm[i].sum()
            per_class_acc[letter] = float(cm[i, i] / row_sum) if row_sum > 0 else None

        # Weighted sub-type distance error
        # Distance 0 = correct letter; distance 1 = adjacent letter, etc.
        distances = [_mk_letter_distance(t, p) for t, p in zip(y_true, y_pred)
                     if _mk_letter_distance(t, p) >= 0]
        mean_distance = float(np.mean(distances)) if distances else None

        # Near-miss accuracy (distance <= 1, i.e. correct or adjacent letter)
        near_miss_acc = float(
            sum(1 for d in distances if d <= 1) / len(distances)
        ) if distances else None

        return {
            "n_stars":              len(rows),
            "overall_accuracy":     round(acc, 4),
            "bootstrap_ci_95":      [round(ci_lo, 4), round(ci_hi, 4)],
            "macro_f1":             round(macro_f1, 4),
            "weighted_f1":          round(weighted_f1, 4),
            "cohen_kappa":          round(kappa, 4),
            "mean_mk_letter_distance": round(mean_distance, 4) if mean_distance else None,
            "near_miss_accuracy_d1":   round(near_miss_acc, 4) if near_miss_acc else None,
            "per_class_f1": {
                letter: round(float(f1), 4)
                for letter, f1 in zip(labels, per_class_f1)
            },
            "per_class_accuracy":   {k: (round(v, 4) if v is not None else None)
                                     for k, v in per_class_acc.items()},
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
        }

    # ------------------------------------------------------------------
    # Level 2 — Physical Parameter Validation
    # ------------------------------------------------------------------

    def _level2_pastel(self, matched: list[dict]) -> dict:
        """
        Computes Level 2 metrics against PASTEL parameters.
        Only stars with teff_pastel != None are included.
        """
        pastel_rows = [r for r in matched if r.get("teff_pastel") is not None]
        log.info(f"Level 2: {len(pastel_rows)} stars with PASTEL Teff.")

        if not pastel_rows:
            return {}

        # ΔTeff — estimate predicted Teff from sub_type_range
        teff_deltas    = []
        teff_rel_deltas = []
        weights        = []

        for r in pastel_rows:
            pred_teff = _teff_from_subtype_range(
                r.get("pred_letter", ""),
                r.get("pred_subrange", ""),
            )
            if pred_teff is None:
                continue
            delta     = abs(pred_teff - r["teff_pastel"])
            rel_delta = delta / r["teff_pastel"]
            w         = r.get("n_pastel_measurements") or 1
            teff_deltas.append(delta * w)
            teff_rel_deltas.append(rel_delta * w)
            weights.append(w)

        mean_delta_teff     = float(sum(teff_deltas) / sum(weights)) if weights else None
        mean_rel_delta_teff = float(sum(teff_rel_deltas) / sum(weights)) if weights else None

        # Δlogg
        logg_rows = [r for r in pastel_rows if r.get("logg_pastel") is not None]
        log.info(f"Level 2: {len(logg_rows)} stars with PASTEL logg.")

        logg_deltas = []
        logg_weights = []

        lum_to_logg = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 3.7, "V": 4.3}
        for r in logg_rows:
            pred_lc   = r.get("pred_luminosity_class")
            pred_logg = lum_to_logg.get(pred_lc)
            if pred_logg is None:
                continue
            w = r.get("n_pastel_measurements") or 1
            logg_deltas.append(abs(pred_logg - r["logg_pastel"]) * w)
            logg_weights.append(w)

        mean_delta_logg = (
            float(sum(logg_deltas) / sum(logg_weights))
            if logg_weights else None
        )

        # Population group confusion (vs [Fe/H])
        feh_rows = [r for r in pastel_rows if r.get("feh_pastel") is not None]
        log.info(f"Level 2: {len(feh_rows)} stars with PASTEL [Fe/H].")

        def _feh_to_pop(feh: float) -> str:
            if feh < -1.0:
                return "Halo"
            elif feh < -0.3:
                return "Disco Grueso"
            else:
                return "Disco Fino"

        pop_true = [_feh_to_pop(r["feh_pastel"]) for r in feh_rows]
        pop_pred = [r.get("pred_population", "") for r in feh_rows]

        pop_acc   = accuracy_score(pop_true, pop_pred) if pop_true else None
        pop_kappa = cohen_kappa_score(pop_true, pop_pred) if len(set(pop_true)) > 1 else None
        pop_cm    = confusion_matrix(
            pop_true, pop_pred, labels=POP_ORDER
        ).tolist() if pop_true else None

        return {
            "teff_validation": {
                "n_stars":               len(weights),
                "mean_abs_delta_teff_k": round(mean_delta_teff, 1) if mean_delta_teff else None,
                "mean_rel_delta_teff":   round(mean_rel_delta_teff, 4) if mean_rel_delta_teff else None,
            },
            "logg_validation": {
                "n_stars":               len(logg_weights),
                "mean_abs_delta_logg":   round(mean_delta_logg, 4) if mean_delta_logg else None,
            },
            "population_validation": {
                "n_stars":         len(feh_rows),
                "accuracy":        round(pop_acc, 4) if pop_acc is not None else None,
                "cohen_kappa":     round(pop_kappa, 4) if pop_kappa is not None else None,
                "confusion_matrix": {
                    "labels": POP_ORDER,
                    "matrix": pop_cm,
                },
            },
        }

    # ------------------------------------------------------------------
    # Level 3 — Confidence Calibration
    # ------------------------------------------------------------------

    def _level3_confidence(self, matched: list[dict]) -> dict:
        """
        Checks whether confidence_scores correlate with classification
        correctness and respect the binary penalty rule.
        """
        rows = [r for r in matched
                if r.get("true_letter") is not None
                and r.get("pred_letter") is not None
                and r.get("spectral_type_confidence") is not None]

        if not rows:
            return {}

        correct   = [r for r in rows if r["pred_letter"] == r["true_letter"]]
        incorrect = [r for r in rows if r["pred_letter"] != r["true_letter"]]

        mean_conf_correct   = float(np.mean([r["spectral_type_confidence"] for r in correct]))   if correct   else None
        mean_conf_incorrect = float(np.mean([r["spectral_type_confidence"] for r in incorrect])) if incorrect else None

        # Spearman correlation between confidence and binary correctness (1/0)
        confs    = [r["spectral_type_confidence"] for r in rows]
        is_right = [1 if r["pred_letter"] == r["true_letter"] else 0 for r in rows]
        spear_r, spear_p = spearmanr(confs, is_right)

        # Binary candidate penalty check
        binary_rows     = [r for r in rows if r.get("is_binary_candidate")]
        non_binary_rows = [r for r in rows if not r.get("is_binary_candidate")]

        mean_conf_binary     = float(np.mean([r["spectral_type_confidence"] for r in binary_rows]))     if binary_rows     else None
        mean_conf_non_binary = float(np.mean([r["spectral_type_confidence"] for r in non_binary_rows])) if non_binary_rows else None

        return {
            "n_stars":                        len(rows),
            "mean_confidence_correct":        round(mean_conf_correct, 4)   if mean_conf_correct   is not None else None,
            "mean_confidence_incorrect":      round(mean_conf_incorrect, 4) if mean_conf_incorrect is not None else None,
            "spearman_confidence_correctness": {
                "r":       round(float(spear_r), 4),
                "p_value": round(float(spear_p), 6),
            },
            "binary_penalty_check": {
                "n_binary":                len(binary_rows),
                "n_non_binary":            len(non_binary_rows),
                "mean_confidence_binary":     round(mean_conf_binary, 4)     if mean_conf_binary     is not None else None,
                "mean_confidence_non_binary": round(mean_conf_non_binary, 4) if mean_conf_non_binary is not None else None,
                "penalty_applied":            (
                    mean_conf_binary < mean_conf_non_binary
                    if (mean_conf_binary is not None and mean_conf_non_binary is not None)
                    else None
                ),
            },
        }

    # ------------------------------------------------------------------
    # Luminosity class confusion
    # ------------------------------------------------------------------

    def _luminosity_confusion(self, matched: list[dict]) -> dict:
        """Builds the luminosity class confusion matrix."""
        rows = [r for r in matched
                if r.get("true_lum_class") is not None
                and r.get("pred_luminosity_class") is not None]

        if not rows:
            return {}

        y_true = [r["true_lum_class"] for r in rows]
        y_pred = [r["pred_luminosity_class"] for r in rows]

        acc   = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else None
        cm    = confusion_matrix(y_true, y_pred, labels=LUM_ORDER).tolist()

        return {
            "n_stars":     len(rows),
            "accuracy":    round(acc, 4),
            "cohen_kappa": round(kappa, 4) if kappa is not None else None,
            "confusion_matrix": {
                "labels": LUM_ORDER,
                "matrix": cm,
            },
        }

    # ------------------------------------------------------------------
    # Summary writer
    # ------------------------------------------------------------------

    def _write_summary(self, report: dict, path: Path) -> None:
        """Writes a human-readable validation summary for the thesis."""
        lines = []
        lines.append("=" * 60)
        lines.append("SC VALIDATION SUMMARY — Hybrid Stellar Classifier HC+SC")
        lines.append("=" * 60)
        lines.append("")

        l1 = report.get("level1_mk_coarse", {})
        if l1:
            lines.append("LEVEL 1 — MK Coarse (vs SIMBAD)")
            lines.append(f"  Stars evaluated     : {l1.get('n_stars')}")
            lines.append(f"  Overall accuracy    : {l1.get('overall_accuracy'):.4f}")
            ci = l1.get('bootstrap_ci_95', [None, None])
            lines.append(f"  95% Bootstrap CI    : [{ci[0]:.4f}, {ci[1]:.4f}]")
            lines.append(f"  Macro F1            : {l1.get('macro_f1'):.4f}")
            lines.append(f"  Weighted F1         : {l1.get('weighted_f1'):.4f}")
            lines.append(f"  Cohen Kappa         : {l1.get('cohen_kappa'):.4f}")
            lines.append(f"  Mean MK distance    : {l1.get('mean_mk_letter_distance')}")
            lines.append(f"  Near-miss acc (d<=1): {l1.get('near_miss_accuracy_d1')}")
            lines.append("")
            lines.append("  Per-class F1:")
            for letter, f1 in l1.get("per_class_f1", {}).items():
                lines.append(f"    {letter}: {f1:.4f}")
            lines.append("")

        l2 = report.get("level2_pastel", {})
        if l2:
            lines.append("LEVEL 2 — Physical Parameters (vs PASTEL)")
            tv = l2.get("teff_validation", {})
            lines.append(f"  Teff — n stars      : {tv.get('n_stars')}")
            lines.append(f"  Mean |ΔTeff|        : {tv.get('mean_abs_delta_teff_k')} K")
            lines.append(f"  Mean |ΔTeff|/Teff   : {tv.get('mean_rel_delta_teff')}")
            lv = l2.get("logg_validation", {})
            lines.append(f"  logg — n stars      : {lv.get('n_stars')}")
            lines.append(f"  Mean |Δlogg|        : {lv.get('mean_abs_delta_logg')} dex")
            pv = l2.get("population_validation", {})
            lines.append(f"  Population acc      : {pv.get('accuracy')}")
            lines.append(f"  Population kappa    : {pv.get('cohen_kappa')}")
            lines.append("")

        l3 = report.get("level3_confidence", {})
        if l3:
            lines.append("LEVEL 3 — Confidence Calibration")
            lines.append(f"  Mean conf (correct) : {l3.get('mean_confidence_correct')}")
            lines.append(f"  Mean conf (wrong)   : {l3.get('mean_confidence_incorrect')}")
            sp = l3.get("spearman_confidence_correctness", {})
            lines.append(f"  Spearman r          : {sp.get('r')}  (p={sp.get('p_value')})")
            bp = l3.get("binary_penalty_check", {})
            lines.append(f"  Mean conf (binary)  : {bp.get('mean_confidence_binary')}")
            lines.append(f"  Mean conf (non-bin) : {bp.get('mean_confidence_non_binary')}")
            lines.append(f"  Penalty applied     : {bp.get('penalty_applied')}")
            lines.append("")

        lines.append("=" * 60)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log.info(f"Summary written -> {path}")

    # ------------------------------------------------------------------
    # CSV confusion matrix writers
    # ------------------------------------------------------------------

    def _write_confusion_csv(
        self,
        matrix: list[list[int]],
        labels: list[str],
        path: Path,
    ) -> None:
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        df.index.name   = "true \\ pred"
        df.to_csv(path)
        log.info(f"Confusion matrix written -> {path}")

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def validate(self) -> Path:
        """
        Orchestrates the full validation pipeline.

        Returns the path to the written validation_report.json.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = self._load_results()
        corpus  = self._load_corpus()

        # Build matched rows — one dict per star found in both results and corpus
        matched: list[dict] = []

        for sid, star in corpus.items():
            result = results.get(sid)
            if result is None:
                log.warning(f"  [{sid}] No SC result found — skipping.")
                continue

            gt  = star.get("ground_truth", {})
            clf = result.get("classification", {})
            conf = result.get("confidence_scores", {})

            # Extract SIMBAD letter from sp_type
            true_letter = _extract_simbad_letter(gt.get("sp_type"))

            # Extract luminosity class from SIMBAD sp_type
            # Simple heuristic: look for Roman numeral suffix
            sp_raw = gt.get("sp_type", "") or ""
            lum_match = re.search(r"\b(I{1,3}V?|IV|V)\b", sp_raw)
            true_lum = lum_match.group(1) if lum_match else None
            # Normalize Ia/Ib -> I
            if true_lum and true_lum.startswith("I") and len(true_lum) > 2:
                true_lum = "I"

            matched.append({
                "source_id":                sid,
                "pred_letter":              clf.get("spectral_type"),
                "pred_subrange":            clf.get("sub_type_range"),
                "pred_luminosity_class":    clf.get("luminosity_class"),
                "pred_population":          clf.get("population_group"),
                "true_letter":              true_letter,
                "true_lum_class":           true_lum,
                "teff_pastel":              gt.get("teff_pastel"),
                "logg_pastel":              gt.get("logg_pastel"),
                "feh_pastel":               gt.get("feh_pastel"),
                "n_pastel_measurements":    gt.get("n_pastel_measurements"),
                "spectral_type_confidence": conf.get("spectral_type_confidence"),
                "luminosity_confidence":    conf.get("luminosity_confidence"),
                "population_confidence":    conf.get("population_confidence"),
                "is_binary_candidate":      star.get("logical_flags", {}).get("is_binary_candidate", False),
                "quality_score":            star.get("quality_score"),
            })

        log.info(f"Matched pairs built: {len(matched)}")

        # Run all validation levels
        log.info("Running Level 1 — MK Coarse...")
        l1 = self._level1_mk(matched)

        log.info("Running Level 2 — PASTEL...")
        l2 = self._level2_pastel(matched)

        log.info("Running Level 3 — Confidence Calibration...")
        l3 = self._level3_confidence(matched)

        log.info("Running Luminosity Class Confusion...")
        lum = self._luminosity_confusion(matched)

        # Assemble full report
        report = {
            "n_results":           len(results),
            "n_corpus":            len(corpus),
            "n_matched":           len(matched),
            "level1_mk_coarse":    l1,
            "level2_pastel":       l2,
            "level3_confidence":   l3,
            "luminosity_class":    lum,
        }

        # Write JSON report
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info(f"validation_report.json written -> {report_path}")

        # Write confusion CSVs
        if l1.get("confusion_matrix"):
            self._write_confusion_csv(
                l1["confusion_matrix"]["matrix"],
                l1["confusion_matrix"]["labels"],
                self.output_dir / "confusion_spectral.csv",
            )

        pop_cm = l2.get("population_validation", {}).get("confusion_matrix")
        if pop_cm and pop_cm.get("matrix"):
            self._write_confusion_csv(
                pop_cm["matrix"],
                pop_cm["labels"],
                self.output_dir / "confusion_population.csv",
            )

        lum_cm = lum.get("confusion_matrix")
        if lum_cm and lum_cm.get("matrix"):
            self._write_confusion_csv(
                lum_cm["matrix"],
                lum_cm["labels"],
                self.output_dir / "confusion_luminosity.csv",
            )

        # Write human-readable summary
        self._write_summary(report, self.output_dir / "validation_summary.txt")

        # Final log
        log.info("=" * 60)
        log.info("VALIDATION COMPLETE")
        log.info(f"  Matched pairs  : {len(matched)}")
        if l1:
            log.info(f"  Accuracy (MK)  : {l1.get('overall_accuracy')}")
            log.info(f"  Macro F1       : {l1.get('macro_f1')}")
            log.info(f"  Cohen Kappa    : {l1.get('cohen_kappa')}")
        log.info("=" * 60)

        return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HC+SC Stellar Classifier — SC Output Validator v1.0"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to sc_results_*.json (output of inference_manager.py)",
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
        default=Path("outputs/validation"),
        help="Output directory (default: outputs/validation)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        log.error(f"Results file not found: {args.results}")
        raise FileNotFoundError(f"Results file not found: {args.results}")

    if not args.corpus.exists():
        log.error(f"Corpus file not found: {args.corpus}")
        raise FileNotFoundError(f"Corpus file not found: {args.corpus}")

    validator = SCValidator(
        results_path = args.results,
        corpus_path  = args.corpus,
        output_dir   = args.output,
    )
    report_path = validator.validate()
    log.info(f"Done. Report at: {report_path}")


if __name__ == "__main__":
    main()