"""
validator_v5.py — SC Output Validator for V5
=============================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0

Extends validator.py with V5-specific metrics:

Level 4 — Stellar Description Quality (NEW)
    Evaluates the stellar_description field using:
      4a. ROUGE-L + ROUGE-1 against 20 human reference descriptions
      4b. Internal coherence — values mentioned in text match hc_anchor
      4c. Completeness — all three fields present and within word limits
      4d. Sub-type accuracy within class — evaluates the sub-type range
          given that the letter is fixed by HC

Levels 1-3 are inherited from validator.py (same metrics, same logic).

Outputs (in addition to validator.py outputs):
  <output_dir>/rouge_scores.json         — Per-star ROUGE scores
  <output_dir>/rouge_summary.json        — Aggregate ROUGE metrics
  <output_dir>/description_coherence.json — Coherence check results
  <output_dir>/subtype_accuracy.json     — Sub-type accuracy within class

Usage:
    python3 validator_v5.py \\
        --results   /path/to/sc_results_v5.json \\
        --corpus    /path/to/stellar_corpus.json \\
        --references /path/to/reference_descriptions.json \\
        --output    /path/to/outputs/validation_v5

Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 1.0
"""

import json
import re
import logging
import argparse
import math
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional
from collections import Counter

from sklearn.metrics import (
    f1_score, cohen_kappa_score, confusion_matrix, accuracy_score
)
from scipy.stats import spearmanr
from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="validator_v5.log",
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

MK_ORDER   = ["O", "B", "A", "F", "G", "K", "M"]
POP_ORDER  = ["Halo", "Disco Grueso", "Disco Fino"]
LUM_ORDER  = ["I", "II", "III", "IV", "V"]

_SP_PATTERN  = re.compile(r"^([OBAFGKM])")

WORD_LIMITS = {
    "physical_profile":   60,
    "population_context": 30,
    "notable_features":   30,
}

# Sub-type bins per class — for within-class accuracy
SUBTYPE_BINS = {
    "O": [(60000, 1e9, "0-1"), (45000, 60000, "2-3"), (35000, 45000, "4-6"), (30000, 35000, "7-9")],
    "B": [(25000, 30000, "0-1"), (18000, 25000, "2-3"), (13000, 18000, "4-6"), (10000, 13000, "7-9")],
    "A": [(9000, 9800, "0-1"),   (8400, 9000, "2-3"),  (7900, 8400, "4-6"),  (7500, 7900, "7-9")],
    "F": [(7200, 7500, "0-1"),   (6900, 7200, "2-3"),  (6500, 6900, "4-5"),  (6200, 6500, "6-7"), (6000, 6200, "8-9")],
    "G": [(5800, 6000, "0-1"),   (5600, 5800, "2-3"),  (5450, 5600, "4-5"),  (5300, 5450, "6-7"), (5200, 5300, "8-9")],
    "K": [(5000, 5200, "0-1"),   (4700, 5000, "2-3"),  (4300, 4700, "4-5"),  (4000, 4300, "6-7"), (3700, 4000, "8-9")],
    "M": [(3400, 3700, "0-1"),   (3100, 3400, "2-3"),  (2800, 3100, "4-5"),  (0, 2800, "6-9")],
}


def _expected_subtype(letter: str, teff_k: float) -> Optional[str]:
    """Returns the expected sub-type bin for a given letter and Teff."""
    bins = SUBTYPE_BINS.get(letter, [])
    for lo, hi, label in bins:
        if lo <= teff_k < hi:
            return label
    return None


def _extract_simbad_letter(sp_type: Optional[str]) -> Optional[str]:
    if not isinstance(sp_type, str): return None
    m = _SP_PATTERN.match(sp_type.strip())
    return m.group(1) if m else None


def _word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0


def _bootstrap_accuracy_ci(y_true, y_pred, n=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    arr_t, arr_p = np.array(y_true), np.array(y_pred)
    scores = [accuracy_score(arr_t[rng.integers(0, len(arr_t), size=len(arr_t))],
                             arr_p[rng.integers(0, len(arr_p), size=len(arr_p))])
              for _ in range(n)]
    alpha = 1 - ci
    return float(np.percentile(scores, 100*alpha/2)), float(np.percentile(scores, 100*(1-alpha/2)))


# ---------------------------------------------------------------------------
# Level 4a — ROUGE-L + ROUGE-1
# ---------------------------------------------------------------------------

class RougeEvaluator:
    """Computes ROUGE-L and ROUGE-1 scores against reference descriptions."""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], use_stemmer=True
        )

    def score_description(
        self,
        hypothesis: dict,
        reference: dict,
    ) -> dict:
        """
        Computes ROUGE scores for each field and returns per-field
        and aggregate scores.

        Args:
            hypothesis: stellar_description dict from LLM output.
            reference:  reference_description dict from human annotations.

        Returns:
            dict with per-field and aggregate ROUGE scores.
        """
        fields = ["physical_profile", "population_context", "notable_features"]
        results = {}

        for field in fields:
            hyp = hypothesis.get(field, "") or ""
            ref = reference.get(field, "") or ""

            if not ref.strip():
                results[field] = {"rouge1_f": None, "rougeL_f": None}
                continue

            scores = self.scorer.score(ref, hyp)
            results[field] = {
                "rouge1_f":  round(scores["rouge1"].fmeasure, 4),
                "rougeL_f":  round(scores["rougeL"].fmeasure, 4),
                "rouge1_p":  round(scores["rouge1"].precision, 4),
                "rouge1_r":  round(scores["rouge1"].recall, 4),
                "rougeL_p":  round(scores["rougeL"].precision, 4),
                "rougeL_r":  round(scores["rougeL"].recall, 4),
            }

        # Aggregate — mean across non-null fields
        rouge1_vals = [v["rouge1_f"] for v in results.values() if v.get("rouge1_f") is not None]
        rougeL_vals = [v["rougeL_f"] for v in results.values() if v.get("rougeL_f") is not None]

        results["aggregate"] = {
            "mean_rouge1_f": round(float(np.mean(rouge1_vals)), 4) if rouge1_vals else None,
            "mean_rougeL_f": round(float(np.mean(rougeL_vals)), 4) if rougeL_vals else None,
        }

        return results


# ---------------------------------------------------------------------------
# Level 4b — Internal coherence check
# ---------------------------------------------------------------------------

def check_coherence(
    result: dict,
    star: dict,
) -> dict:
    """
    Verifies that numeric values mentioned in the stellar_description
    match the hc_anchor and physical_vector of the star.

    Returns a dict with coherence flags and detected issues.
    """
    sd  = result.get("stellar_description", {})
    pv  = star.get("physical_vector", {})

    issues = []
    checks_passed = 0
    checks_total  = 0

    teff_k = pv.get("teff_k")
    logg   = pv.get("logg")
    fe_h   = pv.get("fe_h")
    met    = pv.get("metallicity")
    v_tan  = pv.get("v_tan")

    full_text = " ".join([
        sd.get("physical_profile", ""),
        sd.get("population_context", ""),
        sd.get("notable_features", ""),
    ])

    def check_value_mentioned(value, tolerance_pct=0.05, field_name="value"):
        """Check if a numeric value is mentioned in the text within tolerance."""
        nonlocal checks_passed, checks_total
        if value is None or math.isnan(float(value)):
            return
        checks_total += 1
        # Extract all numbers from text
        numbers = re.findall(r"-?\d+\.?\d*", full_text)
        for n_str in numbers:
            try:
                n = float(n_str)
                if abs(n - value) / (abs(value) + 1e-6) <= tolerance_pct:
                    checks_passed += 1
                    return
                if abs(n - value) <= 1:  # within 1 unit
                    checks_passed += 1
                    return
            except ValueError:
                continue
        issues.append(f"{field_name}={value} not found in description text")

    # Check key values
    if teff_k: check_value_mentioned(teff_k, 0.02, "teff_k")
    if logg:   check_value_mentioned(logg,   0.05, "logg")
    if v_tan:  check_value_mentioned(v_tan,  0.05, "v_tan")
    chem = fe_h if (fe_h and fe_h != 0.0) else met
    if chem:   check_value_mentioned(chem,   0.05, "chemistry")

    coherence_score = checks_passed / checks_total if checks_total > 0 else None

    # Check word limits
    limit_violations = []
    for field, limit in WORD_LIMITS.items():
        wc = _word_count(sd.get(field, ""))
        if wc > limit:
            limit_violations.append(f"{field}: {wc} words (limit {limit})")

    return {
        "coherence_score":    round(coherence_score, 3) if coherence_score is not None else None,
        "checks_passed":      checks_passed,
        "checks_total":       checks_total,
        "issues":             issues,
        "word_limit_violations": limit_violations,
    }


# ---------------------------------------------------------------------------
# Main Validator V5
# ---------------------------------------------------------------------------

class SCValidatorV5:
    """
    Full validator for V5 results including ROUGE-L evaluation.
    """

    def __init__(
        self,
        results_path: Path,
        corpus_path: Path,
        references_path: Path,
        output_dir: Path,
    ):
        self.results_path    = results_path
        self.corpus_path     = corpus_path
        self.references_path = references_path
        self.output_dir      = output_dir
        self.rouge_eval      = RougeEvaluator()

    def _load_results(self) -> dict:
        log.info(f"Loading SC results: {self.results_path}")
        with open(self.results_path) as f:
            results = json.load(f)
        log.info(f"  {len(results)} results loaded")
        return {str(r["source_id"]): r for r in results}

    def _load_corpus(self) -> dict:
        log.info(f"Loading corpus: {self.corpus_path}")
        with open(self.corpus_path) as f:
            corpus = json.load(f)
        return {str(s["source_id"]): s for s in corpus}

    def _load_references(self) -> dict:
        log.info(f"Loading references: {self.references_path}")
        with open(self.references_path) as f:
            refs = json.load(f)
        log.info(f"  {len(refs)} reference descriptions loaded")
        return {r["source_id"]: r for r in refs}

    # ------------------------------------------------------------------
    # Levels 1-3 (inherited from validator.py)
    # ------------------------------------------------------------------

    def _level1_mk(self, matched: list) -> dict:
        rows = [r for r in matched if r["true_letter"] and r["pred_letter"]]
        if not rows: return {}
        y_true = [r["true_letter"] for r in rows]
        y_pred = [r["pred_letter"] for r in rows]
        acc    = accuracy_score(y_true, y_pred)
        ci_lo, ci_hi = _bootstrap_accuracy_ci(y_true, y_pred)
        labels = MK_ORDER
        macro_f1    = f1_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        per_f1      = f1_score(y_true, y_pred, labels=labels, average=None,       zero_division=0)
        kappa       = cohen_kappa_score(y_true, y_pred)
        cm          = confusion_matrix(y_true, y_pred, labels=labels)
        distances   = [abs(MK_ORDER.index(t) - MK_ORDER.index(p))
                       for t, p in zip(y_true, y_pred) if t in MK_ORDER and p in MK_ORDER]
        return {
            "n_stars": len(rows),
            "overall_accuracy":    round(acc, 4),
            "bootstrap_ci_95":     [round(ci_lo, 4), round(ci_hi, 4)],
            "macro_f1":            round(macro_f1, 4),
            "weighted_f1":         round(weighted_f1, 4),
            "cohen_kappa":         round(kappa, 4),
            "mean_mk_letter_distance": round(float(np.mean(distances)), 4) if distances else None,
            "near_miss_accuracy_d1":   round(sum(1 for d in distances if d <= 1) / len(distances), 4) if distances else None,
            "per_class_f1":        {l: round(float(f), 4) for l, f in zip(labels, per_f1)},
            "confusion_matrix":    {"labels": labels, "matrix": cm.tolist()},
        }

    def _level2_pastel(self, matched: list) -> dict:
        pastel = [r for r in matched if r.get("teff_pastel")]
        if not pastel: return {}
        lum_to_logg = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 3.7, "V": 4.3}
        teff_d, logg_d, weights = [], [], []
        for r in pastel:
            pred_teff = None
            letter    = r.get("pred_letter", "")
            teff_hc   = r.get("teff_k_hc")
            if teff_hc and letter in SUBTYPE_BINS:
                for lo, hi, _ in SUBTYPE_BINS[letter]:
                    if lo <= teff_hc < hi:
                        pred_teff = (lo + hi) / 2
                        break
            w = r.get("n_pastel_measurements") or 1
            if pred_teff:
                teff_d.append(abs(pred_teff - r["teff_pastel"]) * w)
                weights.append(w)
            if r.get("logg_pastel") and r.get("pred_luminosity_class") in lum_to_logg:
                logg_d.append(abs(lum_to_logg[r["pred_luminosity_class"]] - r["logg_pastel"]) * w)

        feh_rows = [r for r in pastel if r.get("feh_pastel")]
        def _feh_to_pop(feh): return "Halo" if feh < -1.0 else ("Disco Grueso" if feh < -0.2 else "Disco Fino")
        pop_true = [_feh_to_pop(r["feh_pastel"]) for r in feh_rows]
        pop_pred = [r.get("pred_population", "") for r in feh_rows]
        pop_acc  = accuracy_score(pop_true, pop_pred) if pop_true else None
        pop_kappa = cohen_kappa_score(pop_true, pop_pred) if len(set(pop_true)) > 1 else None
        pop_cm   = confusion_matrix(pop_true, pop_pred, labels=POP_ORDER).tolist() if pop_true else None

        return {
            "teff_validation": {
                "n_stars": len(weights),
                "mean_abs_delta_teff_k": round(sum(teff_d)/sum(weights), 1) if weights else None,
            },
            "logg_validation": {
                "n_stars": len(logg_d),
                "mean_abs_delta_logg": round(sum(logg_d)/sum(weights), 4) if logg_d and weights else None,
            },
            "population_validation": {
                "n_stars":    len(feh_rows),
                "accuracy":   round(pop_acc, 4) if pop_acc else None,
                "cohen_kappa": round(pop_kappa, 4) if pop_kappa else None,
                "confusion_matrix": {"labels": POP_ORDER, "matrix": pop_cm},
            },
        }

    def _level3_confidence(self, matched: list) -> dict:
        rows = [r for r in matched if r.get("true_letter") and r.get("pred_letter")
                and r.get("spectral_type_confidence") is not None]
        if not rows: return {}
        correct   = [r for r in rows if r["pred_letter"] == r["true_letter"]]
        incorrect = [r for r in rows if r["pred_letter"] != r["true_letter"]]
        confs     = [r["spectral_type_confidence"] for r in rows]
        is_right  = [1 if r["pred_letter"] == r["true_letter"] else 0 for r in rows]
        sr, sp    = spearmanr(confs, is_right)
        binary    = [r for r in rows if r.get("is_binary_candidate")]
        non_bin   = [r for r in rows if not r.get("is_binary_candidate")]
        return {
            "n_stars": len(rows),
            "mean_confidence_correct":   round(float(np.mean([r["spectral_type_confidence"] for r in correct])),   4) if correct   else None,
            "mean_confidence_incorrect": round(float(np.mean([r["spectral_type_confidence"] for r in incorrect])), 4) if incorrect else None,
            "spearman_confidence_correctness": {"r": round(float(sr), 4), "p_value": round(float(sp), 6)},
            "binary_penalty_check": {
                "n_binary":                len(binary),
                "n_non_binary":            len(non_bin),
                "mean_confidence_binary":     round(float(np.mean([r["spectral_type_confidence"] for r in binary])),   4) if binary  else None,
                "mean_confidence_non_binary": round(float(np.mean([r["spectral_type_confidence"] for r in non_bin])),  4) if non_bin else None,
                "penalty_applied": (float(np.mean([r["spectral_type_confidence"] for r in binary])) <
                                    float(np.mean([r["spectral_type_confidence"] for r in non_bin])))
                                   if binary and non_bin else None,
            },
        }

    # ------------------------------------------------------------------
    # Level 4a — ROUGE
    # ------------------------------------------------------------------

    def _level4a_rouge(self, results: dict, references: dict) -> tuple[dict, list]:
        log.info(f"Level 4a: ROUGE evaluation on {len(references)} reference stars")
        per_star = []
        all_r1, all_rL = [], []

        for sid, ref in references.items():
            result = results.get(sid)
            if not result:
                log.warning(f"  [{sid}] No SC result for reference star")
                continue
            sd  = result.get("stellar_description", {})
            ref_desc = ref.get("reference_description", {})
            scores = self.rouge_eval.score_description(sd, ref_desc)
            scores["source_id"]  = sid
            scores["main_id"]    = ref.get("main_id")
            scores["category"]   = ref.get("category")
            per_star.append(scores)
            if scores["aggregate"].get("mean_rouge1_f") is not None:
                all_r1.append(scores["aggregate"]["mean_rouge1_f"])
            if scores["aggregate"].get("mean_rougeL_f") is not None:
                all_rL.append(scores["aggregate"]["mean_rougeL_f"])

        summary = {
            "n_evaluated":    len(per_star),
            "mean_rouge1_f":  round(float(np.mean(all_r1)), 4) if all_r1 else None,
            "mean_rougeL_f":  round(float(np.mean(all_rL)), 4) if all_rL else None,
            "std_rouge1_f":   round(float(np.std(all_r1)),  4) if all_r1 else None,
            "std_rougeL_f":   round(float(np.std(all_rL)),  4) if all_rL else None,
            "min_rouge1_f":   round(float(np.min(all_r1)),  4) if all_r1 else None,
            "max_rouge1_f":   round(float(np.max(all_r1)),  4) if all_r1 else None,
            "min_rougeL_f":   round(float(np.min(all_rL)),  4) if all_rL else None,
            "max_rougeL_f":   round(float(np.max(all_rL)),  4) if all_rL else None,
        }
        return summary, per_star

    # ------------------------------------------------------------------
    # Level 4b — Coherence
    # ------------------------------------------------------------------

    def _level4b_coherence(self, results: dict, corpus: dict) -> dict:
        log.info("Level 4b: Internal coherence check")
        scores, violations = [], []
        for sid, result in results.items():
            star = corpus.get(sid, {})
            chk  = check_coherence(result, star)
            if chk["coherence_score"] is not None:
                scores.append(chk["coherence_score"])
            if chk["word_limit_violations"]:
                violations.append({"source_id": sid, "violations": chk["word_limit_violations"]})

        return {
            "n_stars":             len(results),
            "mean_coherence":      round(float(np.mean(scores)), 3) if scores else None,
            "pct_full_coherence":  round(sum(1 for s in scores if s >= 1.0) / len(scores), 3) if scores else None,
            "n_word_limit_violations": len(violations),
            "word_limit_violation_sample": violations[:5],
        }

    # ------------------------------------------------------------------
    # Level 4c — Sub-type accuracy within class
    # ------------------------------------------------------------------

    def _level4c_subtype(self, matched: list) -> dict:
        log.info("Level 4c: Sub-type accuracy within confirmed letter class")
        rows = []
        for r in matched:
            letter  = r.get("pred_letter")
            teff    = r.get("teff_k_hc")
            pred_st = r.get("pred_subrange")
            if not all([letter, teff, pred_st]): continue
            expected = _expected_subtype(letter, teff)
            if expected is None: continue
            rows.append({
                "source_id":   r.get("source_id"),
                "letter":      letter,
                "teff_k":      teff,
                "pred_subrange": pred_st,
                "expected_subrange": expected,
                "correct":     pred_st == expected,
            })

        if not rows: return {}
        correct = sum(1 for r in rows if r["correct"])
        per_class = {}
        for letter in MK_ORDER:
            cls_rows = [r for r in rows if r["letter"] == letter]
            if cls_rows:
                per_class[letter] = {
                    "n": len(cls_rows),
                    "accuracy": round(sum(1 for r in cls_rows if r["correct"]) / len(cls_rows), 4),
                }

        return {
            "n_evaluated":      len(rows),
            "overall_accuracy": round(correct / len(rows), 4),
            "per_class":        per_class,
        }

    # ------------------------------------------------------------------
    # Summary writer
    # ------------------------------------------------------------------

    def _write_summary(self, report: dict, path: Path):
        lines = ["=" * 60,
                 "SC V5 VALIDATION SUMMARY — STELLAR HC+SC",
                 "=" * 60, ""]

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
            lines.append(f"  Near-miss acc (d≤1) : {l1.get('near_miss_accuracy_d1')}")
            lines.append("")
            lines.append("  Per-class F1:")
            for k, v in l1.get("per_class_f1", {}).items():
                lines.append(f"    {k}: {v:.4f}")
            lines.append("")

        l2 = report.get("level2_pastel", {})
        if l2:
            lines.append("LEVEL 2 — Physical Parameters (vs PASTEL)")
            tv = l2.get("teff_validation", {})
            lines.append(f"  Mean |ΔTeff| (K)    : {tv.get('mean_abs_delta_teff_k')}")
            lv = l2.get("logg_validation", {})
            lines.append(f"  Mean |Δlogg| (dex)  : {lv.get('mean_abs_delta_logg')}")
            pv = l2.get("population_validation", {})
            lines.append(f"  Population acc      : {pv.get('accuracy')}")
            lines.append(f"  Population kappa    : {pv.get('cohen_kappa')}")
            lines.append("")

        l3 = report.get("level3_confidence", {})
        if l3:
            lines.append("LEVEL 3 — Confidence Calibration")
            lines.append(f"  Conf (correct)      : {l3.get('mean_confidence_correct')}")
            lines.append(f"  Conf (incorrect)    : {l3.get('mean_confidence_incorrect')}")
            sp = l3.get("spearman_confidence_correctness", {})
            lines.append(f"  Spearman r          : {sp.get('r')}  (p={sp.get('p_value')})")
            bp = l3.get("binary_penalty_check", {})
            lines.append(f"  Binary penalty      : {bp.get('penalty_applied')}")
            lines.append("")

        l4a = report.get("level4a_rouge_summary", {})
        if l4a:
            lines.append("LEVEL 4a — Stellar Description (ROUGE vs references)")
            lines.append(f"  Stars evaluated     : {l4a.get('n_evaluated')}")
            lines.append(f"  Mean ROUGE-1 F      : {l4a.get('mean_rouge1_f')}")
            lines.append(f"  Mean ROUGE-L F      : {l4a.get('mean_rougeL_f')}")
            lines.append(f"  Std ROUGE-1         : {l4a.get('std_rouge1_f')}")
            lines.append(f"  Std ROUGE-L         : {l4a.get('std_rougeL_f')}")
            lines.append("")

        l4b = report.get("level4b_coherence", {})
        if l4b:
            lines.append("LEVEL 4b — Description Coherence")
            lines.append(f"  Mean coherence      : {l4b.get('mean_coherence')}")
            lines.append(f"  Full coherence pct  : {l4b.get('pct_full_coherence')}")
            lines.append(f"  Word limit viol.    : {l4b.get('n_word_limit_violations')}")
            lines.append("")

        l4c = report.get("level4c_subtype", {})
        if l4c:
            lines.append("LEVEL 4c — Sub-type Accuracy (within HC letter class)")
            lines.append(f"  Stars evaluated     : {l4c.get('n_evaluated')}")
            lines.append(f"  Overall accuracy    : {l4c.get('overall_accuracy')}")
            lines.append("  Per-class:")
            for k, v in l4c.get("per_class", {}).items():
                lines.append(f"    {k}: {v['accuracy']:.4f} (n={v['n']})")
            lines.append("")

        lines.append("=" * 60)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        log.info(f"Summary written -> {path}")

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def validate(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results    = self._load_results()
        corpus     = self._load_corpus()
        references = self._load_references()

        # Build matched rows
        matched = []
        for sid, star in corpus.items():
            result = results.get(sid)
            if not result: continue
            gt   = star.get("ground_truth", {})
            clf  = result.get("classification", {})
            conf = result.get("confidence_scores", {})
            pv   = star.get("physical_vector", {})
            true_letter = _extract_simbad_letter(gt.get("sp_type"))
            sp_raw = gt.get("sp_type", "") or ""
            lum_m  = re.search(r"\b(I{1,3}V?|IV|V)\b", sp_raw)
            true_lum = lum_m.group(1) if lum_m else None
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
                "teff_k_hc":               pv.get("teff_k"),
                "teff_pastel":             gt.get("teff_pastel"),
                "logg_pastel":             gt.get("logg_pastel"),
                "feh_pastel":              gt.get("feh_pastel"),
                "n_pastel_measurements":   gt.get("n_pastel_measurements"),
                "spectral_type_confidence": conf.get("spectral_type_confidence"),
                "is_binary_candidate":     star.get("logical_flags", {}).get("is_binary_candidate", False),
                "quality_score":           star.get("quality_score"),
            })

        log.info(f"Matched pairs: {len(matched)}")

        l1  = self._level1_mk(matched)
        l2  = self._level2_pastel(matched)
        l3  = self._level3_confidence(matched)
        l4a_summary, l4a_per_star = self._level4a_rouge(results, references)
        l4b = self._level4b_coherence(results, corpus)
        l4c = self._level4c_subtype(matched)

        report = {
            "n_results":              len(results),
            "n_corpus":               len(corpus),
            "n_matched":              len(matched),
            "level1_mk_coarse":       l1,
            "level2_pastel":          l2,
            "level3_confidence":      l3,
            "level4a_rouge_summary":  l4a_summary,
            "level4b_coherence":      l4b,
            "level4c_subtype":        l4c,
        }

        # Write outputs
        report_path = self.output_dir / "validation_report_v5.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        rouge_path = self.output_dir / "rouge_scores.json"
        with open(rouge_path, "w") as f:
            json.dump(l4a_per_star, f, indent=2, ensure_ascii=False)

        # Confusion matrix CSVs
        if l1.get("confusion_matrix"):
            pd.DataFrame(
                l1["confusion_matrix"]["matrix"],
                index=MK_ORDER, columns=MK_ORDER
            ).to_csv(self.output_dir / "confusion_spectral_v5.csv")

        self._write_summary(report, self.output_dir / "validation_summary_v5.txt")

        log.info("=" * 60)
        log.info("V5 VALIDATION COMPLETE")
        log.info(f"  Accuracy (MK)    : {l1.get('overall_accuracy')}")
        log.info(f"  ROUGE-1 mean     : {l4a_summary.get('mean_rouge1_f')}")
        log.info(f"  ROUGE-L mean     : {l4a_summary.get('mean_rougeL_f')}")
        log.info(f"  Coherence mean   : {l4b.get('mean_coherence')}")
        log.info(f"  Subtype accuracy : {l4c.get('overall_accuracy')}")
        log.info("=" * 60)

        return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="STELLAR SC Validator V5")
    parser.add_argument("--results",    type=Path, required=True)
    parser.add_argument("--corpus",     type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True,
                        help="Path to reference_descriptions.json")
    parser.add_argument("--output",     type=Path, default=Path("outputs/validation_v5"))
    args = parser.parse_args()

    for p in [args.results, args.corpus, args.references]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    v = SCValidatorV5(args.results, args.corpus, args.references, args.output)
    v.validate()


if __name__ == "__main__":
    main()
