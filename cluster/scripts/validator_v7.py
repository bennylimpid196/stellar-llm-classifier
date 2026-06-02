"""
validator_v7.py — SC Output Validator for V7
=============================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0-V7
System: STELLAR

Changes vs V6
-------------
V7-VAL-1 | LEVEL 4a-BIS — BERTScore
    Adds semantic evaluation of stellar_description alongside ROUGE.
    Uses roberta-large via the bert-score library.
    Evaluated on the same 17 reference stars as ROUGE.
    Reports P, R, F1 per subfield and aggregate, enabling direct comparison
    with ROUGE to quantify the gap between lexical and semantic similarity.
    BERTScore is run only if bert-score is installed; otherwise skipped with
    a warning (ROUGE results are unaffected).

V7-VAL-2 | LEVEL 5 — RAG RETRIEVAL IMPACT (replaces V6 A/B anchor analysis)
    In V6, Level 5 evaluated the soft A/B anchor disambiguation. In V7 the
    A/B boundary is resolved deterministically by corpus_builder_v7.py, so
    that analysis is obsolete. Level 5 now evaluates whether the RAG system
    retrieved relevant context:
      - For each star, checks if the top retrieved chunk label is thematically
        consistent with the star's spectral letter (e.g. a K star retrieves
        a K-class chunk, an F star retrieves the F subtype calibration chunk).
      - Reports retrieval hit rate by spectral class.
      - Compares ROUGE-1 for stars where the top chunk is relevant vs not
        (proxy for RAG contribution to description quality).
    Requires that sc_results_v7 includes a "rag_top_chunk" field per star,
    which inference_manager_v7.py stamps when RAGEngine retrieves context.

V7-VAL-3 | POPULATION LABELS — ENGLISH
    POP_ORDER updated to ["Halo", "Thick Disk", "Thin Disk"].
    Level 2 population ground truth mapping updated accordingly.

V7-VAL-4 | CORPUS AND SUFFIXES — v7
    --corpus defaults to stellar_corpus_v7.json.
    All output files use v7 suffix.
    Log file: logs/validator_v7.log

Levels 1-4a (ROUGE), 4b (coherence), 4c (subtype) inherited from V6 unchanged.

Outputs
-------
  <output_dir>/validation_report_v7.json       — Full report
  <output_dir>/validation_summary_v7.txt       — Human-readable summary
  <output_dir>/rouge_scores_v7.json            — Per-star ROUGE scores
  <output_dir>/bertscore_v7.json               — Per-star BERTScore (if available)
  <output_dir>/confusion_spectral_v7.csv       — MK confusion matrix
  <output_dir>/rag_impact_v7.json              — Level 5 RAG retrieval impact

Usage
-----
    python3 validator_v7.py \\
        --results    outputs/sc_v7/sc_results_v7_<JID>.json \\
        --corpus     Data/stellar_corpus_v7.json \\
        --references Data/reference_descriptions.json \\
        --output     outputs/validation_v7

Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 3.0
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
from collections import defaultdict

from sklearn.metrics import (
    f1_score, cohen_kappa_score, confusion_matrix, accuracy_score
)
from scipy.stats import spearmanr
from rouge_score import rouge_scorer

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "validator_v7.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_PATH),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_console)

# ── Constants ─────────────────────────────────────────────────────────────────

MK_ORDER  = ["O", "B", "A", "F", "G", "K", "M"]
POP_ORDER = ["Halo", "Thick Disk", "Thin Disk"]   # V7-VAL-3: English
LUM_ORDER = ["I", "II", "III", "IV", "V"]

_SP_PATTERN = re.compile(r"^([OBAFGKM])")

WORD_LIMITS = {
    "physical_profile":   60,
    "population_context": 30,
    "notable_features":   30,
}

SUBTYPE_BINS = {
    "O": [(60000, 1e9,  "0-1"), (45000, 60000, "2-3"), (35000, 45000, "4-6"), (30000, 35000, "7-9")],
    "B": [(25000, 30000,"0-1"), (18000, 25000, "2-3"), (13000, 18000, "4-6"), (10000, 13000, "7-9")],
    "A": [(9000,  9800, "0-1"), (8400,  9000,  "2-3"), (7900,  8400,  "4-6"), (7500, 7900,  "7-9")],
    "F": [(7200,  7500, "0-1"), (6900,  7200,  "2-3"), (6500,  6900,  "4-5"), (6200, 6500,  "6-7"), (6000, 6200, "8-9")],
    "G": [(5800,  6000, "0-1"), (5600,  5800,  "2-3"), (5450,  5600,  "4-5"), (5300, 5450,  "6-7"), (5200, 5300, "8-9")],
    "K": [(5000,  5200, "0-1"), (4700,  5000,  "2-3"), (4300,  4700,  "4-5"), (4000, 4300,  "6-7"), (3700, 4000, "8-9")],
    "M": [(3400,  3700, "0-1"), (3100,  3400,  "2-3"), (2800,  3100,  "4-5"), (0,    2800,  "6-9")],
}

# Keywords in KB chunk labels that indicate relevance per spectral class
RAG_RELEVANCE_KEYWORDS = {
    "B": ["Class B", "subtype_calibration", "B Subtype", "B star"],
    "A": ["Class A", "subtype_calibration", "A/B", "boundary", "luminosity"],
    "F": ["Class F", "subtype_calibration", "F_subtype", "F subtype"],
    "G": ["Class G", "subtype_calibration", "solar"],
    "K": ["Class K", "subtype_calibration", "giant"],
    "M": ["Class M", "subtype_calibration", "dwarf", "emission"],
    "O": ["Class O", "subtype_calibration"],
}

# Flags that should always trigger quality_flags chunk
FLAG_RAG_KEYWORDS = ["quality_flags", "binary", "emission", "fit_diverged", "velocity"]


def _expected_subtype(letter: str, teff_k: float) -> Optional[str]:
    for lo, hi, label in SUBTYPE_BINS.get(letter, []):
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
    scores = [
        accuracy_score(arr_t[rng.integers(0, len(arr_t), size=len(arr_t))],
                       arr_p[rng.integers(0, len(arr_p), size=len(arr_p))])
        for _ in range(n)
    ]
    alpha = 1 - ci
    return float(np.percentile(scores, 100*alpha/2)), float(np.percentile(scores, 100*(1-alpha/2)))


# ── Level 4a — ROUGE (inherited from V6, unchanged) ──────────────────────────

class RougeEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def score_description(self, hypothesis: dict, reference: dict) -> dict:
        fields  = ["physical_profile", "population_context", "notable_features"]
        results = {}
        for field in fields:
            hyp = hypothesis.get(field, "") or ""
            ref = reference.get(field, "") or ""
            if not ref.strip():
                results[field] = {"rouge1_f": None, "rougeL_f": None}
                continue
            scores = self.scorer.score(ref, hyp)
            results[field] = {
                "rouge1_f": round(scores["rouge1"].fmeasure, 4),
                "rougeL_f": round(scores["rougeL"].fmeasure, 4),
                "rouge1_p": round(scores["rouge1"].precision, 4),
                "rouge1_r": round(scores["rouge1"].recall, 4),
                "rougeL_p": round(scores["rougeL"].precision, 4),
                "rougeL_r": round(scores["rougeL"].recall, 4),
            }
        rouge1_vals = [v["rouge1_f"] for v in results.values() if v.get("rouge1_f") is not None]
        rougeL_vals = [v["rougeL_f"] for v in results.values() if v.get("rougeL_f") is not None]
        results["aggregate"] = {
            "mean_rouge1_f": round(float(np.mean(rouge1_vals)), 4) if rouge1_vals else None,
            "mean_rougeL_f": round(float(np.mean(rougeL_vals)), 4) if rougeL_vals else None,
        }
        return results


# ── Level 4a-bis — BERTScore (V7-VAL-1) ──────────────────────────────────────

def _try_bertscore(hypotheses: list[str], references: list[str], lang: str = "en") -> Optional[dict]:
    """
    Compute BERTScore P/R/F1 for a list of (hypothesis, reference) pairs.
    Returns None if bert-score is not installed.
    Uses deberta-xlarge-mnli — best correlation with human judgment per Zhang et al. 2020.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        log.warning("bert-score not installed — BERTScore skipped. "
                    "Install with: pip install bert-score")
        return None

    try:
        log.info(f"Computing BERTScore for {len(hypotheses)} pairs "
                 f"(model: roberta-large)...")
        P, R, F = bert_score_fn(
            hypotheses,
            references,
            model_type="roberta-large",
            lang=lang,
            verbose=False,
        )
        return {
            "precision": [round(float(p), 4) for p in P.tolist()],
            "recall":    [round(float(r), 4) for r in R.tolist()],
            "f1":        [round(float(f), 4) for f in F.tolist()],
        }
    except Exception as e:
        log.warning(f"BERTScore computation failed: {e}")
        return None


# ── Coherence check (inherited from V6, unchanged) ────────────────────────────

def check_coherence(result: dict, star: dict) -> dict:
    sd   = result.get("stellar_description", {})
    pv   = star.get("physical_vector", {})
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
        nonlocal checks_passed, checks_total
        if value is None or math.isnan(float(value)):
            return
        checks_total += 1
        numbers = re.findall(r"-?\d+\.?\d*", full_text)
        for n_str in numbers:
            try:
                n = float(n_str)
                if abs(n - value) / (abs(value) + 1e-6) <= tolerance_pct:
                    checks_passed += 1
                    return
                if abs(n - value) <= 1:
                    checks_passed += 1
                    return
            except ValueError:
                continue
        issues.append(f"{field_name}={value} not found in description text")

    if teff_k: check_value_mentioned(teff_k, 0.02, "teff_k")
    if logg:   check_value_mentioned(logg,   0.05, "logg")
    if v_tan:  check_value_mentioned(v_tan,  0.05, "v_tan")
    chem = fe_h if (fe_h and fe_h != 0.0) else met
    if chem:   check_value_mentioned(chem,   0.05, "chemistry")

    coherence_score = checks_passed / checks_total if checks_total > 0 else None

    limit_violations = []
    for field, limit in WORD_LIMITS.items():
        wc = _word_count(sd.get(field, ""))
        if wc > limit:
            limit_violations.append(f"{field}: {wc} words (limit {limit})")

    return {
        "coherence_score":       round(coherence_score, 3) if coherence_score is not None else None,
        "checks_passed":         checks_passed,
        "checks_total":          checks_total,
        "issues":                issues,
        "word_limit_violations": limit_violations,
    }


# ── Main Validator V7 ─────────────────────────────────────────────────────────

class SCValidatorV7:

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

    # ── Levels 1-3 (inherited from V6, unchanged) ─────────────────────────────

    def _level1_mk(self, matched: list) -> dict:
        rows   = [r for r in matched if r["true_letter"] and r["pred_letter"]]
        if not rows: return {}
        y_true = [r["true_letter"] for r in rows]
        y_pred = [r["pred_letter"] for r in rows]
        acc    = accuracy_score(y_true, y_pred)
        ci_lo, ci_hi = _bootstrap_accuracy_ci(y_true, y_pred)
        labels       = MK_ORDER
        macro_f1     = f1_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
        weighted_f1  = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        per_f1       = f1_score(y_true, y_pred, labels=labels, average=None,       zero_division=0)
        kappa        = cohen_kappa_score(y_true, y_pred)
        cm           = confusion_matrix(y_true, y_pred, labels=labels)
        distances    = [abs(MK_ORDER.index(t) - MK_ORDER.index(p))
                        for t, p in zip(y_true, y_pred) if t in MK_ORDER and p in MK_ORDER]
        return {
            "n_stars":                 len(rows),
            "overall_accuracy":        round(acc, 4),
            "bootstrap_ci_95":         [round(ci_lo, 4), round(ci_hi, 4)],
            "macro_f1":                round(macro_f1, 4),
            "weighted_f1":             round(weighted_f1, 4),
            "cohen_kappa":             round(kappa, 4),
            "mean_mk_letter_distance": round(float(np.mean(distances)), 4) if distances else None,
            "near_miss_accuracy_d1":   round(sum(1 for d in distances if d <= 1) / len(distances), 4) if distances else None,
            "per_class_f1":            {l: round(float(f), 4) for l, f in zip(labels, per_f1)},
            "confusion_matrix":        {"labels": labels, "matrix": cm.tolist()},
        }

    def _level2_pastel(self, matched: list) -> dict:
        pastel      = [r for r in matched if r.get("teff_pastel")]
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

        # V7-VAL-3: English population mapping
        def _feh_to_pop_en(feh):
            if feh < -1.0: return "Halo"
            if feh < -0.2: return "Thick Disk"
            return "Thin Disk"

        pop_true  = [_feh_to_pop_en(r["feh_pastel"]) for r in feh_rows]
        pop_pred  = [r.get("pred_population", "") for r in feh_rows]
        pop_acc   = accuracy_score(pop_true, pop_pred) if pop_true else None
        pop_kappa = cohen_kappa_score(pop_true, pop_pred) if len(set(pop_true)) > 1 else None
        pop_cm    = confusion_matrix(pop_true, pop_pred, labels=POP_ORDER).tolist() if pop_true else None

        return {
            "teff_validation": {
                "n_stars":              len(weights),
                "mean_abs_delta_teff_k": round(sum(teff_d)/sum(weights), 1) if weights else None,
            },
            "logg_validation": {
                "n_stars":             len(logg_d),
                "mean_abs_delta_logg": round(sum(logg_d)/sum(weights), 4) if logg_d and weights else None,
            },
            "population_validation": {
                "n_stars":       len(feh_rows),
                "accuracy":      round(pop_acc, 4) if pop_acc is not None else None,
                "cohen_kappa":   round(pop_kappa, 4) if pop_kappa is not None else None,
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
        bin_conf  = float(np.mean([r["spectral_type_confidence"] for r in binary]))   if binary  else None
        nbin_conf = float(np.mean([r["spectral_type_confidence"] for r in non_bin]))  if non_bin else None
        return {
            "n_stars":                      len(rows),
            "mean_confidence_correct":      round(float(np.mean([r["spectral_type_confidence"] for r in correct])),   4) if correct   else None,
            "mean_confidence_incorrect":    round(float(np.mean([r["spectral_type_confidence"] for r in incorrect])), 4) if incorrect else None,
            "spearman_confidence_correctness": {"r": round(float(sr), 4), "p_value": round(float(sp), 6)},
            "binary_penalty_check": {
                "n_binary":                   len(binary),
                "n_non_binary":               len(non_bin),
                "mean_confidence_binary":     round(bin_conf,  4) if bin_conf  is not None else None,
                "mean_confidence_non_binary": round(nbin_conf, 4) if nbin_conf is not None else None,
                "penalty_applied":            (bin_conf < nbin_conf) if (bin_conf is not None and nbin_conf is not None) else None,
                "mean_delta":                 round(bin_conf - nbin_conf, 4) if (bin_conf is not None and nbin_conf is not None) else None,
                "note": "V7 target: delta <= -0.10 (explicit floor in system_prompt_v7). Positive delta = penalty not applied.",
            },
        }

    # ── Level 4a — ROUGE ─────────────────────────────────────────────────────

    def _level4a_rouge(self, results: dict, references: dict) -> tuple[dict, list]:
        log.info(f"Level 4a: ROUGE evaluation on {len(references)} reference stars")
        per_star = []
        all_r1, all_rL = [], []
        for sid, ref in references.items():
            result = results.get(sid)
            if not result:
                log.warning(f"  [{sid}] No SC result for reference star")
                continue
            sd       = result.get("stellar_description", {})
            ref_desc = ref.get("reference_description", {})
            scores   = self.rouge_eval.score_description(sd, ref_desc)
            scores["source_id"] = sid
            scores["main_id"]   = ref.get("main_id")
            scores["category"]  = ref.get("category")
            per_star.append(scores)
            if scores["aggregate"].get("mean_rouge1_f") is not None:
                all_r1.append(scores["aggregate"]["mean_rouge1_f"])
            if scores["aggregate"].get("mean_rougeL_f") is not None:
                all_rL.append(scores["aggregate"]["mean_rougeL_f"])

        summary = {
            "n_evaluated":   len(per_star),
            "mean_rouge1_f": round(float(np.mean(all_r1)), 4) if all_r1 else None,
            "mean_rougeL_f": round(float(np.mean(all_rL)), 4) if all_rL else None,
            "std_rouge1_f":  round(float(np.std(all_r1)),  4) if all_r1 else None,
            "std_rougeL_f":  round(float(np.std(all_rL)),  4) if all_rL else None,
            "min_rouge1_f":  round(float(np.min(all_r1)),  4) if all_r1 else None,
            "max_rouge1_f":  round(float(np.max(all_r1)),  4) if all_r1 else None,
            "per_field": {
                "physical_profile":   {"mean_rouge1_f": round(float(np.mean([s["physical_profile"]["rouge1_f"]   for s in per_star if s["physical_profile"].get("rouge1_f")   is not None])), 4) if per_star else None},
                "population_context": {"mean_rouge1_f": round(float(np.mean([s["population_context"]["rouge1_f"] for s in per_star if s["population_context"].get("rouge1_f") is not None])), 4) if per_star else None},
                "notable_features":   {"mean_rouge1_f": round(float(np.mean([s["notable_features"]["rouge1_f"]   for s in per_star if s["notable_features"].get("rouge1_f")   is not None])), 4) if per_star else None},
            },
        }
        return summary, per_star

    # ── Level 4a-bis — BERTScore (V7-VAL-1) ──────────────────────────────────

    def _level4a_bis_bertscore(self, results: dict, references: dict) -> Optional[dict]:
        log.info("Level 4a-bis: BERTScore evaluation")
        fields = ["physical_profile", "population_context", "notable_features"]
        per_star_bs = []
        hyps_agg, refs_agg = [], []

        for sid, ref in references.items():
            result = results.get(sid)
            if not result: continue
            sd       = result.get("stellar_description", {})
            ref_desc = ref.get("reference_description", {})
            star_row = {"source_id": sid, "main_id": ref.get("main_id")}

            for field in fields:
                hyp = sd.get(field, "") or ""
                r   = ref_desc.get(field, "") or ""
                star_row[f"hyp_{field}"] = hyp
                star_row[f"ref_{field}"] = r

            # Concatenated aggregate text for overall score
            hyp_concat = " ".join(sd.get(f, "") or "" for f in fields)
            ref_concat = " ".join(ref_desc.get(f, "") or "" for f in fields)
            star_row["hyp_concat"] = hyp_concat
            star_row["ref_concat"] = ref_concat
            hyps_agg.append(hyp_concat)
            refs_agg.append(ref_concat)
            per_star_bs.append(star_row)

        if not per_star_bs:
            log.warning("Level 4a-bis: no reference stars matched — skipping BERTScore")
            return None

        # Aggregate BERTScore
        agg_scores = _try_bertscore(hyps_agg, refs_agg)
        if agg_scores is None:
            return {"status": "skipped", "reason": "bert-score not installed"}

        # Per-field BERTScore
        field_scores = {}
        for field in fields:
            field_hyps = [r[f"hyp_{field}"] for r in per_star_bs]
            field_refs = [r[f"ref_{field}"] for r in per_star_bs]
            fs = _try_bertscore(field_hyps, field_refs)
            if fs:
                field_scores[field] = {
                    "mean_f1": round(float(np.mean(fs["f1"])), 4),
                    "mean_p":  round(float(np.mean(fs["precision"])), 4),
                    "mean_r":  round(float(np.mean(fs["recall"])), 4),
                }

        # Stamp per-star results
        for i, row in enumerate(per_star_bs):
            row["bertscore_f1"] = agg_scores["f1"][i]
            row["bertscore_p"]  = agg_scores["precision"][i]
            row["bertscore_r"]  = agg_scores["recall"][i]

        summary = {
            "n_evaluated":  len(per_star_bs),
            "model":        "roberta-large",
            "mean_f1":      round(float(np.mean(agg_scores["f1"])), 4),
            "mean_p":       round(float(np.mean(agg_scores["precision"])), 4),
            "mean_r":       round(float(np.mean(agg_scores["recall"])), 4),
            "std_f1":       round(float(np.std(agg_scores["f1"])), 4),
            "per_field":    field_scores,
            "per_star":     [
                {"source_id": r["source_id"], "main_id": r["main_id"],
                 "f1": r["bertscore_f1"], "p": r["bertscore_p"], "r": r["bertscore_r"]}
                for r in per_star_bs
            ],
        }
        return summary

    # ── Level 4b — Coherence (inherited from V6, unchanged) ──────────────────

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
            "n_stars":                   len(results),
            "mean_coherence":            round(float(np.mean(scores)), 3) if scores else None,
            "pct_full_coherence":        round(sum(1 for s in scores if s >= 1.0) / len(scores), 3) if scores else None,
            "n_word_limit_violations":   len(violations),
            "word_limit_violation_sample": violations[:5],
        }

    # ── Level 4c — Subtype accuracy (inherited from V6, unchanged) ────────────

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
                "source_id":         r.get("source_id"),
                "letter":            letter,
                "teff_k":            teff,
                "pred_subrange":     pred_st,
                "expected_subrange": expected,
                "correct":           pred_st == expected,
            })
        if not rows: return {}
        correct   = sum(1 for r in rows if r["correct"])
        per_class = {}
        for letter in MK_ORDER:
            cls_rows = [r for r in rows if r["letter"] == letter]
            if cls_rows:
                per_class[letter] = {
                    "n":        len(cls_rows),
                    "accuracy": round(sum(1 for r in cls_rows if r["correct"]) / len(cls_rows), 4),
                }
        return {
            "n_evaluated":      len(rows),
            "overall_accuracy": round(correct / len(rows), 4),
            "per_class":        per_class,
        }

    # ── Level 5 — RAG Retrieval Impact (V7-VAL-2) ────────────────────────────

    def _level5_rag_impact(self, results: dict, corpus: dict, rouge_per_star: list) -> dict:
        """
        Evaluates whether RAG retrieved thematically relevant chunks per star.

        Reads the 'rag_top_chunk' field stamped by inference_manager_v7.py.
        If not present (e.g. ablation run without RAG), reports accordingly.

        Computes:
          - Retrieval hit rate by spectral class (chunk label contains class keyword)
          - Mean ROUGE-1 for stars with relevant vs irrelevant top chunk
          - Flag retrieval hit rate (quality_flags chunk for stars with active flags)
        """
        log.info("Level 5: RAG retrieval impact analysis")

        rouge_map = {r["source_id"]: r["aggregate"].get("mean_rouge1_f")
                     for r in rouge_per_star if r.get("aggregate")}

        per_star   = []
        by_class   = defaultdict(lambda: {"hit": 0, "total": 0})
        flag_hits  = {"hit": 0, "total": 0}
        rouge_hit, rouge_miss = [], []

        for sid, result in results.items():
            top_chunk = result.get("rag_top_chunk", "")
            star      = corpus.get(sid, {})
            anchor    = star.get("hc_anchor", {})
            flags     = star.get("logical_flags", {})
            letter    = anchor.get("mk_letter", "")

            if not top_chunk:
                # RAG context was empty or field not stamped
                per_star.append({"source_id": sid, "letter": letter,
                                 "top_chunk": None, "relevant": None})
                continue

            # Check class relevance
            keywords = RAG_RELEVANCE_KEYWORDS.get(letter, [])
            relevant = any(kw.lower() in top_chunk.lower() for kw in keywords)

            # Check flag relevance
            active_flags = any([
                flags.get("is_binary_candidate"),
                flags.get("has_emission"),
                flags.get("fit_diverged"),
                flags.get("is_high_velocity"),
            ])
            if active_flags:
                flag_hit = any(kw.lower() in top_chunk.lower() for kw in FLAG_RAG_KEYWORDS)
                flag_hits["total"] += 1
                if flag_hit:
                    flag_hits["hit"] += 1

            by_class[letter]["total"] += 1
            if relevant:
                by_class[letter]["hit"] += 1

            r1 = rouge_map.get(sid)
            if r1 is not None:
                if relevant: rouge_hit.append(r1)
                else:        rouge_miss.append(r1)

            per_star.append({
                "source_id":   sid,
                "letter":      letter,
                "top_chunk":   top_chunk,
                "relevant":    relevant,
                "active_flags": active_flags,
            })

        n_with_rag = sum(1 for r in per_star if r["top_chunk"] is not None)
        n_relevant = sum(1 for r in per_star if r.get("relevant"))

        return {
            "n_stars":              len(per_star),
            "n_with_rag_context":   n_with_rag,
            "n_no_rag_context":     len(per_star) - n_with_rag,
            "overall_hit_rate":     round(n_relevant / n_with_rag, 4) if n_with_rag > 0 else None,
            "flag_retrieval": {
                "n_stars_with_active_flags": flag_hits["total"],
                "n_flag_chunk_retrieved":    flag_hits["hit"],
                "flag_hit_rate":             round(flag_hits["hit"] / flag_hits["total"], 4) if flag_hits["total"] > 0 else None,
            },
            "hit_rate_by_class": {
                letter: {
                    "hit_rate": round(v["hit"] / v["total"], 4) if v["total"] > 0 else None,
                    "n": v["total"],
                }
                for letter, v in by_class.items()
            },
            "rouge1_relevant_chunk":     round(float(np.mean(rouge_hit)),  4) if rouge_hit  else None,
            "rouge1_irrelevant_chunk":   round(float(np.mean(rouge_miss)), 4) if rouge_miss else None,
            "rouge1_delta_relevant_vs_irrelevant": round(
                float(np.mean(rouge_hit)) - float(np.mean(rouge_miss)), 4
            ) if rouge_hit and rouge_miss else None,
            "note": (
                "relevant = top chunk label contains keywords consistent with the star's spectral class. "
                "rouge1_delta > 0 indicates RAG improves description quality on matched references."
            ),
        }

    # ── Summary writer ────────────────────────────────────────────────────────

    def _write_summary(self, report: dict, path: Path):
        lines = ["=" * 60,
                 "SC V7 VALIDATION SUMMARY — STELLAR HC+SC",
                 "=" * 60, ""]

        l1 = report.get("level1_mk_coarse", {})
        if l1:
            lines.append("LEVEL 1 — MK Coarse (vs SIMBAD)")
            lines.append(f"  Stars evaluated     : {l1.get('n_stars')}")
            lines.append(f"  Overall accuracy    : {l1.get('overall_accuracy'):.4f}")
            ci = l1.get("bootstrap_ci_95", [None, None])
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
            pv2 = l2.get("population_validation", {})
            lines.append(f"  Population acc      : {pv2.get('accuracy')}")
            lines.append(f"  Population kappa    : {pv2.get('cohen_kappa')}")
            lines.append("")

        l3 = report.get("level3_confidence", {})
        if l3:
            lines.append("LEVEL 3 — Confidence Calibration")
            lines.append(f"  Conf (correct)      : {l3.get('mean_confidence_correct')}")
            lines.append(f"  Conf (incorrect)    : {l3.get('mean_confidence_incorrect')}")
            sp = l3.get("spearman_confidence_correctness", {})
            lines.append(f"  Spearman r          : {sp.get('r')}  (p={sp.get('p_value')})")
            bp = l3.get("binary_penalty_check", {})
            lines.append(f"  Binary penalty      : {bp.get('penalty_applied')}  (delta={bp.get('mean_delta')})")
            lines.append(f"  V7 target           : delta <= -0.10")
            lines.append("")

        l4a = report.get("level4a_rouge_summary", {})
        if l4a:
            lines.append("LEVEL 4a — Stellar Description (ROUGE vs references)")
            lines.append(f"  Stars evaluated     : {l4a.get('n_evaluated')}")
            lines.append(f"  Mean ROUGE-1 F      : {l4a.get('mean_rouge1_f')}")
            lines.append(f"  Mean ROUGE-L F      : {l4a.get('mean_rougeL_f')}")
            pf = l4a.get("per_field", {})
            if pf:
                lines.append("  Per-field ROUGE-1 F:")
                for field, vals in pf.items():
                    lines.append(f"    {field}: {vals.get('mean_rouge1_f')}")
            lines.append("")

        l4ab = report.get("level4a_bis_bertscore", {})
        if l4ab and l4ab.get("status") != "skipped":
            lines.append("LEVEL 4a-bis — BERTScore (semantic similarity)")
            lines.append(f"  Model               : {l4ab.get('model')}")
            lines.append(f"  Stars evaluated     : {l4ab.get('n_evaluated')}")
            lines.append(f"  Mean F1             : {l4ab.get('mean_f1')}")
            lines.append(f"  Mean Precision      : {l4ab.get('mean_p')}")
            lines.append(f"  Mean Recall         : {l4ab.get('mean_r')}")
            pf_bs = l4ab.get("per_field", {})
            if pf_bs:
                lines.append("  Per-field mean F1:")
                for field, vals in pf_bs.items():
                    lines.append(f"    {field}: {vals.get('mean_f1')}")
            lines.append("")
        elif l4ab and l4ab.get("status") == "skipped":
            lines.append("LEVEL 4a-bis — BERTScore: SKIPPED (bert-score not installed)")
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

        l5 = report.get("level5_rag_impact", {})
        if l5:
            lines.append("LEVEL 5 — RAG Retrieval Impact (V7 NEW)")
            lines.append(f"  Stars with RAG      : {l5.get('n_with_rag_context')}")
            lines.append(f"  Stars without RAG   : {l5.get('n_no_rag_context')}")
            lines.append(f"  Overall hit rate    : {l5.get('overall_hit_rate')}")
            fr = l5.get("flag_retrieval", {})
            lines.append(f"  Flag hit rate       : {fr.get('flag_hit_rate')} (n={fr.get('n_stars_with_active_flags')})")
            lines.append(f"  ROUGE-1 w/ relevant : {l5.get('rouge1_relevant_chunk')}")
            lines.append(f"  ROUGE-1 w/o relevant: {l5.get('rouge1_irrelevant_chunk')}")
            lines.append(f"  ROUGE-1 delta       : {l5.get('rouge1_delta_relevant_vs_irrelevant')}")
            lines.append("  Hit rate by class:")
            for letter, vals in l5.get("hit_rate_by_class", {}).items():
                lines.append(f"    {letter}: {vals.get('hit_rate')} (n={vals.get('n')})")
            lines.append("")

        lines.append("=" * 60)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        log.info(f"Summary written -> {path}")

    # ── Public entrypoint ─────────────────────────────────────────────────────

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
            true_pop = star.get("hc_anchor", {}).get("population_group")
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
                "true_population":          true_pop,
            })

        log.info(f"Matched pairs: {len(matched)}")

        l1              = self._level1_mk(matched)
        l2              = self._level2_pastel(matched)
        l3              = self._level3_confidence(matched)
        l4a_sum, l4a_ps = self._level4a_rouge(results, references)
        l4ab            = self._level4a_bis_bertscore(results, references)
        l4b             = self._level4b_coherence(results, corpus)
        l4c             = self._level4c_subtype(matched)
        l5              = self._level5_rag_impact(results, corpus, l4a_ps)

        report = {
            "prompt_version":          "v7",
            "n_results":               len(results),
            "n_corpus":                len(corpus),
            "n_matched":               len(matched),
            "level1_mk_coarse":        l1,
            "level2_pastel":           l2,
            "level3_confidence":       l3,
            "level4a_rouge_summary":   l4a_sum,
            "level4a_bis_bertscore":   l4ab,
            "level4b_coherence":       l4b,
            "level4c_subtype":         l4c,
            "level5_rag_impact":       l5,
        }

        # Write outputs — all v7 suffixed
        report_path = self.output_dir / "validation_report_v7.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        rouge_path = self.output_dir / "rouge_scores_v7.json"
        with open(rouge_path, "w") as f:
            json.dump(l4a_ps, f, indent=2, ensure_ascii=False)

        if l4ab and l4ab.get("status") != "skipped":
            bs_path = self.output_dir / "bertscore_v7.json"
            with open(bs_path, "w") as f:
                json.dump(l4ab, f, indent=2, ensure_ascii=False)

        rag_path = self.output_dir / "rag_impact_v7.json"
        with open(rag_path, "w") as f:
            json.dump(l5, f, indent=2, ensure_ascii=False)

        if l1.get("confusion_matrix"):
            pd.DataFrame(
                l1["confusion_matrix"]["matrix"],
                index=MK_ORDER, columns=MK_ORDER
            ).to_csv(self.output_dir / "confusion_spectral_v7.csv")

        self._write_summary(report, self.output_dir / "validation_summary_v7.txt")

        log.info("=" * 60)
        log.info("V7 VALIDATION COMPLETE")
        log.info(f"  Accuracy (MK)       : {l1.get('overall_accuracy')}")
        log.info(f"  ROUGE-1 mean        : {l4a_sum.get('mean_rouge1_f')}")
        log.info(f"  BERTScore F1 mean   : {l4ab.get('mean_f1') if l4ab and l4ab.get('status') != 'skipped' else 'skipped'}")
        log.info(f"  Coherence mean      : {l4b.get('mean_coherence')}")
        log.info(f"  Subtype accuracy    : {l4c.get('overall_accuracy')}")
        log.info(f"  RAG hit rate (L5)   : {l5.get('overall_hit_rate')}")
        log.info("=" * 60)

        return report_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STELLAR SC Validator V7")
    parser.add_argument("--results",    type=Path, required=True,
                        help="Path to sc_results_v7_<JID>.json")
    parser.add_argument("--corpus",     type=Path, default=Path("Data/stellar_corpus_v7.json"),
                        help="Path to stellar_corpus_v7.json")
    parser.add_argument("--references", type=Path, required=True,
                        help="Path to reference_descriptions.json")
    parser.add_argument("--output",     type=Path, default=Path("outputs/validation_v7"),
                        help="Output directory")
    args = parser.parse_args()

    for p in [args.results, args.corpus, args.references]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    v = SCValidatorV7(args.results, args.corpus, args.references, args.output)
    v.validate()


if __name__ == "__main__":
    main()
