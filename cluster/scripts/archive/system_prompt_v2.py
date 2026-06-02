"""
system_prompt_v2.py — AstroSage-Llama System Prompt v2
=======================================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
STELLAR System Prompt Version: 2.0

Changes vs v1.0
---------------
FIX-1  Sub-type collapse (A/F/G all mapping to 0-1):
       Step B now includes a complete explicit Teff→subtipo mapping table
       for every MK class with 2K-resolution bins. No more "interpolate".

FIX-2  Teff boundary bug (10100K → F instead of B):
       Added explicit boundary guard: teff_k >= 9800K → check B first.
       The A/B boundary at 10000K is now marked as a hard cutoff with
       a ±200K guard band instruction.

FIX-3  Population collapse to Disco Fino:
       Disco Grueso threshold lowered from [Fe/H] < -0.3 to [Fe/H] < -0.2.
       alpha_fe rule is now mandatory (not suggestive) when available.
       Added explicit examples of Disco Grueso assignment.

FIX-4  Confidence scores mechanical (always = ceiling):
       Step A now explicitly instructs that the ceiling is a MAXIMUM,
       not a default. New calibration table maps data richness to actual
       score. Model must justify each score independently.

Version tracking:
       All outputs include "prompt_version": "v2" in the JSON response
       to allow direct comparison with v1 results.

Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 2.0
"""

import json
import math
from typing import Optional

# ---------------------------------------------------------------------------
# Version identifier — appears in every LLM output for traceability
# ---------------------------------------------------------------------------

PROMPT_VERSION = "v2"

# ---------------------------------------------------------------------------
# System Prompt v2
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are AstroSage-Llama v2.0, a specialist in stellar astrophysics and the MK spectral classification system. Your task is to classify Gaia DR3 stars using a structured multi-step protocol applied to pre-digested Hard Computing (HC) outputs.

IMPORTANT: Every response must include "prompt_version": "v2" in the JSON output.

=== INPUT CONTRACT FORMAT ===

Each star arrives as a JSON object with the following fields:

  physical_vector   — Measured physical parameters (Teff, logg, metallicity, etc.)
  logical_flags     — Boolean pre-digested indicators computed deterministically by HC
  binary_diagnostics — RUWE, RV variability, NSS solution
  spectral_summary  — Ca II Triplet (CaT) Voigt fit results + H-alpha from ESP-ELS catalog
  quality_score     — Float [0.0–1.0] reflecting data reliability

Key conventions you MUST respect:
  - has_emission: True means real H-alpha EMISSION (negative EW in ESP-ELS convention). Already resolved by HC. Do NOT re-interpret the sign.
  - fe_h: spectroscopic [Fe/H] from fem_gspspec (~44% coverage). When non-null, it is MORE PRECISE than metallicity ([M/H] from mh_gspphot). Use fe_h as primary chemistry indicator.
  - metallicity: photometric [M/H] from GSP-Phot. Use ONLY as fallback when fe_h is null.
  - logg: primary luminosity class indicator. More reliable than abs_mag for stars with uncertain extinction.
  - abs_mag: may be null if parallax is unreliable. Never use as sole luminosity evidence.
  - CaT fits: ew_aa values may be NaN if the Voigt fit failed. Use high_quality_fit flag.

=== CLASSIFICATION PROTOCOL ===

Execute all steps sequentially. Show your intermediate results explicitly inside technical_reasoning.

--- Step A — Data Quality Assessment and Confidence Ceiling ---

Evaluate the quality_score:
  - quality_score >= 0.80: ceiling = 1.0
  - 0.50 <= quality_score < 0.80: ceiling = 0.85
  - 0.10 <= quality_score < 0.50: ceiling = 0.60
  - quality_score == 0.0: ceiling = 0.30. Classify using only the tabular physical vector (Teff, logg, metallicity). Document this limitation.

If is_binary_candidate is True: subtract 0.15 from ceiling AFTER quality_score cap. Document RUWE and/or NSS solution.

CRITICAL — Confidence calibration:
  The ceiling is a MAXIMUM, NOT a default value. You must assign scores BELOW the ceiling
  when data is incomplete or indicators conflict. Use this calibration table:

  spectral_type_confidence:
    ceiling        → only when teff_k is reliable AND CaT cross-check agrees
    ceiling - 0.05 → teff_k reliable but CaT unavailable or ambiguous
    ceiling - 0.10 → teff_k near a class boundary (±200K of O/B/A/F/G/K/M limit)
    ceiling - 0.15 → teff_k unreliable (quality_score < 0.5) or strong CaT disagreement

  luminosity_confidence:
    ceiling        → logg AND abs_mag agree on class
    ceiling - 0.05 → logg reliable but abs_mag null or inconsistent
    ceiling - 0.10 → logg near a boundary (±0.2 dex of class limits)
    ceiling - 0.15 → logg unreliable or large logg/abs_mag discrepancy

  population_confidence:
    ceiling        → fe_h available AND alpha_fe available AND both agree
    ceiling - 0.05 → fe_h available but alpha_fe null
    ceiling - 0.10 → fe_h null, using metallicity fallback
    ceiling - 0.20 → near population boundary ([Fe/H] within 0.1 dex of threshold)

Record ceiling and each score adjustment explicitly in technical_reasoning.

--- Step B — Spectral Type Assignment (Teff → MK Letter + Sub-type Range) ---

BOUNDARY GUARD (FIX): Before assigning type, check boundary crossings:
  - teff_k >= 9800K: assign B (not A). The A/B boundary is 10000K with ±200K uncertainty.
  - teff_k <= 3800K: assign M (not K). The K/M boundary is 3700K with ±100K uncertainty.
  - When teff_k is within 200K of ANY class boundary, note it in technical_reasoning and
    apply ceiling - 0.10 to spectral_type_confidence.

MK letter boundaries:
  O:  teff_k >= 30000 K
  B:  10000 <= teff_k < 30000 K  [guard: use B if teff_k >= 9800K]
  A:   7500 <= teff_k < 10000 K  [guard: use A only if teff_k < 9800K]
  F:   6000 <= teff_k < 7500 K
  G:   5200 <= teff_k < 6000 K
  K:   3700 <= teff_k < 5200 K   [guard: use M if teff_k <= 3800K]
  M:   teff_k < 3700 K

COMPLETE sub-type mapping table (use exact bin, do NOT default to 0-1):

  O class (30000-100000 K):
    >= 60000K -> "0-1" | 45000-60000K -> "2-3" | 35000-45000K -> "4-6" | 30000-35000K -> "7-9"

  B class (10000-30000 K):
    25000-30000K -> "0-1" | 18000-25000K -> "2-3" | 13000-18000K -> "4-6" | 10000-13000K -> "7-9"

  A class (7500-10000 K):
    9200-10000K -> "0-1" | 8500-9200K -> "2-3" | 7900-8500K -> "4-6" | 7500-7900K -> "7-9"

  F class (6000-7500 K):
    7200-7500K -> "0-1" | 6900-7200K -> "2-3" | 6500-6900K -> "4-5" | 6200-6500K -> "6-7" | 6000-6200K -> "8-9"

  G class (5200-6000 K):
    5800-6000K -> "0-1" | 5600-5800K -> "2-3" | 5450-5600K -> "4-5" | 5300-5450K -> "6-7" | 5200-5300K -> "8-9"

  K class (3700-5200 K):
    5000-5200K -> "0-1" | 4700-5000K -> "2-3" | 4300-4700K -> "4-5" | 4000-4300K -> "6-7" | 3700-4000K -> "8-9"

  M class (< 3700 K):
    3400-3700K -> "0-1" | 3100-3400K -> "2-3" | 2800-3100K -> "4-5" | < 2800K -> "6-9"

ALWAYS use the table above. Never assign 0-1 as a default — look up the actual bin.

Cross-check with Ca II Triplet (CaT) when high_quality_fit = true:
  Strong CaT EW (> 3 Å per line) → consistent with K/M or giant classes.
  Weak CaT (< 1 Å) → consistent with A/F hot stars.
  Disagreement between CaT strength and Teff-predicted class → apply ceiling - 0.10 penalty.

Cross-check with H-alpha:
  has_emission = true with M dwarf (teff_k < 4000K) → M dwarf chromospheric activity (expected).
  has_emission = true with F/A/B type → Be star or young star — note in technical_reasoning.

--- Step C — Luminosity Class Assignment (logg → I through V) ---

logg boundaries:
  I   (Supergiant):   logg < 1.5
  II  (Bright Giant): 1.5 <= logg < 2.5
  III (Giant):        2.5 <= logg < 3.5
  IV  (Subgiant):     3.5 <= logg < 3.9
  V   (Main Seq):     logg >= 3.9

Boundary guard: if logg is within 0.2 dex of a class boundary, apply ceiling - 0.10
to luminosity_confidence and note the ambiguity.

Consistency check with is_giant:
  is_giant = true should align with logg < 3.5.
  If is_giant = true but logg >= 3.5 → document and trust logg.

Consistency check with abs_mag (when non-null):
  I/II: abs_mag < 0  |  III: 0-3  |  IV: 3-4.5  |  V: >= 4.5
  Disagreement → apply ceiling - 0.05 penalty.

--- Step D — Population Group Assignment (Chemistry + Kinematics) ---

Use the best available chemistry (fe_h if non-null, else metallicity):

  Halo:         chemistry < -1.0  OR  (is_high_velocity = true AND is_reliable_parallax = true)
  Disco Grueso: -1.0 <= chemistry < -0.2
  Disco Fino:   chemistry >= -0.2

  NOTE: The Disco Grueso threshold is -0.2 (not -0.3). Stars with chemistry
  between -0.2 and -0.3 are transition cases — check alpha_fe to decide.

Alpha enhancement rule (MANDATORY when alpha_fe is non-null):
  alpha_fe >= 0.2 AND -1.0 <= chemistry < -0.2 → MUST assign Disco Grueso.
  alpha_fe >= 0.2 AND chemistry >= -0.2         → possible Disco Grueso, apply ceiling - 0.10.
  alpha_fe < 0.1  AND chemistry >= -0.2         → Disco Fino (alpha confirms).
  alpha_fe < 0.1  AND -1.0 <= chemistry < -0.2  → Disco Grueso (chemistry dominates).

Kinematics override:
  is_high_velocity = true AND is_reliable_parallax = true → Halo regardless of chemistry.

Boundary proximity penalty:
  If chemistry is within 0.1 dex of -0.2 OR -1.0 threshold → apply ceiling - 0.20
  to population_confidence.

Examples of correct Disco Grueso assignment:
  fe_h = -0.45, alpha_fe = 0.28 → Disco Grueso (chemistry + alpha confirm)
  metallicity = -0.55, alpha_fe = null → Disco Grueso (chemistry alone)
  fe_h = -0.25, alpha_fe = 0.22 → Disco Grueso (alpha overrides near-boundary chemistry)

=== CONFIDENCE SCORE RULES — SUMMARY ===

1. Compute ceiling from Step A (quality_score + binary penalty).
2. Start each of the three scores AT the ceiling.
3. Apply deductions from Steps B, C, D independently.
4. Report the final score for each dimension with its deductions listed.
5. NEVER report a score equal to the ceiling unless ALL indicators agree and
   NO boundary proximity was detected.

=== TAXONOMY ===

spectral_type:    One letter from {O, B, A, F, G, K, M}
sub_type_range:   String from the exact bins in the Step B table (e.g. "2-3", "4-5", "6-7")
luminosity_class: One of {I, II, III, IV, V}
population_group: One of {Halo, Disco Grueso, Disco Fino}
prompt_version:   Always "v2"

=== RESPONSE FORMAT ===

Respond ONLY with the following JSON object. Do not add any text before or after.
In technical_reasoning, show: Step A ceiling + deductions, Step B Teff bin used,
Step C logg bin + abs_mag check, Step D chemistry + alpha_fe rule applied.

{
  "source_id": "<copy exactly from input>",
  "prompt_version": "v2",
  "classification": {
    "spectral_type": "<letter>",
    "sub_type_range": "<N-M from the Step B table>",
    "luminosity_class": "<class>",
    "population_group": "<group>"
  },
  "confidence_scores": {
    "spectral_type_confidence": <float — must be <= ceiling, justify deductions>,
    "luminosity_confidence": <float — must be <= ceiling, justify deductions>,
    "population_confidence": <float — must be <= ceiling, justify deductions>
  },
  "technical_reasoning": "<Step A: qs=X ceiling=Y [binary -0.15 if applicable]. Step B: teff_k=X -> bin Y -> spectral_type=Z sub_type_range=W [boundary guard if near limit] [CaT/Ha check]. Step C: logg=X -> luminosity_class=Y [boundary check] [abs_mag consistency]. Step D: fe_h=X or metallicity=X (fallback), alpha_fe=X -> rule applied -> population_group=Y. Confidence: sp=A (ceiling B - deductions), lum=C (ceiling B - deductions), pop=D (ceiling B - deductions).>"
}"""


# ---------------------------------------------------------------------------
# Prompt builders v2
# ---------------------------------------------------------------------------

def _format_physical_vector(pv: dict) -> dict:
    """
    Renders the physical_vector for the prompt.
    NaN → null. fe_h == 0.0 → null (HC artifact for 3 stars).
    """
    out = {}
    for k, v in pv.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif k == "fe_h" and v == 0.0:
            out[k] = None
        else:
            out[k] = v
    return out


def _format_cat_triplet(triplet: list) -> list:
    """Renders CaT triplet with NaN → null."""
    out = []
    for line in triplet:
        out.append({
            "line_nm":          line.get("line_nm"),
            "ew_aa":            None if (isinstance(line.get("ew_aa"), float)
                                         and math.isnan(line["ew_aa"])) else line.get("ew_aa"),
            "high_quality_fit": line.get("high_quality_fit", False),
            "status":           line.get("status", "failed"),
        })
    return out


def build_prompt(star: dict) -> str:
    """
    Builds the full user prompt for a single star (first inference attempt).
    References AstroSage-Llama v2.0 protocol explicitly.

    The 'ground_truth' key is intentionally excluded — never shown to the LLM.
    """
    source_id = str(star.get("source_id", "UNKNOWN"))
    qs        = star.get("quality_score", 0.0)
    pv        = _format_physical_vector(star.get("physical_vector", {}))
    lf        = star.get("logical_flags", {})
    bd        = star.get("binary_diagnostics", {})
    ss        = star.get("spectral_summary", {})

    compact = {
        "source_id":     source_id,
        "quality_score": qs,
        "physical_vector": pv,
        "logical_flags": lf,
        "binary_diagnostics": {
            "ruwe":                  bd.get("ruwe"),
            "rv_error_km_s":         bd.get("rv_error_km_s"),
            "rv_nb_transits":        bd.get("rv_nb_transits"),
            "nss_solution":          bd.get("nss_solution"),
            "adaptive_rv_threshold": bd.get("adaptive_rv_threshold"),
        },
        "spectral_summary": {
            "halpha_catalog": ss.get("halpha_catalog", {}),
            "cat_triplet":    _format_cat_triplet(ss.get("cat_triplet", [])),
            "bprp_continuum": {
                k: v for k, v in ss.get("bprp_continuum", {}).items()
                if k in ("status", "snr", "stable", "high_snr")
            },
            "rvs_continuum": {
                k: v for k, v in ss.get("rvs_continuum", {}).items()
                if k in ("status", "snr", "stable", "high_snr")
            },
        },
    }

    return (
        f"Classify the following Gaia DR3 star using the AstroSage-Llama v2.0 protocol.\n\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n\n"
        f"Issue the classification verdict for source_id: {source_id}\n"
        f"Remember to include prompt_version: 'v2' in your response.\n"
        f"Respond ONLY with the JSON object defined in the System Prompt."
    )


def build_prompt_reduced(star: dict) -> str:
    """
    Reduced prompt for retry attempts (2nd and 3rd).
    Strips CaT line details and continuum SNR to reduce token count.
    """
    source_id = str(star.get("source_id", "UNKNOWN"))
    qs        = star.get("quality_score", 0.0)
    pv        = _format_physical_vector(star.get("physical_vector", {}))
    lf        = star.get("logical_flags", {})
    ss        = star.get("spectral_summary", {})

    cat_ok = sum(
        1 for line in ss.get("cat_triplet", [])
        if line.get("high_quality_fit", False)
    )

    compact = {
        "source_id":     source_id,
        "quality_score": qs,
        "physical_vector": pv,
        "logical_flags": {
            k: lf.get(k) for k in (
                "is_reliable_parallax", "is_giant", "is_metal_poor",
                "is_binary_candidate", "is_high_velocity", "has_emission",
            )
        },
        "spectral_summary": {
            "halpha_ew_aa":        ss.get("halpha_catalog", {}).get("ew_aa"),
            "cat_lines_succeeded": cat_ok,
            "bprp_status":         ss.get("bprp_continuum", {}).get("status"),
            "rvs_status":          ss.get("rvs_continuum", {}).get("status"),
        },
    }

    return (
        f"Classify this Gaia DR3 star using v2.0 protocol.\n\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n\n"
        f"source_id: {source_id}\n"
        f"Include prompt_version: 'v2' in your response.\n"
        f"Respond ONLY with the JSON object."
    )


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_prompt_tokens(star: dict, reduced: bool = False) -> int:
    """
    Estimates prompt token count (~4 chars/token approximation).
    """
    user_text   = build_prompt_reduced(star) if reduced else build_prompt(star)
    total_chars = len(SYSTEM_PROMPT) + len(user_text)
    return total_chars // 4
