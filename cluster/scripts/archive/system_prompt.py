"""
system_prompt.py — AstroSage-Llama System Prompt
==================================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0

Defines the SYSTEM_PROMPT constant consumed by inference_manager.py and the
build_prompt() / build_prompt_reduced() functions that format each HC contract
into an LLM-ready user message.

Design decisions encoded here:
  - Sub-type granularity: 2-subtype range (e.g. "K1-K2"), reflecting the
    inherent 100-200 K uncertainty of Gaia GSP-Phot Teff estimates.
  - Reasoning style: explicit step-by-step protocol (Steps A through D),
    auditable for thesis evaluation.
  - quality_score == 0.0 handling: included with an explicit LOW_QUALITY flag
    in the prompt; confidence scores are automatically capped at 0.3.
  - fe_h precedence: when fe_h is non-null, it takes priority over metallicity
    ([M/H]) for population assignment.
  - is_binary_candidate: mandatory confidence score reduction documented in
    the protocol.
  - H-alpha source: exclusively from the ESP-ELS catalog (ew_aa in
    spectral_summary.halpha_catalog). The sign convention is pre-resolved by
    the HC layer — has_emission=True means real chromospheric emission.

Author: Hybrid Stellar Classifier Project / CIMAT
Version: 1.0
"""

import json
import math
from typing import Optional

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are AstroSage-Llama v1.0, a specialist in stellar astrophysics and the MK spectral classification system. Your task is to classify Gaia DR3 stars using a structured multi-step protocol applied to pre-digested Hard Computing (HC) outputs.

=== INPUT CONTRACT FORMAT ===

Each star arrives as a JSON object with the following fields:

  physical_vector   — Measured physical parameters (Teff, logg, metallicity, etc.)
  logical_flags     — Boolean pre-digested indicators computed deterministically by HC
  binary_diagnostics — RUWE, RV variability, NSS solution
  spectral_summary  — Ca II Triplet (CaT) Voigt fit results + H-alpha from ESP-ELS catalog
  quality_score     — Float [0.0–1.0] reflecting data reliability

Key conventions you MUST respect:
  - has_emission: True means real H-alpha EMISSION (negative EW in ESP-ELS convention). Already resolved by HC. Do NOT re-interpret the sign.
  - fe_h: spectroscopic [Fe/H] from fem_gspspec (~44% coverage). When non-null, it is MORE PRECISE than metallicity ([M/H] from mh_gspphot). Use fe_h as primary chemistry indicator when available.
  - metallicity: photometric [M/H] from GSP-Phot. Use as fallback when fe_h is null.
  - logg: passed directly from the Gaia catalog to the physical_vector. Use it as the primary luminosity class indicator — it is more reliable than abs_mag for stars with uncertain extinction.
  - abs_mag: may be null if parallax is unreliable (is_reliable_parallax = false). Never use abs_mag as sole luminosity evidence.
  - CaT fits: ew_aa values may be NaN if the Voigt fit failed. Use high_quality_fit flag to assess reliability.

=== CLASSIFICATION PROTOCOL ===

Execute all steps sequentially. Show your intermediate results explicitly inside technical_reasoning.

--- Step A — Data Quality Assessment ---

Evaluate the quality_score:
  - quality_score >= 0.80: Full confidence allowed (cap: 1.0)
  - 0.50 <= quality_score < 0.80: Moderate data quality. Cap all confidence scores at 0.85.
  - quality_score < 0.50: Degraded data. Cap all confidence scores at 0.60. Document which spectral data is missing or unreliable.
  - quality_score == 0.0: Parallax unreliable (is_reliable_parallax = false). Cap ALL confidence scores at 0.30. Classify using only the tabular physical vector (Teff, logg, metallicity). Document this limitation explicitly.

If is_binary_candidate is True: apply an ADDITIONAL confidence penalty of -0.15 to all scores AFTER the quality_score cap. Document the binary diagnostics (RUWE value and/or NSS solution) that triggered this flag.

Record the final confidence ceiling for this star before proceeding.

--- Step B — Spectral Type Assignment (Teff → MK Letter + Sub-type Range) ---

Map the effective temperature (teff_k) to the MK spectral type using these calibrated boundaries:

  O:  teff_k >= 30000 K
  B:  10000 <= teff_k < 30000 K
  A:   7500 <= teff_k < 10000 K
  F:   6000 <= teff_k < 7500 K
  G:   5200 <= teff_k < 6000 K
  K:   3700 <= teff_k < 5200 K
  M:   teff_k < 3700 K

Sub-type range assignment (2-subtype window reflecting GSP-Phot uncertainty of ~150 K):
  Within each class, assign a 2-subtype range based on position within the thermal window.
  Example mappings (illustrative, interpolate for intermediate values):
    F: 7200-7500 K -> "0-1" | 6800-7200 K -> "2-3" | 6400-6800 K -> "4-5" | 6000-6400 K -> "6-7"
    G: 5700-6000 K -> "0-2" | 5450-5700 K -> "2-4" | 5200-5450 K -> "5-6"
    K: 4800-5200 K -> "0-1" | 4400-4800 K -> "2-3" | 3900-4400 K -> "4-5"
  For O/B/A/M types: use analogous 2-subtype windows scaled to the class thermal range.

Cross-check with Ca II Triplet (CaT) EW when high_quality_fit = true:
  Strong CaT absorption (high EW) is consistent with K/M cool stars and giant luminosity classes.
  Weak or absent CaT is consistent with A/F hot stars.
  This is a consistency check, not a primary classifier.

Cross-check with H-alpha:
  has_emission = true signals chromospheric activity (M dwarfs, young stars, Be stars).
  This is diagnostic for spectral type refinement but does NOT override Teff.

--- Step C — Luminosity Class Assignment (logg → I through V) ---

Map logg to luminosity class using these boundaries:

  Ia/Ib (Supergiant):  logg < 1.5
  II (Bright Giant):   1.5 <= logg < 2.5
  III (Giant):         2.5 <= logg < 3.5
  IV (Subgiant):       3.5 <= logg < 3.9
  V (Main Sequence):   logg >= 3.9

Consistency check with is_giant flag:
  is_giant = true (M_G < 3 AND Teff < 7000 K AND reliable parallax) should align with logg < 3.5.
  If logg and is_giant disagree, document the discrepancy and trust logg as the primary indicator.

Consistency check with abs_mag (when non-null):
  Luminosity class I/II: abs_mag typically < 0
  Luminosity class III:  0 <= abs_mag < 3
  Luminosity class IV:   3 <= abs_mag < 4.5
  Luminosity class V:    abs_mag >= 4.5
  Document any disagreement between logg-derived class and abs_mag-derived class.

--- Step D — Population Group Assignment (Chemistry + Kinematics) ---

Use the best available chemistry indicator (fe_h if non-null, else metallicity):

  Halo:         [Fe/H] < -1.0  OR  (is_high_velocity = true AND is_reliable_parallax = true)
  Thick Disk:   -1.0 <= [Fe/H] < -0.3  AND  (alpha_fe > 0.2 if available)
  Thin Disk:    [Fe/H] >= -0.3

Refinement using alpha enhancement (alpha_fe, available for ~53% of stars):
  alpha_fe > 0.2 combined with [Fe/H] in range -1.0 to -0.3 strengthens Thick Disk assignment.
  alpha_fe <= 0.1 combined with [Fe/H] >= -0.3 strengthens Thin Disk assignment.

Kinematics override:
  If is_high_velocity = true AND is_reliable_parallax = true, assign Halo regardless of chemistry,
  but document any chemical inconsistency.

=== CONFIDENCE SCORE RULES ===

After all four steps, assign three independent confidence scores in [0.0, 1.0]:
  spectral_type_confidence: quality of Teff measurement + CaT cross-check agreement
  luminosity_confidence:    reliability of logg + abs_mag consistency
  population_confidence:    availability of fe_h vs metallicity fallback + alpha_fe coverage

Apply caps from Step A (quality_score cap and binary penalty) to ALL three scores.
Never assign a score higher than the ceiling established in Step A.

=== TAXONOMY ===

spectral_type:   One letter from {O, B, A, F, G, K, M}
sub_type_range:  String of two consecutive integers "N-(N+1)" or "N-(N+2)" (e.g. "0-1", "3-4", "5-6")
luminosity_class: One of {I, II, III, IV, V}
population_group: One of {Halo, Disco Grueso, Disco Fino}

=== RESPONSE FORMAT ===

Respond ONLY with the following JSON object. Do not add any text before or after.
In technical_reasoning, show the Step A ceiling, Step B Teff mapping, Step C logg mapping,
and Step D chemistry values used. Be explicit and auditable.

{
  "source_id": "<copy exactly from input>",
  "classification": {
    "spectral_type": "<letter>",
    "sub_type_range": "<N-M>",
    "luminosity_class": "<class>",
    "population_group": "<group>"
  },
  "confidence_scores": {
    "spectral_type_confidence": <float 0.0-1.0>,
    "luminosity_confidence": <float 0.0-1.0>,
    "population_confidence": <float 0.0-1.0>
  },
  "technical_reasoning": "<Step A: quality_score=X, ceiling=Y [binary penalty if applicable]. Step B: teff_k=X -> spectral_type=Y, sub_type_range=Z [CaT/Ha cross-check]. Step C: logg=X -> luminosity_class=Y [abs_mag consistency]. Step D: fe_h=X (or metallicity=X as fallback), alpha_fe=X -> population_group=Y.>"
}"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_physical_vector(pv: dict) -> dict:
    """
    Renders the physical_vector for the prompt, replacing NaN float values
    with null for LLM readability.

    Special case — fe_h == 0.0 as absence proxy:
    The HC pipeline wrote 0.0 into fe_h when fem_gspspec was NaN (3 stars).
    A true [Fe/H] of exactly 0.0 is physically valid for solar-metallicity
    stars, but in this corpus 0.0 is an artifact. We treat it as null so the
    LLM correctly falls back to metallicity ([M/H]) for those 3 stars.
    Stars with genuine non-zero fe_h values are unaffected.
    """
    out = {}
    for k, v in pv.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif k == "fe_h" and v == 0.0:
            # Artifact: HC wrote 0.0 when fem_gspspec was unavailable.
            out[k] = None
        else:
            out[k] = v
    return out


def _format_cat_triplet(triplet: list) -> list:
    """
    Renders the CaT triplet, replacing NaN EW/FWHM with null and keeping
    only the fields relevant to the LLM.
    """
    out = []
    for line in triplet:
        out.append({
            "line_nm":         line.get("line_nm"),
            "ew_aa":           None if (isinstance(line.get("ew_aa"), float)
                                        and math.isnan(line["ew_aa"])) else line.get("ew_aa"),
            "high_quality_fit": line.get("high_quality_fit", False),
            "status":          line.get("status", "failed"),
        })
    return out


def build_prompt(star: dict) -> str:
    """
    Builds the full user prompt for a single star (first inference attempt).

    Includes the complete physical vector, all logical flags, binary
    diagnostics, spectral summary, and quality score.

    Args:
        star (dict): A single entry from stellar_corpus.json.
                     The 'ground_truth' key is intentionally excluded
                     from the prompt — it is never shown to the LLM.

    Returns:
        str: Formatted prompt string.
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
            "ruwe":                   bd.get("ruwe"),
            "rv_error_km_s":          bd.get("rv_error_km_s"),
            "rv_nb_transits":         bd.get("rv_nb_transits"),
            "nss_solution":           bd.get("nss_solution"),
            "adaptive_rv_threshold":  bd.get("adaptive_rv_threshold"),
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
        f"Classify the following Gaia DR3 star using the AstroSage-Llama v1.0 protocol.\n\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n\n"
        f"Issue the classification verdict for source_id: {source_id}\n"
        f"Respond ONLY with the JSON object defined in the System Prompt."
    )


def build_prompt_reduced(star: dict) -> str:
    """
    Builds a reduced user prompt for retry attempts (2nd and 3rd).

    Strips CaT individual line details and continuum SNR fields to reduce
    token count, giving the model more output budget for the JSON response.
    Retrying with an identical prompt on a deterministic model (do_sample=False)
    produces the same failure — the reduced prompt changes the input context.

    Args:
        star (dict): A single entry from stellar_corpus.json.

    Returns:
        str: Shortened prompt string.
    """
    source_id = str(star.get("source_id", "UNKNOWN"))
    qs        = star.get("quality_score", 0.0)
    pv        = _format_physical_vector(star.get("physical_vector", {}))
    lf        = star.get("logical_flags", {})
    ss        = star.get("spectral_summary", {})

    # CaT: only summary (how many lines succeeded), not per-line details
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
        f"Classify this Gaia DR3 star.\n\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n\n"
        f"source_id: {source_id}\n"
        f"Respond ONLY with the JSON object."
    )


# ---------------------------------------------------------------------------
# Token estimation (rough, for pre-flight logging)
# ---------------------------------------------------------------------------

def estimate_prompt_tokens(star: dict, reduced: bool = False) -> int:
    """
    Estimates the number of tokens in the full prompt (system + user).
    Uses a ~4 chars/token approximation — sufficient for pre-flight checks.

    Args:
        star    (dict): A single corpus entry.
        reduced (bool): If True, estimates the reduced prompt.

    Returns:
        int: Approximate token count.
    """
    user_text   = build_prompt_reduced(star) if reduced else build_prompt(star)
    total_chars = len(SYSTEM_PROMPT) + len(user_text)
    return total_chars // 4