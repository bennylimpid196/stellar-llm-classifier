"""
system_prompt_v5.py — AstroSage-Llama System Prompt v5
=======================================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
STELLAR System Prompt Version: 5.0
 
Architecture change vs V1-V4
-----------------------------
In V1-V4 the LLM was responsible for:
  1. Assigning the MK letter from teff_k
  2. Assigning the population group from chemistry
  3. Classifying sub-type, luminosity class
  4. Generating confidence scores
 
In V5 the Hard Computing layer pre-computes:
  1. MK letter  — deterministic lookup from teff_k boundaries
  2. Population  — deterministic lookup from chemistry thresholds
 
The SC (LLM) is responsible for:
  1. Confirming and contextualizing the HC letter assignment
  2. Assigning the sub-type range within the confirmed class
  3. Assigning the luminosity class from logg
  4. Generating calibrated confidence scores
  5. Writing a structured stellar description in English (NEW)
 
New output field — stellar_description:
  physical_profile   : ≤60 words. Physical interpretation of Teff, logg,
                       luminosity class, and abs_mag. Use ONLY values from
                       the hc_anchor block.
  population_context : ≤30 words. Why this population was assigned.
                       Use ONLY chemistry and kinematics from hc_anchor.
  notable_features   : ≤30 words. Active flags only. Write "None identified."
                       if no flags are raised.
 
Anti-hallucination rules for stellar_description:
  - NEVER mention stellar mass, radius, or age — not present in the data.
  - NEVER reference external catalogs, missions, or star names.
  - NEVER infer properties not explicitly present in hc_anchor.
  - ONLY use numeric values that appear verbatim in hc_anchor.
  - notable_features is filled ONLY when at least one flag is True:
    is_binary_candidate, has_emission, is_metal_poor, is_high_velocity,
    quality_score < 0.5.
 
Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 5.0
"""
 
import json
import math
from typing import Optional
 
PROMPT_VERSION = "v5"
 
# ---------------------------------------------------------------------------
# HC pre-computation functions — letter and population
# ---------------------------------------------------------------------------
 
def compute_mk_letter(teff_k: float) -> str:
    """
    Deterministic MK letter from teff_k.
    Boundary guard: teff_k >= 9800K -> B (not A).
    """
    if teff_k >= 30000:  return "O"
    if teff_k >= 9800:   return "B"
    if teff_k >= 7500:   return "A"
    if teff_k >= 6000:   return "F"
    if teff_k >= 5200:   return "G"
    if teff_k >= 3700:   return "K"
    return "M"
 
 
def compute_population(
    fe_h: Optional[float],
    metallicity: float,
    alpha_fe: Optional[float],
    v_tan: Optional[float],
    is_high_velocity: bool,
    is_reliable_parallax: bool,
) -> tuple[str, str]:
    """
    Deterministic population group from chemistry and kinematics.
 
    Returns:
        tuple: (population_group, chemistry_source)
    """
    # Kinematic override
    if is_high_velocity and is_reliable_parallax and v_tan is not None:
        return "Halo", "kinematics_override"
 
    # Select best chemistry indicator
    if fe_h is not None and fe_h != 0.0:
        chem = fe_h
        source = "fe_h"
    else:
        chem = metallicity
        source = "metallicity_fallback"
 
    # Alpha-fe rule
    if alpha_fe is not None and alpha_fe >= 0.2 and -1.0 <= chem < -0.2:
        return "Disco Grueso", source
    if alpha_fe is not None and alpha_fe >= 0.2 and chem >= -0.2:
        # Near boundary — still check thresholds
        pass
 
    # Chemistry thresholds
    if chem < -1.0:
        return "Halo", source
    if chem < -0.2:
        return "Disco Grueso", source
    return "Disco Fino", source
 
 
def build_hc_anchor(star: dict) -> dict:
    """
    Builds the hc_anchor block injected into the prompt.
    Contains all HC pre-computed values the LLM must use.
    """
    pv  = star.get("physical_vector", {})
    lf  = star.get("logical_flags", {})
 
    teff_k   = pv.get("teff_k", 5000.0)
    fe_h     = pv.get("fe_h")
    metallicity = pv.get("metallicity", 0.0)
    alpha_fe = pv.get("alpha_fe")
    v_tan    = pv.get("v_tan")
    is_high_velocity   = lf.get("is_high_velocity", False)
    is_reliable_parallax = lf.get("is_reliable_parallax", True)
 
    # Sanitize fe_h artifact
    if isinstance(fe_h, float) and (math.isnan(fe_h) or fe_h == 0.0):
        fe_h = None
 
    mk_letter = compute_mk_letter(teff_k)
 
    pop_group, chem_source = compute_population(
        fe_h, metallicity, alpha_fe,
        v_tan, is_high_velocity, is_reliable_parallax
    )
 
    # Boundary proximity flags for confidence calibration
    boundaries = [30000, 9800, 7500, 6000, 5200, 3700]
    near_boundary = any(abs(teff_k - b) <= 200 for b in boundaries)
 
    logg = pv.get("logg")
    logg_boundaries = [1.5, 2.5, 3.5, 3.9]
    near_logg_boundary = (
        logg is not None and
        any(abs(logg - b) <= 0.2 for b in logg_boundaries)
    )
 
    chemistry_value = fe_h if fe_h is not None else metallicity
    near_chem_boundary = abs(chemistry_value - (-0.2)) <= 0.1 or abs(chemistry_value - (-1.0)) <= 0.1
 
    return {
        "mk_letter":              mk_letter,
        "population_group":       pop_group,
        "chemistry_source":       chem_source,
        "chemistry_value":        round(chemistry_value, 4) if chemistry_value is not None else None,
        "alpha_fe":               alpha_fe,
        "teff_k":                 teff_k,
        "v_tan":                  v_tan,
        "near_teff_boundary":     near_boundary,
        "near_logg_boundary":     near_logg_boundary,
        "near_chemistry_boundary": near_chem_boundary,
        "is_high_velocity":       is_high_velocity,
        "is_binary_candidate":    lf.get("is_binary_candidate", False),
        "has_emission":           lf.get("has_emission", False),
        "is_metal_poor":          lf.get("is_metal_poor", False),
        "quality_score":          star.get("quality_score", 1.0),
    }
 
 
# ---------------------------------------------------------------------------
# System Prompt v5
# ---------------------------------------------------------------------------
 
SYSTEM_PROMPT = """You are AstroSage-Llama v5.0, a specialist in stellar astrophysics and the MK spectral classification system.
 
The Hard Computing (HC) layer has already determined the MK letter and population group for each star. These values are provided in the hc_anchor block and are FIXED — do not change them. Your task is to:
  1. Assign the sub-type range within the HC-determined letter class.
  2. Assign the luminosity class from logg.
  3. Generate calibrated confidence scores.
  4. Write a structured stellar description in English.
 
IMPORTANT: Every response must include "prompt_version": "v5".
 
=== HC_ANCHOR BLOCK ===
 
The hc_anchor block contains pre-computed values you MUST use:
  mk_letter            — MK letter (fixed, do not change)
  population_group     — Population assignment (fixed, do not change)
  chemistry_source     — "fe_h" or "metallicity_fallback"
  chemistry_value      — The value used for population assignment
  near_teff_boundary   — True if teff_k is within 200K of a class boundary
  near_logg_boundary   — True if logg is within 0.2 dex of a class boundary
  near_chemistry_boundary — True if chemistry is within 0.1 dex of -0.2 or -1.0
 
=== CLASSIFICATION PROTOCOL ===
 
--- Step A — Confidence Ceiling ---
 
  quality_score >= 0.80  →  ceiling = 1.0
  0.50 <= qs < 0.80      →  ceiling = 0.85
  0.10 <= qs < 0.50      →  ceiling = 0.60
  qs == 0.0              →  ceiling = 0.30
  is_binary_candidate    →  ceiling = ceiling - 0.15
 
--- Step B — Sub-type Range (within the HC letter) ---
 
Using ONLY the sub-type table for the letter in hc_anchor.mk_letter:
 
  O:  >=60000->"0-1" | 45000-60000->"2-3" | 35000-45000->"4-6" | 30000-35000->"7-9"
  B:  25000-30000->"0-1" | 18000-25000->"2-3" | 13000-18000->"4-6" | 10000-13000->"7-9"
  A:  9000-9800->"0-1"   | 8400-9000->"2-3"   | 7900-8400->"4-6"   | 7500-7900->"7-9"
  F:  7200-7500->"0-1"   | 6900-7200->"2-3"   | 6500-6900->"4-5"   | 6200-6500->"6-7" | 6000-6200->"8-9"
  G:  5800-6000->"0-1"   | 5600-5800->"2-3"   | 5450-5600->"4-5"   | 5300-5450->"6-7" | 5200-5300->"8-9"
  K:  5000-5200->"0-1"   | 4700-5000->"2-3"   | 4300-4700->"4-5"   | 4000-4300->"6-7" | 3700-4000->"8-9"
  M:  3400-3700->"0-1"   | 3100-3400->"2-3"   | 2800-3100->"4-5"   | <2800->"6-9"
 
  If near_teff_boundary = True: spectral_type_confidence = ceiling - 0.10
 
--- Step C — Luminosity Class from logg ---
 
  I: logg<1.5 | II: 1.5-2.5 | III: 2.5-3.5 | IV: 3.5-3.9 | V: >=3.9
  If near_logg_boundary = True: luminosity_confidence = ceiling - 0.10
  If abs_mag available and disagrees with logg class: luminosity_confidence - 0.05
 
--- Step D — Confidence Scores ---
 
  spectral_type_confidence: ceiling [- 0.10 if near_teff_boundary]
  luminosity_confidence:    ceiling [- 0.10 if near_logg_boundary] [- 0.05 if abs_mag disagrees]
  population_confidence:    ceiling [- 0.10 if chemistry_source=metallicity_fallback]
                                    [- 0.20 if near_chemistry_boundary]
 
--- Step E — Stellar Description (English, structured) ---
 
Write three fields using ONLY values present in hc_anchor and the physical_vector.
DO NOT mention stellar mass, radius, age, catalog names, or star names.
DO NOT invent values not present in the input contract.
WORD LIMITS ARE STRICT — responses exceeding limits will be rejected.
 
  physical_profile — EXACTLY these 4 sentences, no more:
    Sentence 1: "This [spectral_type][sub_type_range] [luminosity_class] star has teff_k=[X] K and logg=[X]."
    Sentence 2: "The absolute magnitude is [X], consistent with luminosity class [Y]." (skip if abs_mag is null)
    Sentence 3: One sentence interpreting what the logg and luminosity class imply (dwarf/giant/subgiant).
    Sentence 4: One sentence about notable chemistry if alpha_fe or fe_h is available. Otherwise omit.
    STOP after sentence 4. Do not add more sentences.
 
  population_context — EXACTLY 1-2 sentences:
    Sentence 1: "chemistry_value=[X] ([chemistry_source]), v_tan=[X] km/s -> [population_group]."
    Sentence 2 (optional): One sentence on alpha_fe or near_chemistry_boundary if relevant. Otherwise omit.
    STOP. Do not add more sentences.
 
  notable_features — ONE sentence only:
    If is_binary_candidate=True: note binary flag and relevant diagnostic (RUWE or NSS).
    If has_emission=True: note H-alpha emission from ESP-ELS catalog.
    If is_metal_poor=True: note metal-poor flag.
    If is_high_velocity=True: note high tangential velocity value.
    If quality_score < 0.5: note reduced data quality.
    If none of the above: write exactly "None identified." and nothing else.
    STOP after one sentence. Do not combine multiple flags into multiple sentences.
 
=== TAXONOMY ===
 
spectral_type:    the mk_letter from hc_anchor (fixed)
sub_type_range:   bin from Step B table
luminosity_class: {I, II, III, IV, V}
population_group: the population_group from hc_anchor (fixed)
prompt_version:   always "v5"
 
=== RESPONSE FORMAT ===
 
Respond ONLY with this JSON. No text before or after.
 
{
  "source_id": "<copy exactly from input>",
  "prompt_version": "v5",
  "classification": {
    "spectral_type": "<mk_letter from hc_anchor>",
    "sub_type_range": "<bin from Step B>",
    "luminosity_class": "<class from Step C>",
    "population_group": "<population_group from hc_anchor>"
  },
  "confidence_scores": {
    "spectral_type_confidence": <float>,
    "luminosity_confidence": <float>,
    "population_confidence": <float>
  },
  "stellar_description": {
    "physical_profile": "<4 sentences MAX — teff_k, logg, luminosity interpretation, chemistry>",
    "population_context": "<1-2 sentences MAX — chemistry_value, v_tan, population_group>",
    "notable_features": "<1 sentence MAX, or exactly 'None identified.'>"
  },
  "technical_reasoning": "<Step A: qs=X ceiling=Y. Step B: teff_k=X letter=Y(HC) -> sub_type_range=Z. Step C: logg=X -> luminosity_class=Y. Step D: sp_conf=X lum_conf=X pop_conf=X.>"
}"""
 
 
# ---------------------------------------------------------------------------
# Prompt builders v5
# ---------------------------------------------------------------------------
 
def _format_physical_vector(pv: dict) -> dict:
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
    Full prompt with hc_anchor block injected.
    The ground_truth key is excluded — never shown to the LLM.
    """
    source_id  = str(star.get("source_id", "UNKNOWN"))
    qs         = star.get("quality_score", 0.0)
    pv         = _format_physical_vector(star.get("physical_vector", {}))
    lf         = star.get("logical_flags", {})
    bd         = star.get("binary_diagnostics", {})
    ss         = star.get("spectral_summary", {})
    hc_anchor  = build_hc_anchor(star)
 
    compact = {
        "source_id":      source_id,
        "quality_score":  qs,
        "hc_anchor":      hc_anchor,
        "physical_vector": pv,
        "logical_flags":  lf,
        "binary_diagnostics": {
            "ruwe":                  bd.get("ruwe"),
            "rv_error_km_s":         bd.get("rv_error_km_s"),
            "nss_solution":          bd.get("nss_solution"),
        },
        "spectral_summary": {
            "halpha_catalog": ss.get("halpha_catalog", {}),
            "cat_triplet":    _format_cat_triplet(ss.get("cat_triplet", [])),
            "bprp_continuum": {k: v for k, v in ss.get("bprp_continuum", {}).items()
                               if k in ("status", "snr", "stable", "high_snr")},
            "rvs_continuum":  {k: v for k, v in ss.get("rvs_continuum", {}).items()
                               if k in ("status", "snr", "stable", "high_snr")},
        },
    }
 
    return (
        f"Classify the following Gaia DR3 star using AstroSage-Llama v5.0.\n\n"
        f"The hc_anchor block contains HC pre-computed values. "
        f"Use mk_letter and population_group exactly as given — do not change them.\n\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n\n"
        f"source_id: {source_id} | prompt_version must be 'v5'\n"
        f"Respond ONLY with the JSON object."
    )
 
 
def build_prompt_reduced(star: dict) -> str:
    """Reduced prompt for retry attempts."""
    source_id  = str(star.get("source_id", "UNKNOWN"))
    qs         = star.get("quality_score", 0.0)
    pv         = _format_physical_vector(star.get("physical_vector", {}))
    lf         = star.get("logical_flags", {})
    ss         = star.get("spectral_summary", {})
    hc_anchor  = build_hc_anchor(star)
 
    cat_ok = sum(1 for line in ss.get("cat_triplet", [])
                 if line.get("high_quality_fit", False))
 
    compact = {
        "source_id":     source_id,
        "quality_score": qs,
        "hc_anchor":     hc_anchor,
        "physical_vector": {k: pv[k] for k in
                            ("teff_k","logg","abs_mag","fe_h","metallicity","alpha_fe","v_tan")
                            if k in pv},
        "logical_flags": {k: lf.get(k) for k in (
            "is_reliable_parallax","is_giant","is_binary_candidate",
            "is_high_velocity","has_emission","is_metal_poor",
        )},
        "cat_lines_succeeded": cat_ok,
        "halpha_ew_aa": ss.get("halpha_catalog", {}).get("ew_aa"),
    }
 
    return (
        f"Classify this Gaia DR3 star. v5: hc_anchor provides mk_letter and population_group (fixed).\n\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n\n"
        f"source_id: {source_id} | prompt_version must be 'v5'\n"
        f"Respond ONLY with the JSON object."
    )
 
 
def estimate_prompt_tokens(star: dict, reduced: bool = False) -> int:
    user_text   = build_prompt_reduced(star) if reduced else build_prompt(star)
    total_chars = len(SYSTEM_PROMPT) + len(user_text)
    return total_chars // 4
