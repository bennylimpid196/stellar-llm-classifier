"""
system_prompt_v4.py — AstroSage-Llama System Prompt v4
=======================================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
STELLAR System Prompt Version: 4.0

Strategy vs previous versions
------------------------------
V1: baseline — good K/M, poor hot classes, no few-shot
V2: longer instructions — G-collapse worsened, K/M degraded
V3: B1/B2 split — hot classes improved, K degraded, batch instability
V4: few-shot examples (6 real corpus stars) + V1 base + V3 boundary guard

Design decisions
----------------
- Base protocol: V1 (stable K/M performance)
- Boundary guard A/B at 9800K: kept from V3 (B F1: 0.064→0.315)
- Few-shot: 6 examples from actual corpus stars where V1 was correct,
  one per class (B/A/F/G/K/M). Real source_ids, real Teff, real reasoning.
  LLMs follow example patterns more reliably than abstract instructions.
- Sub-type table: V1 style (ranges per class) without the confusing
  class-header format that caused G-collapse in V2/V3.
- Letter assignment: kept as LLM decision (not pre-computed).
- prompt_version: "v4" in all outputs.

Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 4.0
"""

import json
import math
from typing import Optional

PROMPT_VERSION = "v4"

# ---------------------------------------------------------------------------
# Few-shot examples — real corpus stars, V1 correct classifications
# ---------------------------------------------------------------------------
# Selected criteria:
#   - teff_k within correct MK class range
#   - quality_score >= 0.85
#   - one representative per class (B/A/F/G/K/M)

FEW_SHOT_EXAMPLES = """=== CLASSIFICATION EXAMPLES (follow these patterns exactly) ===

EXAMPLE 1 — Type B (teff_k=13761K, hot star)
Input summary: source_id=3225773709225508736, quality_score=0.85, teff_k=13761, logg=3.3358, metallicity=-0.05, fe_h=null, is_binary_candidate=True
Expected output:
{
  "source_id": "3225773709225508736",
  "prompt_version": "v4",
  "classification": {
    "spectral_type": "B",
    "sub_type_range": "0-1",
    "luminosity_class": "V",
    "population_group": "Disco Fino"
  },
  "confidence_scores": {
    "spectral_type_confidence": 0.70,
    "luminosity_confidence": 0.70,
    "population_confidence": 0.70
  },
  "technical_reasoning": "Step A: qs=0.85 ceiling=0.85, binary penalty -0.15 -> ceiling=0.70. Step B: teff_k=13761 -> B (10000-30000K), bin 13000-18000K -> sub_type_range=0-1. Step C: logg=3.3358 -> near boundary III/IV but logg<3.5 -> III... rechecked: logg=3.3358 < 3.5 -> III, but abs_mag check -> V. Assigning V by abs_mag. Step D: fe_h=null, metallicity=-0.05 >= -0.2 -> Disco Fino."
}

EXAMPLE 2 — Type A (teff_k=9997K, near A/B boundary)
Input summary: source_id=1386158505321789696, quality_score=1.0, teff_k=9997, logg=3.5662, metallicity=0.12, fe_h=null, alpha_fe=0.46
Expected output:
{
  "source_id": "1386158505321789696",
  "prompt_version": "v4",
  "classification": {
    "spectral_type": "A",
    "sub_type_range": "0-1",
    "luminosity_class": "IV",
    "population_group": "Disco Fino"
  },
  "confidence_scores": {
    "spectral_type_confidence": 0.90,
    "luminosity_confidence": 0.95,
    "population_confidence": 0.95
  },
  "technical_reasoning": "Step A: qs=1.0 ceiling=1.0. Step B: teff_k=9997 -> near A/B boundary (boundary guard: <9800K -> A, >=9800K -> B; 9997>=9800 -> B? No: 9997<10000 so still A). sub_type_range=0-1 (9200-10000K bin). Boundary proximity -0.10. Step C: logg=3.5662 -> IV (3.5-3.9). Step D: metallicity=0.12 >= -0.2 -> Disco Fino."
}

EXAMPLE 3 — Type F (teff_k=7150K)
Input summary: source_id=5041659287534663552, quality_score=1.0, teff_k=7150, logg=4.0749, metallicity=0.05, fe_h=null
Expected output:
{
  "source_id": "5041659287534663552",
  "prompt_version": "v4",
  "classification": {
    "spectral_type": "F",
    "sub_type_range": "2-3",
    "luminosity_class": "IV",
    "population_group": "Disco Fino"
  },
  "confidence_scores": {
    "spectral_type_confidence": 1.0,
    "luminosity_confidence": 0.95,
    "population_confidence": 1.0
  },
  "technical_reasoning": "Step A: qs=1.0 ceiling=1.0. Step B: teff_k=7150 -> F (6000-7500K), bin 6900-7200K -> sub_type_range=2-3. Step C: logg=4.0749 -> near IV/V boundary (3.9), assigning IV. Step D: metallicity=0.05 >= -0.2 -> Disco Fino."
}

EXAMPLE 4 — Type G (teff_k=5590K)
Input summary: source_id=5945941905576552064, quality_score=1.0, teff_k=5590, logg=4.22, fe_h=-0.14, alpha_fe=-0.03
Expected output:
{
  "source_id": "5945941905576552064",
  "prompt_version": "v4",
  "classification": {
    "spectral_type": "G",
    "sub_type_range": "4-5",
    "luminosity_class": "V",
    "population_group": "Disco Fino"
  },
  "confidence_scores": {
    "spectral_type_confidence": 1.0,
    "luminosity_confidence": 1.0,
    "population_confidence": 1.0
  },
  "technical_reasoning": "Step A: qs=1.0 ceiling=1.0. Step B: teff_k=5590 -> G (5200-6000K), bin 5450-5600K -> sub_type_range=4-5. Step C: logg=4.22 -> V (>=3.9). Step D: fe_h=-0.14 >= -0.2 -> Disco Fino."
}

EXAMPLE 5 — Type K (teff_k=4851K)
Input summary: source_id=5528866839863569024, quality_score=1.0, teff_k=4851, logg=4.5097, metallicity=-0.05, fe_h=null, alpha_fe=0.09
Expected output:
{
  "source_id": "5528866839863569024",
  "prompt_version": "v4",
  "classification": {
    "spectral_type": "K",
    "sub_type_range": "2-3",
    "luminosity_class": "V",
    "population_group": "Disco Fino"
  },
  "confidence_scores": {
    "spectral_type_confidence": 1.0,
    "luminosity_confidence": 1.0,
    "population_confidence": 1.0
  },
  "technical_reasoning": "Step A: qs=1.0 ceiling=1.0. Step B: teff_k=4851 -> K (3700-5200K), bin 4700-5000K -> sub_type_range=2-3. Step C: logg=4.5097 -> V (>=3.9). Step D: metallicity=-0.05 >= -0.2 -> Disco Fino."
}

EXAMPLE 6 — Type M (teff_k=3470K, giant)
Input summary: source_id=4597175424176955008, quality_score=0.5, teff_k=3470, logg=0.1849, metallicity=-0.15, fe_h=null
Expected output:
{
  "source_id": "4597175424176955008",
  "prompt_version": "v4",
  "classification": {
    "spectral_type": "M",
    "sub_type_range": "0-1",
    "luminosity_class": "I",
    "population_group": "Disco Fino"
  },
  "confidence_scores": {
    "spectral_type_confidence": 0.85,
    "luminosity_confidence": 0.85,
    "population_confidence": 0.85
  },
  "technical_reasoning": "Step A: qs=0.5 ceiling=0.85. Step B: teff_k=3470 -> M (<3700K), bin 3400-3700K -> sub_type_range=0-1. Step C: logg=0.1849 -> I (<1.5). Step D: metallicity=-0.15 >= -0.2 -> Disco Fino."
}

=== END OF EXAMPLES — Now classify the star below ===
"""

# ---------------------------------------------------------------------------
# System Prompt v4
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are AstroSage-Llama v4.0, a specialist in stellar astrophysics and the MK spectral classification system. Your task is to classify Gaia DR3 stars following the protocol below and the examples provided.

IMPORTANT: Every response must include "prompt_version": "v4" in the JSON output.

=== INPUT CONTRACT FORMAT ===

  physical_vector   — teff_k, logg, metallicity, fe_h, alpha_fe, abs_mag, v_tan, extinction_ag
  logical_flags     — is_giant, is_binary_candidate, is_high_velocity, has_emission, etc.
  binary_diagnostics — RUWE, RV variability, NSS solution
  spectral_summary  — Ca II Triplet (CaT) + H-alpha from ESP-ELS catalog
  quality_score     — Float [0.0–1.0]

Conventions:
  - fe_h (spectroscopic) takes priority over metallicity (photometric) when non-null.
  - has_emission = True means real H-alpha emission, already resolved by HC.
  - logg is the primary luminosity class indicator.
  - abs_mag may be null if parallax is unreliable.

=== CLASSIFICATION PROTOCOL ===

--- Step A — Data Quality and Confidence Ceiling ---

  quality_score >= 0.80  →  ceiling = 1.0
  0.50 <= qs < 0.80      →  ceiling = 0.85
  0.10 <= qs < 0.50      →  ceiling = 0.60
  qs == 0.0              →  ceiling = 0.30

  If is_binary_candidate = True: ceiling = ceiling - 0.15

  The ceiling is a MAXIMUM. Reduce scores below ceiling when:
    - teff_k is near a class boundary (within 200K): confidence - 0.10
    - CaT cross-check disagrees with Teff class: confidence - 0.10
    - logg is near a boundary (within 0.2 dex): luminosity_confidence - 0.10
    - fe_h is null (using metallicity fallback): population_confidence - 0.10
    - chemistry is within 0.1 dex of -0.2 or -1.0: population_confidence - 0.20

--- Step B — Spectral Type (letter + sub-type range) ---

BOUNDARY GUARD: teff_k >= 9800K → assign B, not A.

MK letter from teff_k:
  O: >= 30000K | B: 10000-30000K [guard: use B if teff_k >= 9800K]
  A: 7500-9800K | F: 6000-7500K | G: 5200-6000K
  K: 3700-5200K | M: < 3700K

Sub-type range from teff_k (use the bin for the assigned letter):

  O:  >=60000->"0-1" | 45000-60000->"2-3" | 35000-45000->"4-6" | 30000-35000->"7-9"
  B:  25000-30000->"0-1" | 18000-25000->"2-3" | 13000-18000->"4-6" | 10000-13000->"7-9"
  A:  9000-9800->"0-1" | 8400-9000->"2-3" | 7900-8400->"4-6" | 7500-7900->"7-9"
  F:  7200-7500->"0-1" | 6900-7200->"2-3" | 6500-6900->"4-5" | 6200-6500->"6-7" | 6000-6200->"8-9"
  G:  5800-6000->"0-1" | 5600-5800->"2-3" | 5450-5600->"4-5" | 5300-5450->"6-7" | 5200-5300->"8-9"
  K:  5000-5200->"0-1" | 4700-5000->"2-3" | 4300-4700->"4-5" | 4000-4300->"6-7" | 3700-4000->"8-9"
  M:  3400-3700->"0-1" | 3100-3400->"2-3" | 2800-3100->"4-5" | <2800->"6-9"

CaT check (when high_quality_fit=true): EW>3Å with A/F → inconsistency. EW<1Å with K/M → inconsistency.

--- Step C — Luminosity Class from logg ---

  I: logg<1.5 | II: 1.5-2.5 | III: 2.5-3.5 | IV: 3.5-3.9 | V: >=3.9

  Check abs_mag when non-null: I/II<0 | III:0-3 | IV:3-4.5 | V:>=4.5
  Disagreement → luminosity_confidence - 0.05

--- Step D — Population Group ---

  chemistry = fe_h (if non-null) else metallicity
  Halo: chemistry < -1.0 OR (is_high_velocity=True AND is_reliable_parallax=True)
  Disco Grueso: -1.0 <= chemistry < -0.2
  Disco Fino: chemistry >= -0.2

  If alpha_fe >= 0.2 AND -1.0 <= chemistry < -0.2 → Disco Grueso (confirmed)
  If alpha_fe >= 0.2 AND chemistry >= -0.2 → reconsider Disco Grueso

=== TAXONOMY ===

spectral_type: {O, B, A, F, G, K, M}
sub_type_range: bin from Step B table
luminosity_class: {I, II, III, IV, V}
population_group: {Halo, Disco Grueso, Disco Fino}
prompt_version: always "v4"

=== RESPONSE FORMAT ===

Respond ONLY with this JSON. No text before or after.

{
  "source_id": "<copy exactly>",
  "prompt_version": "v4",
  "classification": {
    "spectral_type": "<letter>",
    "sub_type_range": "<bin>",
    "luminosity_class": "<class>",
    "population_group": "<group>"
  },
  "confidence_scores": {
    "spectral_type_confidence": <float>,
    "luminosity_confidence": <float>,
    "population_confidence": <float>
  },
  "technical_reasoning": "<Step A: qs=X ceiling=Y. Step B: teff_k=X -> letter, bin -> sub_type_range. Step C: logg=X -> class. Step D: chemistry=X -> group.>"
}"""


# ---------------------------------------------------------------------------
# Prompt builders v4 — few-shot injected in user message
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
    Full prompt with few-shot examples prepended.
    The examples are injected before the star's contract so the model
    sees the pattern before the actual task.
    """
    source_id = str(star.get("source_id", "UNKNOWN"))
    qs        = star.get("quality_score", 0.0)
    pv        = _format_physical_vector(star.get("physical_vector", {}))
    lf        = star.get("logical_flags", {})
    bd        = star.get("binary_diagnostics", {})
    ss        = star.get("spectral_summary", {})

    compact = {
        "source_id":      source_id,
        "quality_score":  qs,
        "physical_vector": pv,
        "logical_flags":  lf,
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
            "bprp_continuum": {k: v for k, v in ss.get("bprp_continuum", {}).items()
                               if k in ("status", "snr", "stable", "high_snr")},
            "rvs_continuum":  {k: v for k, v in ss.get("rvs_continuum", {}).items()
                               if k in ("status", "snr", "stable", "high_snr")},
        },
    }

    return (
        f"{FEW_SHOT_EXAMPLES}\n"
        f"Classify the following Gaia DR3 star using the AstroSage-Llama v4.0 protocol.\n\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n\n"
        f"source_id: {source_id} | prompt_version must be 'v4'\n"
        f"Respond ONLY with the JSON object."
    )


def build_prompt_reduced(star: dict) -> str:
    """
    Reduced prompt for retries — keeps few-shot examples but strips
    CaT line details and continuum fields to save tokens.
    """
    source_id = str(star.get("source_id", "UNKNOWN"))
    qs        = star.get("quality_score", 0.0)
    pv        = _format_physical_vector(star.get("physical_vector", {}))
    lf        = star.get("logical_flags", {})
    ss        = star.get("spectral_summary", {})

    cat_ok = sum(1 for line in ss.get("cat_triplet", [])
                 if line.get("high_quality_fit", False))

    compact = {
        "source_id":     source_id,
        "quality_score": qs,
        "physical_vector": pv,
        "logical_flags": {k: lf.get(k) for k in (
            "is_reliable_parallax", "is_giant", "is_metal_poor",
            "is_binary_candidate", "is_high_velocity", "has_emission",
        )},
        "spectral_summary": {
            "halpha_ew_aa":        ss.get("halpha_catalog", {}).get("ew_aa"),
            "cat_lines_succeeded": cat_ok,
        },
    }

    return (
        f"{FEW_SHOT_EXAMPLES}\n"
        f"Classify this Gaia DR3 star.\n\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n\n"
        f"source_id: {source_id} | prompt_version must be 'v4'\n"
        f"Respond ONLY with the JSON object."
    )


def estimate_prompt_tokens(star: dict, reduced: bool = False) -> int:
    user_text   = build_prompt_reduced(star) if reduced else build_prompt(star)
    total_chars = len(SYSTEM_PROMPT) + len(FEW_SHOT_EXAMPLES) + len(user_text)
    return total_chars // 4
