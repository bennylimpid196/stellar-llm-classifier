"""
system_prompt_v6.py — SC System Prompt for V6
==============================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0
System: STELLAR (Spectral Type Estimation via Language Learning and Astronomical Reasoning)

Changes vs V5
-------------
V6-CHG-1 | SOFT A/B ANCHOR (near_teff_boundary)
    Motivation: empirical analysis of V5 results shows that when HC assigns letter A
    with near_teff_boundary=True, the LLM achieves 66.7% accuracy (vs HC's 50%) by
    implicitly using logg to distinguish warm B subgiants (logg~3.4-3.7) from true A
    dwarfs (logg~3.9-4.3). The optimal rule logg>=3.80→A achieves 83.3% accuracy.
    Fix: when near_teff_boundary=True for A/B boundary stars, the prompt explicitly
    instructs the LLM to consider logg as a tiebreaker rather than treating the HC
    letter as a hard anchor.

V6-CHG-2 | B SUBTYPE FEW-SHOT
    Motivation: V5 collapsed B subtype to "0-1" for 66/70 B stars (94%). The corpus
    contains B stars with Teff 10,000-14,083K (median 10,715K), which correspond to
    subtypes B4-9, not B0-1. The model's LLM prior strongly associates "B star" with
    B0-B1 (the most cited in literature). Few-shot examples with B4-6 and B7-9 should
    break this prior.

V6-CHG-3 | NOTABLE FEATURES — EXPLICIT FLAG RULES
    Motivation: V5 wrote "None identified." for 66/111 stars (59.5%) with active flags.
    ROUGE-1 for notable_features collapsed to 0.029 (mean) across all 11 reference stars
    that had active flags. When the LLM did write something, it was qualitatively correct
    ("H-alpha emission, chromospherically active star"). The problem is activation, not
    capability. Explicit conditional rules per flag with value templates should fix this.

Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 6.0
"""

from __future__ import annotations

import json
import textwrap
from typing import Any


# ---------------------------------------------------------------------------
# Output schema (unchanged from V5)
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA = {
    "source_id": "string",
    "prompt_version": "v6",
    "classification": {
        "spectral_type": "one of: O B A F G K M",
        "sub_type_range": "string e.g. '0-1', '2-3', '4-5', '6-7', '7-9'",
        "luminosity_class": "one of: I II III IV V",
        "population_group": "one of: Halo Disco Grueso Disco Fino",
    },
    "confidence_scores": {
        "spectral_type_confidence": "float 0.0-1.0",
        "luminosity_confidence": "float 0.0-1.0",
        "population_confidence": "float 0.0-1.0",
    },
    "stellar_description": {
        "physical_profile": "4 sentences MAX about Teff, logg, luminosity, chemistry",
        "population_context": "1-2 sentences about chemistry_value, v_tan, population",
        "notable_features": "1 sentence per active flag. 'None identified.' ONLY if ALL flags are False.",
    },
    "technical_reasoning": "string — brief step-by-step justification",
}

# ---------------------------------------------------------------------------
# V6-CHG-2 Few-shot examples for B subtype
# Real corpus stars with confirmed ground truth, n_pastel >= 1
# ---------------------------------------------------------------------------

B_SUBTYPE_FEWSHOT = """
## B Subtype Reference Examples

The following examples show how Teff maps to B subtypes. Use these as calibration.
The B sequence covers ~10,000-30,000K. Most stars in this corpus are late-B (B4-B9).

Example B4-6 (Teff ~13,000-18,000K):
  teff_k=13274, logg=3.589, abs_mag=-1.698 → sub_type_range: "4-6"
  Ground truth: B5V  (n_pastel=2)

Example B4-6 (Teff ~13,250K, giant):
  teff_k=13250, logg=3.196, abs_mag=-2.002 → sub_type_range: "4-6"
  Ground truth: B7III  (n_pastel=2)

Example B7-9 (Teff ~10,000-13,000K):
  teff_k=11544, logg=4.088, abs_mag=0.837 → sub_type_range: "7-9"
  Ground truth: B9V  (n_pastel=10)

Example B7-9 (Teff ~12,043K, peculiar):
  teff_k=12043, logg=3.747, abs_mag=-1.008 → sub_type_range: "7-9"
  Ground truth: B9pHgMn  (n_pastel=6)

Subtype bins for B:
  B0-1: Teff 25,000-35,000 K  (very rare in Gaia DR3 solar neighborhood sample)
  B2-3: Teff 18,000-25,000 K
  B4-6: Teff 13,000-18,000 K
  B7-9: Teff 10,000-13,000 K  ← most B stars in this corpus fall here
"""

# ---------------------------------------------------------------------------
# V6-CHG-3 Notable features rules with value templates
# ---------------------------------------------------------------------------

NOTABLE_FEATURES_RULES = """
## Notable Features — Mandatory Rules

RULE 1: If ALL flags are False → write exactly: "None identified."
RULE 2: If ANY flag is True → you MUST describe it. "None identified." is FORBIDDEN.
RULE 3: Write one sentence per active flag. Use the templates below.

Templates (substitute {value} with the actual number from the physical vector):

is_binary_candidate=True:
  → "Binary candidate flagged by RUWE={ruwe_value}; physical parameters (Teff, logg,
     chemistry) may reflect flux contamination from an unresolved companion, and the
     spectral classification should be treated with corresponding caution."

has_emission=True:
  → "H-alpha emission detected (EW={ew_ha_value} Å, ESP-ELS); consistent with
     chromospheric activity, a Be-type disk, or an active stellar phenomenon."

is_metal_poor=True:
  → "Metal-poor star ([Fe/H]≈{chemistry_value} dex); the sub-solar chemistry is
     consistent with membership in an old stellar population."

is_high_velocity=True:
  → "High tangential velocity (v_tan={v_tan_value} km/s); this kinematic outlier
     strongly supports Halo membership regardless of photometric metallicity."

Multiple flags: write one sentence per flag, concatenated in the same field.
"""

# ---------------------------------------------------------------------------
# Main system prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """\
You are AstroSage, a specialized astrophysical reasoning model for stellar spectral
classification. You receive pre-processed data from the Hard Computing (HC) module of
the STELLAR pipeline (Gaia DR3). Your task is to assign sub-type, luminosity class,
and produce a structured stellar description.

## Your role in the HC+SC hybrid architecture

The HC module has already computed:
  - spectral_type (letter): deterministic from Teff via MK calibration
  - population_group: deterministic from kinematics (v_tan) + chemistry
  - All logical flags (is_binary_candidate, has_emission, etc.)

Your job is to determine:
  1. sub_type_range (numeric bin within the letter class)
  2. luminosity_class (I/II/III/IV/V from logg context)
  3. confidence_scores (calibrated — lower when flags are active or near boundaries)
  4. stellar_description (three structured fields in English)

{b_subtype_fewshot}

{notable_features_rules}

## Confidence calibration rules

- Base confidence ceiling = quality_score (provided per star)
- is_binary_candidate=True → multiply spectral_type_confidence × 0.85
- near_teff_boundary=True → multiply spectral_type_confidence × 0.90
- near_logg_boundary=True → multiply luminosity_confidence × 0.90
- Do NOT report confidence > quality_score ceiling
- Stars where you are uncertain: lower confidence, do not force high values

## Output format

Respond ONLY with valid JSON. No preamble, no markdown, no explanation outside the JSON.

{schema}
"""


def build_system_prompt() -> str:
    """Returns the complete system prompt string for V6."""
    schema_str = json.dumps(OUTPUT_SCHEMA, indent=2, ensure_ascii=False)
    return SYSTEM_PROMPT_BASE.format(
        b_subtype_fewshot=B_SUBTYPE_FEWSHOT,
        notable_features_rules=NOTABLE_FEATURES_RULES,
        schema=schema_str,
    )


# ---------------------------------------------------------------------------
# V6-CHG-1 Soft A/B anchor — user prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(star: dict) -> str:
    """
    Builds the per-star user prompt for V6.

    Key change vs V5: when hc_anchor.mk_letter == 'A' and
    near_teff_boundary == True, the anchor is presented as a soft suggestion
    with an explicit logg-based disambiguation instruction. For all other
    letters, the anchor is presented as a hard constraint.

    Args:
        star: one entry from stellar_corpus_v5.json

    Returns:
        str: formatted user prompt string
    """
    source_id = star["source_id"]
    anchor    = star["hc_anchor"]
    pv        = star["physical_vector"]
    flags     = star["logical_flags"]
    spec_sum  = star.get("spectral_summary", {})

    mk_letter   = anchor["mk_letter"]
    pop_group   = anchor["population_group"]
    near_teff   = anchor.get("near_teff_boundary", False)
    near_logg   = anchor.get("near_logg_boundary", False)
    near_chem   = anchor.get("near_chemistry_boundary", False)
    quality     = anchor.get("quality_score", star.get("quality_score", 0.5))

    teff_k      = anchor["teff_k"]
    logg        = anchor["logg"]
    abs_mag     = anchor.get("abs_mag")
    v_tan       = anchor.get("v_tan")
    chem_val    = anchor.get("chemistry_value")
    alpha_fe    = anchor.get("alpha_fe")
    fe_h        = pv.get("fe_h")
    metallicity = pv.get("metallicity")
    ew_ha       = pv.get("ew_ha")
    ruwe        = pv.get("ruwe")

    # ── V6-CHG-1: soft anchor for A/B boundary ──────────────────────────────
    # Empirical finding from V5: logg >= 3.80 → A (83.3% accuracy),
    # logg < 3.80 → B (subgiant / evolved B cooling toward A boundary).
    # When near_teff_boundary=True on an A star, give the LLM the context
    # to make the call itself rather than forcing the HC letter.

    if mk_letter == "A" and near_teff:
        if logg is not None and logg >= 3.80:
            letter_block = (
                f"HC spectral letter: A [BOUNDARY — Teff={teff_k:.0f}K, limit=10000K]\n"
                f"logg={logg:.3f} (≥3.80 → consistent with dwarf/subgiant A classification)\n"
                f"Use A as the spectral_type unless logg and luminosity context strongly indicate B."
            )
        else:
            letter_block = (
                f"HC spectral letter: A [BOUNDARY — Teff={teff_k:.0f}K, limit=10000K]\n"
                f"logg={logg:.3f} (<3.80 → may indicate evolved B star cooling toward A boundary)\n"
                f"Consider whether B is more appropriate given the lower surface gravity. "
                f"Assign the spectral_type using your astrophysical judgment on Teff + logg."
            )
    else:
        letter_block = (
            f"HC spectral letter: {mk_letter} [FIXED — use exactly this letter as spectral_type]"
        )

    # ── Population block (always from HC, LLM copies it) ────────────────────
    pop_block = (
        f"HC population: {pop_group} [FIXED — use exactly this as population_group]\n"
        f"  chemistry_value={chem_val}  alpha_fe={alpha_fe}  v_tan={v_tan:.1f} km/s"
        if v_tan is not None else
        f"HC population: {pop_group} [FIXED — use exactly this as population_group]\n"
        f"  chemistry_value={chem_val}  alpha_fe={alpha_fe}"
    )

    # ── Flags block (V6-CHG-3 context values for templates) ─────────────────
    flag_lines = []
    if flags.get("is_binary_candidate"):
        flag_lines.append(f"  is_binary_candidate: True  (ruwe={ruwe})")
    else:
        flag_lines.append(f"  is_binary_candidate: False")

    if flags.get("has_emission"):
        flag_lines.append(f"  has_emission: True  (ew_ha={ew_ha} Å)")
    else:
        flag_lines.append(f"  has_emission: False")

    if flags.get("is_metal_poor"):
        flag_lines.append(f"  is_metal_poor: True  (chemistry_value={chem_val} dex)")
    else:
        flag_lines.append(f"  is_metal_poor: False")

    if flags.get("is_high_velocity"):
        flag_lines.append(f"  is_high_velocity: True  (v_tan={v_tan} km/s)")
    else:
        flag_lines.append(f"  is_high_velocity: False")

    flags_block = "\n".join(flag_lines)

    # ── Boundary warnings ────────────────────────────────────────────────────
    boundary_notes = []
    if near_logg:
        boundary_notes.append("near_logg_boundary=True → luminosity class uncertain, lower luminosity_confidence")
    if near_chem:
        boundary_notes.append("near_chemistry_boundary=True → population assignment uncertain")
    boundary_str = "\n".join(boundary_notes) if boundary_notes else "None"

    # ── Spectral features from HC ────────────────────────────────────────────
    cat_lines = spec_sum.get("cat_lines_detected", [])
    cat_ews   = spec_sum.get("cat_ew_summary", {})
    bp_snr    = spec_sum.get("bprp_snr", None)
    rvs_snr   = spec_sum.get("rvs_snr", None)

    prompt = textwrap.dedent(f"""\
        Classify the following star from Gaia DR3.

        ## HC Anchor (Hard Computing output — respect these)

        {letter_block}
        {pop_block}

        ## Physical vector

        source_id   : {source_id}
        teff_k      : {teff_k:.1f} K
        logg        : {logg}
        abs_mag     : {abs_mag}
        metallicity : {metallicity}   (photometric [M/H])
        fe_h        : {fe_h}          (spectroscopic [Fe/H], higher precision when available)
        alpha_fe    : {alpha_fe}
        v_tan       : {v_tan} km/s
        ew_ha       : {ew_ha} Å       (negative = emission)
        ruwe        : {ruwe}

        ## Quality

        quality_score : {quality}  ← confidence ceiling for this star
        Boundary warnings: {boundary_str}

        ## Logical flags

        {flags_block}

        ## Spectral diagnostics (HC)

        CaT lines detected : {cat_lines}
        CaT EW summary     : {cat_ews}
        BP/RP SNR          : {bp_snr}
        RVS SNR            : {rvs_snr}

        ## Instructions

        1. spectral_type: use the HC letter above (or apply logg judgment if BOUNDARY noted).
        2. sub_type_range: derive from teff_k using the B subtype examples above if letter=B.
           For other letters use standard MK Teff calibration.
        3. luminosity_class: derive from logg (I≈1.0, II≈2.0, III≈3.0, IV≈3.7, V≈4.3).
        4. population_group: copy exactly from HC — do not change.
        5. confidence_scores: apply the calibration rules from the system prompt.
        6. stellar_description: follow all rules for notable_features (see system prompt).
           Physical_profile: max 4 sentences. Population_context: max 2 sentences.
           Notable_features: one sentence per active flag; "None identified." ONLY if all flags False.
        7. technical_reasoning: brief step-by-step (A→letter, B→sub_type, C→lum, D→pop, E→flags).

        Output ONLY valid JSON. No markdown, no preamble.
    """)
    return prompt


# ---------------------------------------------------------------------------
# Convenience: build full message list for inference_manager
# ---------------------------------------------------------------------------

def build_messages(star: dict) -> list[dict]:
    """Returns [system_msg, user_msg] for the HuggingFace pipeline."""
    return [
        {"role": "system",  "content": build_system_prompt()},
        {"role": "user",    "content": build_user_prompt(star)},
    ]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Load first B star from corpus for spot check
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "Data/stellar_corpus_v5.json"
    try:
        with open(corpus_path) as f:
            corpus = json.load(f)
        b_stars = [s for s in corpus if s["hc_anchor"]["mk_letter"] == "B"]
        boundary_a = [s for s in corpus
                      if s["hc_anchor"]["mk_letter"] == "A"
                      and s["hc_anchor"]["near_teff_boundary"]]

        print("=== SYSTEM PROMPT (first 800 chars) ===")
        sp = build_system_prompt()
        print(sp[:800])

        print("\n=== USER PROMPT — B star (spot check V6-CHG-2) ===")
        print(build_user_prompt(b_stars[0]))

        print("\n=== USER PROMPT — A near_boundary (spot check V6-CHG-1) ===")
        if boundary_a:
            print(build_user_prompt(boundary_a[0]))
        else:
            print("No boundary A stars found")

    except FileNotFoundError:
        print(f"Corpus not found at {corpus_path}")
        print("Run: python3 system_prompt_v6.py Data/stellar_corpus_v5.json")
