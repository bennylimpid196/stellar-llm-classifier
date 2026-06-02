"""
system_prompt_v7.py — SC System Prompt for V7
==============================================
Hybrid Stellar Classifier HC+SC — Pipeline Version HC-2.0-V7
System: STELLAR (Spectral Type Estimation via Language Learning and Astronomical Reasoning)

Changes vs V6
-------------
V7-CHG-1 | RAG CONTEXT INJECTION
    The user prompt now receives a === RETRIEVED CONTEXT === block with the
    top-3 knowledge base chunks retrieved by RAGEngine for this specific star.
    The LLM is instructed to use this context to calibrate subtype, luminosity,
    flag interpretation, and description format.

V7-CHG-2 | A/B BOUNDARY — HARD ANCHOR (replaces V6 soft anchor)
    V6 used a soft anchor for boundary A stars, asking the LLM to apply logg
    judgment. Analysis showed the LLM did not use logg as a discriminant (0 pp
    improvement). In V7, corpus_builder_v7.py applies the logg >= 3.8 rule
    deterministically and sets ab_boundary_logg_corrected=True. The user prompt
    presents the corrected letter as FIXED — no soft reasoning requested.

V7-CHG-3 | POPULATION GROUP — ENGLISH
    V6 emitted Spanish labels (Disco Fino, Disco Grueso, Halo).
    V7 corpus emits English labels (Thin Disk, Thick Disk, Halo).
    Output schema and all prompt text updated accordingly.

V7-CHG-4 | BINARY PENALTY — EXPLICIT FLOOR
    V6 multiplied spectral_type_confidence × 0.85 for binary candidates.
    Analysis showed the LLM applied the penalty nominally — binaries had
    HIGHER confidence than non-binaries (0.784 vs 0.762). V7 replaces the
    multiplicative rule with an explicit floor: confidence must be AT LEAST
    0.10 lower than it would be for a clean star of the same class.

V7-CHG-5 | POPULATION_CONTEXT WORD LIMIT — ENFORCED WITH EXAMPLE
    V6 violated the 30-word limit in 55.5% of stars, mainly in
    population_context. V7 adds an inline example showing a compliant
    sentence and explicitly states the word count before and after.

Author: Hybrid Stellar Classifier Project / CIMAT — STELLAR
Version: 7.0
"""

from __future__ import annotations

import json
import textwrap
from typing import Any


# ── Output schema ─────────────────────────────────────────────────────────────
# V7-CHG-3: population_group now in English

OUTPUT_SCHEMA = {
    "source_id": "string",
    "prompt_version": "v7",
    "classification": {
        "spectral_type": "one of: O B A F G K M",
        "sub_type_range": "string e.g. '0-1', '2-3', '4-5', '6-7', '7-9'",
        "luminosity_class": "one of: I II III IV V",
        "population_group": "one of: Halo Thick Disk Thin Disk",
    },
    "confidence_scores": {
        "spectral_type_confidence": "float 0.0-1.0",
        "luminosity_confidence": "float 0.0-1.0",
        "population_confidence": "float 0.0-1.0",
    },
    "stellar_description": {
        "physical_profile": "3-4 sentences about Teff, logg, luminosity, chemistry. Hard limit: 4 sentences.",
        "population_context": "1-2 sentences. Hard limit: 30 words total. Count before writing.",
        "notable_features": "1 sentence per active flag. 'None identified.' ONLY if ALL flags are False.",
    },
    "technical_reasoning": "string — brief step-by-step justification",
}


# ── B subtype few-shot (inherited from V6, unchanged) ─────────────────────────

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


# ── Notable features rules (inherited from V6, unchanged) ────────────────────

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


# ── System prompt ─────────────────────────────────────────────────────────────

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
- is_binary_candidate=True → spectral_type_confidence MUST be at least 0.10 lower
  than it would be for a clean star of the same spectral class. This is a hard floor,
  not a suggestion. A binary with quality_score=0.85 cannot have spectral_type_confidence > 0.75.
- near_teff_boundary=True → multiply spectral_type_confidence × 0.90
- near_logg_boundary=True → multiply luminosity_confidence × 0.90
- Do NOT report confidence > quality_score ceiling
- Stars where you are uncertain: lower confidence, do not force high values

## stellar_description format rules

physical_profile: exactly 3-4 sentences. Hard stop at 4. Do NOT write a 5th sentence.

population_context: maximum 30 words. Count your words before writing.
  COMPLIANT example (18 words):
    "Subsolar metallicity ([M/H] = -0.45) and alpha-enhancement ([α/Fe] = 0.28)
     place this star in the Thick Disk."
  If you exceed 30 words, rewrite and cut.

notable_features: follow the mandatory rules above.

## Output format

Respond ONLY with valid JSON. No preamble, no markdown, no explanation outside the JSON.

{schema}
"""


def build_system_prompt() -> str:
    """Returns the complete system prompt string for V7."""
    schema_str = json.dumps(OUTPUT_SCHEMA, indent=2, ensure_ascii=False)
    return SYSTEM_PROMPT_BASE.format(
        b_subtype_fewshot=B_SUBTYPE_FEWSHOT,
        notable_features_rules=NOTABLE_FEATURES_RULES,
        schema=schema_str,
    )


# ── User prompt ───────────────────────────────────────────────────────────────

def build_user_prompt(star: dict, rag_context: str = "") -> str:
    """
    Build the per-star user prompt for V7.

    V7-CHG-1: rag_context is injected as a === RETRIEVED CONTEXT === block
    when provided (non-empty string from RAGEngine.retrieve()).

    V7-CHG-2: The A/B boundary is now a hard anchor. corpus_builder_v7.py
    has already applied the logg >= 3.8 correction. The prompt presents the
    corrected letter as FIXED with a note that the builder already resolved
    the boundary — no soft reasoning is requested from the LLM.

    V7-CHG-3: population_group labels are now in English in the corpus.

    Args:
        star        : one entry from stellar_corpus_v7.json
        rag_context : str returned by RAGEngine.retrieve(star). Pass "" to
                      disable RAG injection (e.g. for ablation studies).

    Returns:
        str: formatted user prompt string
    """
    source_id = star["source_id"]
    anchor    = star["hc_anchor"]
    pv        = star["physical_vector"]
    flags     = star["logical_flags"]
    spec_sum  = star.get("spectral_summary", {})

    mk_letter    = anchor["mk_letter"]
    pop_group    = anchor["population_group"]   # English in V7
    near_teff    = anchor.get("near_teff_boundary", False)
    near_logg    = anchor.get("near_logg_boundary", False)
    near_chem    = anchor.get("near_chemistry_boundary", False)
    ab_corrected = anchor.get("ab_boundary_logg_corrected", False)
    quality      = anchor.get("quality_score", star.get("quality_score", 0.5))

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

    # ── V7-CHG-2: A/B boundary — hard anchor ─────────────────────────────────
    # corpus_builder_v7.py already applied logg >= 3.8 → A correction.
    # If ab_corrected=True, the letter has been changed from B to A by the
    # builder — present it as FIXED with a clarifying note.
    # If near_teff but not corrected, the star stays B — also FIXED.
    # No soft reasoning is requested in either case.

    if ab_corrected:
        letter_block = (
            f"HC spectral letter: {mk_letter} [FIXED — A/B boundary corrected by builder "
            f"(Teff={teff_k:.0f} K, logg={logg:.3f} >= 3.80 → A). Use exactly this letter.]"
        )
    elif near_teff:
        letter_block = (
            f"HC spectral letter: {mk_letter} [FIXED — near A/B boundary "
            f"(Teff={teff_k:.0f} K, logg={logg:.3f} < 3.80 → {mk_letter} confirmed). "
            f"Use exactly this letter.]"
        )
    else:
        letter_block = (
            f"HC spectral letter: {mk_letter} [FIXED — use exactly this letter as spectral_type]"
        )

    # ── Population block (V7-CHG-3: English labels) ───────────────────────────
    pop_block = (
        f"HC population: {pop_group} [FIXED — use exactly this as population_group]\n"
        f"  chemistry_value={chem_val}  alpha_fe={alpha_fe}  v_tan={v_tan:.1f} km/s"
        if v_tan is not None else
        f"HC population: {pop_group} [FIXED — use exactly this as population_group]\n"
        f"  chemistry_value={chem_val}  alpha_fe={alpha_fe}"
    )

    # ── Flags block (with values for notable_features templates) ─────────────
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

    # ── Boundary warnings ─────────────────────────────────────────────────────
    boundary_notes = []
    if near_logg:
        boundary_notes.append(
            "near_logg_boundary=True → luminosity class uncertain, lower luminosity_confidence"
        )
    if near_chem:
        boundary_notes.append(
            "near_chemistry_boundary=True → population assignment uncertain"
        )
    boundary_str = "\n".join(boundary_notes) if boundary_notes else "None"

    # ── Spectral diagnostics ──────────────────────────────────────────────────
    cat_lines = spec_sum.get("cat_lines_detected", [])
    cat_ews   = spec_sum.get("cat_ew_summary", {})
    bp_snr    = spec_sum.get("bprp_snr", None)
    rvs_snr   = spec_sum.get("rvs_snr", None)

    # ── V7-CHG-1: RAG context block ───────────────────────────────────────────
    if rag_context and rag_context.strip():
        rag_block = textwrap.dedent(f"""
            === RETRIEVED CONTEXT (from STELLAR knowledge base) ===
            The following chunks were retrieved as most relevant for this star.
            Use them to calibrate your subtype assignment, luminosity class,
            flag interpretation, and stellar_description format.

            {rag_context}

            === END RETRIEVED CONTEXT ===
        """).strip()
    else:
        rag_block = ""

    # ── Assemble prompt ───────────────────────────────────────────────────────
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

        {rag_block}

        ## Instructions

        1. spectral_type: use the HC letter above (FIXED — do not change).
        2. sub_type_range: derive from teff_k using the B subtype examples if letter=B.
           For other letters use standard MK Teff calibration.
        3. luminosity_class: derive from logg (I<1.5, II 1.5-2.5, III 2.5-3.5, IV 3.5-4.0, V>4.0).
        4. population_group: copy exactly from HC — do not change.
        5. confidence_scores: apply calibration rules from system prompt.
           If is_binary_candidate=True: spectral_type_confidence must be at least 0.10
           below the value you would assign to a clean star of the same class.
        6. stellar_description: physical_profile max 4 sentences; population_context
           max 30 words (count them); notable_features per flag rules.
        7. technical_reasoning: brief step-by-step (A→letter, B→sub_type, C→lum, D→pop, E→flags).

        Output ONLY valid JSON. No markdown, no preamble.
    """)

    # Strip leading blank line from rag_block area if RAG is empty
    return prompt.strip()


# ── Convenience: build full message list for inference_manager ────────────────

def build_messages(star: dict, rag_context: str = "") -> list[dict]:
    """
    Returns [system_msg, user_msg] for the HuggingFace pipeline.

    Args:
        star        : one entry from stellar_corpus_v7.json
        rag_context : str from RAGEngine.retrieve(star). Pass "" to disable.
    """
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": build_user_prompt(star, rag_context=rag_context)},
    ]


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "Data/stellar_corpus_v7.json"
    try:
        with open(corpus_path) as f:
            corpus = json.load(f)

        b_stars      = [s for s in corpus if s["hc_anchor"]["mk_letter"] == "B"]
        ab_corrected = [s for s in corpus if s["hc_anchor"].get("ab_boundary_logg_corrected")]
        binary_stars = [s for s in corpus if s["logical_flags"]["is_binary_candidate"]]

        print("=== SYSTEM PROMPT (first 800 chars) ===")
        print(build_system_prompt()[:800])

        print("\n=== USER PROMPT — B star (subtype few-shot) ===")
        if b_stars:
            print(build_user_prompt(b_stars[0]))

        print("\n=== USER PROMPT — A/B corrected star ===")
        if ab_corrected:
            print(build_user_prompt(ab_corrected[0]))
        else:
            print("No A/B corrected stars found")

        print("\n=== USER PROMPT — binary candidate ===")
        if binary_stars:
            print(build_user_prompt(binary_stars[0]))

        print("\n=== USER PROMPT — with mock RAG context ===")
        mock_rag = "[Chunk 1 | subtype_calibration_guide.md — Class F | score=0.821]\n## Class F..."
        if b_stars:
            print(build_user_prompt(b_stars[0], rag_context=mock_rag))

    except FileNotFoundError:
        print(f"Corpus not found at {corpus_path}")
        print("Run: python3 system_prompt_v7.py Data/stellar_corpus_v7.json")
