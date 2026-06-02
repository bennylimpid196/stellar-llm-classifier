# STELLAR Pipeline — MK Classification Rules (HC Layer)

## Overview mk_rules pipeline hc_anchor spectral_type teff thresholds population priority chain

This document describes the internal deterministic rules of the Hard Computing (HC) layer in STELLAR v7. These rules are pipeline-specific and may differ from standard MK atlas conventions. When the SC module receives an `hc_anchor` field, the spectral letter and population group have already been fixed by these rules and must not be overridden. The SC task is to assign subtype, luminosity class, confidence, and stellar description.

## Spectral Letter Assignment spectral_type:O spectral_type:B spectral_type:A spectral_type:F spectral_type:G spectral_type:K spectral_type:M teff thresholds letter assignment

The HC computes the MK letter from `teff_gspphot` using strict inequality thresholds. No probability or uncertainty is applied at this stage.

| Teff range (K) | Assigned letter |
|---|---|
| ≥ 30 000 | O |
| 10 000 – 29 999 | B |
| 7 500 – 9 999 | A |
| 6 000 – 7 499 | F |
| 5 200 – 5 999 | G |
| 3 700 – 5 199 | K |
| < 3 700 | M |

The B/A boundary is set at exactly 10 000 K with no guard band. This is intentionally strict. Stars with Teff between 9 700–10 100 K are assigned B if Teff ≥ 10 000 K and A otherwise, regardless of luminosity class. See the A/B boundary section below for the logg correction applied in V7.

## A/B Boundary Rule logg Correction spectral_type:A spectral_type:B teff:9700 teff:10100 logg:3.8 ab_boundary_logg_corrected dwarf subgiant main_sequence

Stars in the Teff range 9 700–10 100 K receive an additional logg-based correction before the HC anchor is emitted:

- If `logg_gspphot ≥ 3.8` → letter is set to **A** (dwarf/subgiant regime)
- If `logg_gspphot < 3.8` → letter follows the raw Teff rule above

This rule was introduced in V7 after V6 analysis showed that the deterministic logg rule achieves 76.5% accuracy on boundary stars vs. 43.1% for the strict Teff threshold alone. The SC module should treat this corrected anchor as authoritative.

## Population Group Assignment Priority Chain population:Thin_Disk population:Thick_Disk population:Halo priority kinematic chemistry alpha_fe metallicity fe_h

Population is assigned by a strict priority chain. The first condition that matches wins; subsequent conditions are not evaluated.

### Priority 1 — Kinematic Override
If `is_high_velocity = True` AND `is_reliable_parallax = True` → **Halo**

`is_high_velocity` is set by the AstrometryAgent when `v_tan > 200 km/s`. This override has the highest priority because kinematics are a more direct indicator of Halo membership than photometric chemistry alone.

### Priority 2 — Alpha-enhancement Rule
If `alpha_fe ≥ 0.2` AND `-1.0 ≤ chemistry < -0.2` → **Thick Disk**

`alpha_fe` corresponds to `alphafe_gspspec` in Gaia DR3. Coverage is approximately 53% of the corpus — when the field is null, this priority is skipped entirely and evaluation falls through to Priority 3.

### Priority 3 — Chemistry Thresholds
Applied when neither Priority 1 nor Priority 2 fired:

| Chemistry value | Assigned population |
|---|---|
| < −1.0 | Halo |
| −1.0 to −0.2 | Thick Disk |
| ≥ −0.2 | Thin Disk |

Chemistry is resolved as follows: `fem_gspspec` (spectroscopic [Fe/M]) takes precedence over `mh_gspphot` (photometric [M/H]) when both are available. If `fe_h = 0.0` exactly, it is treated as an artifact (null) and `mh_gspphot` is used instead.

## Corpus Distribution spectral_type:B spectral_type:A spectral_type:F spectral_type:G spectral_type:K spectral_type:M population:Thin_Disk population:Thick_Disk population:Halo n_stars 498

| Letter | N | Population | N |
|---|---|---|---|
| O | 0* | Thin Disk | 265 |
| B | 99 | Thick Disk | 146 |
| A | 79 | Halo | 87 |
| F | 93 | | |
| G | 80 | | |
| K | 90 | | |
| M | 57 | | |

## What the SC Module Must Not Do sc_rules hc_anchor locked spectral_type population_group override prohibited

The SC module must not recompute or override the spectral letter or population group when an `hc_anchor` is present. These fields are locked. The SC may express uncertainty through lower confidence scores, but the output JSON fields `spectral_type` and `population_group` must exactly match the values in `hc_anchor`.
