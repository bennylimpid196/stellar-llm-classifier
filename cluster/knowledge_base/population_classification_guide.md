# STELLAR Pipeline — Population Classification Guide

## Overview population:Thin_Disk population:Thick_Disk population:Halo hc_anchor locked population_confidence population_context justification

Population group (Thin Disk, Thick Disk, Halo) is assigned deterministically by the HC layer and locked in the `hc_anchor`. The SC must justify the assignment in `population_context` and set `population_confidence` based on evidence strength.

## Population Groups Physical Meaning population:Thin_Disk metallicity:thin_disk_solar solar young population:Thick_Disk metallicity:thick_disk_subsolar alpha_enhanced old kinematically_hot population:Halo metallicity:halo_metalpoor metal_poor very_old high_velocity_star v_tan retrograde

**Thin Disk:** Young-to-intermediate age. Roughly solar metallicity ([M/H] ≥ −0.2). Low alpha-enhancement. Low space velocities.

**Thick Disk:** Old, kinematically hot. Subsolar metallicity (−1.0 ≤ [M/H] < −0.2) with alpha-enhancement ([α/Fe] ≥ 0.2). Formed during early rapid star formation epoch.

**Halo:** Very old, metal-poor. Either [M/H] < −1.0 (chemistry) or v_tan > 200 km/s (kinematic override takes priority).

## HC Decision Paths SC Confidence is_high_velocity v_tan population:Halo kinematic_override population_confidence alpha_enhanced alpha_fe population:Thick_Disk fe_h metallicity spectroscopic photometric

### Path 1: Kinematic override (v_tan > 200 km/s)
`population_confidence ≥ 0.85`. Strongest evidence. Always include `v_tan` in `population_context`.

### Path 2: Alpha-enhancement ([α/Fe] ≥ 0.2 AND −1.0 ≤ [M/H] < −0.2)
`population_confidence`: 0.70–0.85. Include both alpha-fe and metallicity values.

### Path 3a: Chemistry Halo ([M/H] < −1.0)
`population_confidence`: 0.75–0.85 (spectroscopic), 0.60–0.75 (photometric).

### Path 3b: Chemistry Thick Disk (−1.0 ≤ [M/H] < −0.2, no alpha-fe)
`population_confidence`: 0.55–0.70. Weakest path — acknowledge limitation.

### Path 3c: Chemistry Thin Disk ([M/H] ≥ −0.2)
`population_confidence`: 0.65–0.80.

## Alpha Coverage Problem alpha_fe alphafe_gspspec null missing coverage 53percent thick_disk fallthrough priority2 skipped

`alphafe_gspspec` is null for ~47% of stars. When null, Priority 2 is skipped entirely. Do not infer alpha-fe from its absence.

## Chemistry Field Priority fem_gspspec mh_gspphot fe_h spectroscopic photometric metallicity precedence chemistry_source

1. `fem_gspspec` (spectroscopic [Fe/M]) — highest reliability, ~44% coverage
2. `mh_gspphot` (photometric [M/H]) — lower reliability, broader coverage
3. `fe_h = 0.0` — treated as null artifact; falls back to photometric

## Corpus Population Distribution population:Thin_Disk population:Thick_Disk population:Halo n_stars 265 146 87

| Population | N (498 stars) | Fraction |
|---|---|---|
| Thin Disk | 265 | 53.2% |
| Thick Disk | 146 | 29.3% |
| Halo | 87 | 17.5% |

## Population Confidence Summary population_confidence kinematic alpha_fe spectroscopic photometric thin_disk thick_disk halo confidence_range

| Assignment path | Recommended confidence range |
|---|---|
| Kinematic override (v_tan > 200) | 0.85 – 0.95 |
| Alpha-fe rule | 0.70 – 0.85 |
| Spectroscopic [Fe/H] → Halo | 0.75 – 0.85 |
| Spectroscopic [Fe/H] → Thick Disk | 0.65 – 0.75 |
| Photometric [M/H] → Halo | 0.60 – 0.75 |
| Photometric [M/H] only → Thick Disk | 0.55 – 0.70 |
| [M/H] ≥ −0.2 → Thin Disk | 0.65 – 0.80 |
