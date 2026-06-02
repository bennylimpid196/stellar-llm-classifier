# STELLAR Pipeline — Luminosity Class Guide

## Overview luminosity_class logg logg:giant logg:dwarf logg:subgiant logg:supergiant assignment sc_responsibility

Luminosity class (I through V) is assigned by the SC module based primarily on `logg_gspphot`, which bypasses all HC logic and is passed directly to the SC layer. This is by design: `logg` is the most direct photometric proxy for surface gravity, which physically determines the luminosity class. The HC layer does not compute luminosity class — it is exclusively an SC responsibility.

## logg to Luminosity Class Mapping logg:supergiant logg:giant logg:subgiant logg:dwarf luminosity_class:I luminosity_class:II luminosity_class:III luminosity_class:IV luminosity_class:V gspphot systematic_offset

The following ranges reflect Gaia DR3 GSP-Phot values for the STELLAR corpus.

| Luminosity class | Common name | logg range (dex) | Notes |
|---|---|---|---|
| I | Supergiant | < 1.5 | Very rare in Gaia DR3 magnitude-limited sample |
| II | Bright giant | 1.5 – 2.5 | Uncommon; verify with abs_mag if parallax reliable |
| III | Giant | 2.5 – 3.5 | Common for K and M types in this corpus |
| IV | Subgiant | 3.5 – 4.0 | Transition zone; significant overlap with III and V |
| V | Main sequence dwarf | > 4.0 | Most common in the corpus |

The subgiant zone (logg 3.5–4.0) is the most ambiguous. When logg falls in this range, use Teff and absolute magnitude (`abs_mag`) as secondary discriminants.

## logg Bypasses the HC logg:giant spectral_type:K spectral_type:M gspphot systematic cool_giant overestimate logg:dwarf spectral_type:B spectral_type:A hot_star nan_logg

Because `logg_gspphot` is passed directly to the SC without HC transformation, the SC receives the raw GSP-Phot value. Known systematic issues in Gaia DR3:

**Cool giants (K/M III):** GSP-Phot tends to overestimate logg for cool evolved stars. A K giant with true logg ~2.0 may arrive with `logg_gspphot ~2.8`. Treat logg for K/M giants as a lower bound — err toward III over II for these types.

**Hot stars (B/A):** GSP-Phot logg for OB stars is less reliable. Cross-check against `abs_mag` and the Teff-luminosity relation.

**NaN logg:** If `logg_gspphot` is null, infer from `abs_mag` or use the most probable class. For M-type stars with null logg, M III is the more probable default due to Malmquist bias.

## Luminosity Class and A/B Boundary spectral_type:A ab_boundary_logg_corrected teff:9700 teff:10100 logg:3.8 luminosity_class:IV luminosity_class:V dwarf subgiant

Stars with letter A and Teff in the 9 700–10 100 K range always have logg ≥ 3.8, placing them in the IV–V zone. Assign luminosity class IV or V for these stars — never I, II, or III.

## Cross-validation with Absolute Magnitude abs_mag is_reliable_parallax luminosity_class:I luminosity_class:II luminosity_class:III luminosity_class:IV luminosity_class:V magnitude binary contamination

| Luminosity class | Approximate M_G range |
|---|---|
| I (supergiant) | < −3 |
| II (bright giant) | −3 to −1 |
| III (giant) | −1 to +3 |
| IV (subgiant) | +3 to +5 |
| V (dwarf) | > +3 (Teff-dependent) |

If `abs_mag` and `logg` disagree by more than one class, note the discrepancy and lower `luminosity_confidence` — often indicates binary contamination.

## Luminosity Confidence Guidelines luminosity_confidence logg is_reliable_parallax nan_logg data_quality confidence_range

- `logg` available AND `is_reliable_parallax = True` → confidence 0.75–0.90
- `logg` available, parallax not reliable → confidence 0.60–0.75
- `logg` is null, `is_reliable_parallax = True` → confidence 0.50–0.65
- Both null → confidence 0.35–0.50; note data limitation in `notable_features`
