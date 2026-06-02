# STELLAR Pipeline — Quality Flags Interpretation

## Overview quality_flags hc_agents boolean is_binary_candidate has_emission fit_diverged is_giant is_high_velocity confidence notable_features

The HC layer emits a set of boolean flags in the contract that encode data quality, physical peculiarities, and astrometric reliability. The SC module must use these flags to modulate confidence scores and populate the `notable_features` field of `stellar_description`.

## is_reliable_parallax parallax_error astrometry abs_mag v_tan unreliable distance

**Source:** AstrometryAgent — `parallax / parallax_error > 5`

- When `True`: `abs_mag` and `v_tan` are physically valid. Use them for luminosity and population cross-validation.
- When `False`: do not use `abs_mag` or `v_tan` in reasoning.

## is_giant logg:giant abs_mag luminosity_class:III luminosity_class:II spectral_type:K spectral_type:M evolved red_giant giant_branch

**Source:** AstrometryAgent — `abs_mag < 3.0` AND `teff_k < 7000` AND `is_reliable_parallax = True`

Strong prior toward luminosity class III or II. If `is_giant = True` but `logg_gspphot > 4.0`, contradiction likely indicates a binary — lower `luminosity_confidence`.

## is_high_velocity high_velocity_star v_tan population:Halo kinematic halo_membership fast_moving

**Source:** AstrometryAgent — `v_tan > 200 km/s` AND `is_reliable_parallax = True`

Population is already set to Halo. Set `population_confidence ≥ 0.85`. Always include `v_tan` in `population_context` and `notable_features`.

## is_binary_candidate binary ruwe is_astrometric_binary is_rv_variable is_confirmed_nss spectral_type_confidence penalty contamination unresolved_companion

**Source:** BinaryDetectorAgent — composite from: `ruwe > 1.4`, RV variability, or Gaia NSS solution.

**V7 rule:** when `is_binary_candidate = True`, reduce `spectral_type_confidence` by at least 0.10. If `is_confirmed_nss = True`, penalty is at least 0.15. Always mention in `notable_features`.

## has_emission halpha ew_halpha emission_line spectral_type:M spectral_type:B spectral_type:A be_star chromospheric_activity active_dwarf

**Source:** LineAgent — negative EW from `ew_espels_halpha` (quality flag = 0).

- M dwarfs: chromospheric activity, flare star.
- B/A stars: Be star or Herbig Ae/Be.
- K/G giants: symbiotic binary or active giant.

Always mention with EW value in `notable_features`.

## continuum_is_stable continuum_unstable snr spectral_quality teff uncertainty subtype_range widen

**Source:** ContinuumAgent — std of normalized flux within continuum mask < 0.05.

When `False`: widen the subtype range by 1–2 units. Mention only when it materially affects confidence.

## fit_diverged continuum_quality_flag spectral_type_confidence low_confidence unreliable teff uncertain sigma_clipping convergence_failure

**Source:** ContinuumAgent — non-physical fit values OR fewer than 10 points survived sigma-clipping.

Most severe flag: reduce `spectral_type_confidence` to ≤ 0.50. Use widest plausible subtype range. Always mention in `notable_features`.

## Flag Combination Patterns is_binary_candidate fit_diverged is_high_velocity is_giant has_emission spectral_type:M spectral_type:B composite_spectrum halo_giant symbiotic active_dwarf be_star

| Pattern | Interpretation |
|---|---|
| `is_binary_candidate` + `fit_diverged` | Composite spectrum — very low confidence |
| `is_high_velocity` + `is_giant` | Halo giant — old metal-poor population |
| `has_emission` + class M | Active M dwarf — chromospheric activity |
| `has_emission` + class B | Be star candidate — circumstellar disk |
| `fit_diverged` + `is_giant` | Evolved star with complex spectrum |
| `is_binary_candidate` + `has_emission` | Possible symbiotic or interacting binary |

## When notable_features Says None Identified notable_features none_identified all_flags_false clean_star no_flags

Only when ALL of these are False: `is_binary_candidate`, `is_high_velocity`, `has_emission`, `fit_diverged`. If any flag is True, `notable_features` must contain a specific statement.
