# STELLAR Pipeline — Subtype Calibration Guide

## Overview subtype calibration teff mapping accuracy per_class sub_type_range

This document provides Teff-to-subtype mapping and known calibration issues for each MK letter class in the STELLAR corpus. Subtype assignment is the primary task of the SC module within a letter class already fixed by the HC anchor. Accuracy varies significantly by class; this guide documents where the model performs well and where it should apply additional caution.

## General Principle subtype range uncertainty gspphot teff offset calibration failure_modes

Subtype is expressed as a range (e.g., `"3-5"`) rather than a single integer to reflect the ~100–200 K uncertainty inherent in Gaia DR3 GSP-Phot Teff estimates. Ranges should span no more than 3 subtype units. Avoid collapsing to `"0-1"` or `"8-9"` — these are known failure modes from V1–V4.

## Class B spectral_type:B teff:10000 teff:15000 teff:20000 teff:30000 subtype B0 B1 B2 B3 B5 B7 B8 B9 logg:giant logg:dwarf hot_star early_type

V6 subtype accuracy: **90.8%** — the strongest-performing class.

The physical anchors for B subtype are clear: Teff scales monotonically and the early subtypes (B0–B3) correspond to temperatures well above 20 000 K where AstroSage-8B discriminates reliably.

| Subtype | Approximate Teff (K) | Notes |
|---|---|---|
| B0 | 30 000 – 35 000 | Boundary with O; rare in corpus |
| B1–B3 | 20 000 – 30 000 | Best-calibrated range |
| B5–B7 | 15 000 – 20 000 | Moderate confidence |
| B8–B9 | 10 000 – 15 000 | Boundary with A; watch logg |

When assigning B subtype, `logg` is a secondary discriminant: B giants (logg ~3.0–3.5) at Teff ~12 000 K are B5–B7 III, not B9 V. Do not ignore luminosity when assigning late-B subtypes.

## Class A spectral_type:A teff:7500 teff:8500 teff:9000 teff:9999 subtype A0 A1 A2 A3 A5 A7 A9 logg:dwarf logg:subgiant main_sequence am_star chemically_peculiar ab_boundary_logg_corrected

V6 subtype accuracy: **26.7%** — moderate performance.

The A band is narrow in Teff and the corpus is relatively sparse (n=79 after logg correction). Stars near the A/B boundary (Teff 9 700–10 100 K, logg ≥ 3.8) arrive as A due to the V7 logg rule — assign A0–A2 for these.

| Subtype | Approximate Teff (K) |
|---|---|
| A0–A2 | 9 200 – 9 999 |
| A3–A5 | 8 500 – 9 200 |
| A7–A9 | 7 500 – 8 500 |

Chemically peculiar A stars (Am, Ap) are common in this temperature range. If `is_binary_candidate = True` combined with A-type Teff, consider Am classification — these are often tidally synchronized binaries.

## Class F spectral_type:F teff:6000 teff:6300 teff:6700 teff:7000 teff:7499 subtype F0 F2 F3 F5 F6 F7 F8 F9 logg:dwarf logg:subgiant solar_analog thin_disk subtype_collapse poor_discrimination

V6 subtype accuracy: **0.0%** — total collapse. This is the most critical failure in STELLAR V6.

The F band has the poorest subtype discrimination in the corpus. The model assigns F subtypes uniformly without correlation to Teff. To counter this collapse, use the following Teff anchors explicitly:

| Subtype | Approximate Teff (K) | Key discriminant |
|---|---|---|
| F0–F2 | 7 000 – 7 499 | Near A boundary; often Am-like |
| F3–F5 | 6 700 – 7 000 | Solar-analog precursors |
| F6–F8 | 6 300 – 6 700 | Common in Thin Disk |
| F9 | 6 000 – 6 300 | Near G boundary; watch logg |

Do not default to F5 as a generic midpoint. If Teff is available in the contract, map it directly using the table above. Express uncertainty through the confidence score, not through a midpoint subtype.

## Class G spectral_type:G teff:5200 teff:5300 teff:5500 teff:5778 teff:5800 teff:5999 subtype G0 G1 G2 G3 G5 G6 G8 G9 logg:dwarf logg:subgiant solar_analog sun solar_type thin_disk metal_rich

V6 subtype accuracy: **16.1%** — weak. G subtypes are highly degenerate in Gaia DR3 photometric parameters.

| Subtype | Approximate Teff (K) |
|---|---|
| G0–G2 | 5 800 – 5 999 |
| G3–G5 | 5 500 – 5 800 |
| G6–G8 | 5 300 – 5 500 |
| G9 | 5 200 – 5 300 |

Solar-type stars (Teff ~5 778 K) fall at G2 V. The Sun is the canonical reference: if the star has Teff ~5 750–5 800 K and logg ~4.4, the assignment G2 V is well-justified and high confidence is appropriate.

## Class K spectral_type:K teff:3700 teff:4100 teff:4500 teff:4900 teff:5199 subtype K0 K1 K2 K3 K4 K5 K7 logg:giant logg:subgiant logg:dwarf giant_branch red_giant thick_disk metal_poor subsolar

V6 subtype accuracy: **26.4%** — moderate. K spans a wide Teff range and intersects heavily with the giant branch.

| Subtype | Approximate Teff (K) |
|---|---|
| K0–K1 | 4 900 – 5 199 |
| K2–K3 | 4 500 – 4 900 |
| K4–K5 | 4 100 – 4 500 |
| K7 | 3 700 – 4 100 |

K giants (luminosity class III) with Teff ~4 200–4 800 K are common in this corpus. Do not assign K2 V for a star with logg ~2.5 — match luminosity class to logg first, then assign subtype.

## Class M spectral_type:M teff:2800 teff:3200 teff:3500 teff:3700 subtype M0 M1 M2 M3 M4 M5 M6 M7 M8 logg:giant logg:dwarf red_dwarf red_giant has_emission chromospheric_activity

V6 subtype accuracy: **74.1%** — good. The narrower Teff range per subtype unit makes M more discriminable.

| Subtype | Approximate Teff (K) |
|---|---|
| M0–M1 | 3 500 – 3 700 |
| M2–M3 | 3 200 – 3 500 |
| M4–M5 | 2 800 – 3 200 |
| M6–M8 | < 2 800 |

M giants (M III) at Teff ~3 500–3 800 K may overlap with late-K. Use `logg` as the primary luminosity discriminant. M dwarfs with H-alpha emission (`has_emission = True`) should note chromospheric activity explicitly in `notable_features`.

## Confidence Guidelines spectral_type_confidence per_class B A F G K M expected_range

| Class | Expected confidence range | Notes |
|---|---|---|
| B | 0.80 – 0.95 | High — strong physical anchors |
| M | 0.70 – 0.90 | Good — narrow Teff per subtype |
| K | 0.55 – 0.80 | Moderate — giant/dwarf ambiguity |
| A | 0.55 – 0.75 | Moderate — sparse corpus |
| G | 0.50 – 0.75 | Weak discrimination |
| F | 0.45 – 0.65 | Lowest — document uncertainty explicitly |
| O | 0.30 – 0.50 | Effectively inactive in corpus |
