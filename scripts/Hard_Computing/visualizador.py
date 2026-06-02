# -*- coding: utf-8 -*-
"""
Gaia DR3 — Visualization & Diagnostics
========================================
Standalone script. Reads data produced by gaia_dataset_builder.py.
Never re-downloads from Gaia.

Input structure expected (from gaia_dataset_builder.py):
    /home/cesar/Documentos/Tesis-cimat/Estancia/data/raw/
    └── DB-{N}-{YYYY-MM-DD}/
        ├── catalog.csv
        ├── spectra_bprp.npy
        ├── spectra_bprp_ids.npy
        ├── sampling_bprp.npy
        ├── run_manifest.json
        └── rvs/
            └── {source_id}.npz

Output structure (created by this script):
    /home/cesar/Documentos/Tesis-cimat/Estancia/data/raw/
    └── VIZ-{N}-{YYYY-MM-DD}/
        ├── diagnostico_catalogo.png
        └── espectros_{source_id}.png

Usage:
    # Visualize most recent DB run
    python visualizacion_diagnostico.py

    # Visualize a specific DB run by number
    python visualizacion_diagnostico.py --db 3

    # Visualize a specific DB run by full path
    python visualizacion_diagnostico.py --db /path/to/DB-3-2026-04-28
"""

import argparse
import json
import logging
import re
import warnings
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['text.usetex']        = False
matplotlib.rcParams['font.family']        = 'monospace'
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore", module='astropy.io.votable')
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

RAW_DATA_ROOT = Path('/home/cesar/Documentos/Tesis-cimat/Estancia/data/raw')

# =============================================================================
# PALETTE AND STYLE
# =============================================================================

DARK_BG    = '#0d0f14'
PANEL_BG   = '#13161e'
GRID_COLOR = '#1f2535'
TEXT_COLOR = '#c8d0e0'
DIM_TEXT   = '#5a6480'
ACCENT1    = '#4fc3f7'   # cyan blue  — BP/RP
ACCENT2    = '#ef5350'   # red        — Ca II / alerts
ACCENT3    = '#66bb6a'   # green      — H-alpha / OK
ACCENT4    = '#ffa726'   # orange     — highlights
ACCENT5    = '#ab47bc'   # purple     — NSS / binaries
ACCENT6    = '#26c6da'   # teal       — category bars


def _apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  DARK_BG,
        'axes.facecolor':    PANEL_BG,
        'axes.edgecolor':    GRID_COLOR,
        'axes.labelcolor':   TEXT_COLOR,
        'axes.titlecolor':   TEXT_COLOR,
        'xtick.color':       DIM_TEXT,
        'ytick.color':       DIM_TEXT,
        'grid.color':        GRID_COLOR,
        'grid.linewidth':    0.5,
        'text.color':        TEXT_COLOR,
        'legend.facecolor':  PANEL_BG,
        'legend.edgecolor':  GRID_COLOR,
        'legend.labelcolor': TEXT_COLOR,
        'font.size':         9,
    })


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Gaia DR3 Visualization & Diagnostics'
    )
    parser.add_argument(
        '--db', type=str, default=None,
        help=(
            'DB run to visualize. '
            'Accepts a run number (e.g. 3) or a full path. '
            'Defaults to the most recent DB-N folder in the raw data root.'
        )
    )
    return parser.parse_args()


# =============================================================================
# FOLDER RESOLUTION
# =============================================================================

def resolve_db_folder(db_arg: Optional[str]) -> Path:
    """
    Resolves which DB-N-DATE folder to read from.

    Rules:
        - db_arg is None      → pick the highest-numbered DB-N folder.
        - db_arg is a digit   → pick DB-{N}-* matching that number.
        - db_arg is a path    → use directly.
    """
    if db_arg is None:
        return _find_latest_db_folder()

    if db_arg.isdigit():
        return _find_db_folder_by_number(int(db_arg))

    path = Path(db_arg)
    if not path.exists():
        raise FileNotFoundError(f"Specified DB path does not exist: {path}")
    return path


def _find_latest_db_folder() -> Path:
    pattern = re.compile(r'^DB-(\d+)-\d{4}-\d{2}-\d{2}$')
    candidates = [
        (int(pattern.match(d.name).group(1)), d)
        for d in RAW_DATA_ROOT.iterdir()
        if d.is_dir() and pattern.match(d.name)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No DB-N-DATE folders found in {RAW_DATA_ROOT}. "
            "Run gaia_dataset_builder.py first."
        )
    _, folder = max(candidates, key=lambda x: x[0])
    return folder


def _find_db_folder_by_number(n: int) -> Path:
    pattern = re.compile(rf'^DB-{n}-\d{{4}}-\d{{2}}-\d{{2}}$')
    matches = [
        d for d in RAW_DATA_ROOT.iterdir()
        if d.is_dir() and pattern.match(d.name)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No DB folder with number {n} found in {RAW_DATA_ROOT}."
        )
    return matches[0]


def resolve_viz_output_folder() -> Path:
    """Creates and returns the next incremental VIZ-N-DATE output folder."""
    RAW_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r'^VIZ-(\d+)-\d{4}-\d{2}-\d{2}$')
    numbers = [
        int(pattern.match(d.name).group(1))
        for d in RAW_DATA_ROOT.iterdir()
        if d.is_dir() and pattern.match(d.name)
    ]
    next_n = (max(numbers) + 1) if numbers else 1

    folder_name = f'VIZ-{next_n:02d}-{date.today().isoformat()}'
    folder = RAW_DATA_ROOT / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# =============================================================================
# DATA LOADING
# =============================================================================

def load_db(db_folder: Path) -> Tuple[pd.DataFrame, dict]:
    """
    Loads catalog.csv and run_manifest.json from a DB folder.

    Returns:
        df       : DataFrame with catalog metadata.
        manifest : Dict with run metadata.
    """
    catalog_path  = db_folder / 'catalog.csv'
    manifest_path = db_folder / 'run_manifest.json'

    if not catalog_path.exists():
        raise FileNotFoundError(f"catalog.csv not found in {db_folder}")

    df = pd.read_csv(catalog_path)
    df['source_id'] = df['source_id'].astype('int64')

    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        logger.warning("run_manifest.json not found. Some metadata will be unavailable.")

    return df, manifest


def load_bprp_spectrum(db_folder: Path, source_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads BP/RP flux and wavelength grid for a single star from the .npy arrays.

    Returns:
        (wavelength_nm, flux) or (None, None) if unavailable.
    """
    ids_path     = db_folder / 'spectra_bprp_ids.npy'
    matrix_path  = db_folder / 'spectra_bprp.npy'
    sampling_path = db_folder / 'sampling_bprp.npy'

    if not all(p.exists() for p in [ids_path, matrix_path, sampling_path]):
        logger.warning("BP/RP numpy arrays not found in DB folder.")
        return None, None

    ids      = np.load(ids_path)
    sampling = np.load(sampling_path)

    idx = np.where(ids == source_id)[0]
    if len(idx) == 0:
        logger.info(f"source_id {source_id} not found in BP/RP matrix.")
        return None, None

    # Load only the required row using memory mapping (avoids loading full matrix)
    matrix = np.load(matrix_path, mmap_mode='r')
    flux = matrix[idx[0]].copy()

    if np.all(np.isnan(flux)):
        return None, None

    return sampling, flux


def load_rvs_spectrum(db_folder: Path, source_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads RVS wavelength and flux from the per-star .npz file.

    Returns:
        (wavelength_nm, flux) or (None, None) if unavailable.
    """
    npz_path = db_folder / 'rvs' / f'{source_id}.npz'
    if not npz_path.exists():
        return None, None

    data = np.load(npz_path)
    return data['wavelength_nm'], data['flux']


# =============================================================================
# CATEGORY CLASSIFICATION (mirrors gaia_dataset_builder.py)
# =============================================================================

# Display labels for each internal category key
CATEGORY_LABELS = {
    'kg_enanas':   'K/G Enanas SP',
    'af_sp':       'A/F SP',
    'kg_gigantes': 'K/G Gigantes',
    'b_calientes': 'B Calientes',
    'm_frias':     'M Frías',
    'halo':        'Halo / Metal-poor',
}

CATEGORY_COLORS = {
    'kg_enanas':   ACCENT1,
    'af_sp':       ACCENT3,
    'kg_gigantes': ACCENT4,
    'b_calientes': ACCENT2,
    'm_frias':     ACCENT5,
    'halo':        ACCENT6,
}


def classify_stars(df: pd.DataFrame) -> pd.Series:
    """Assigns a category label to each row. Mirrors builder logic exactly."""
    t = df['teff_gspphot']
    g = df['logg_gspphot']
    m = df['mh_gspphot']

    cats = pd.Series('otro', index=df.index)
    cats[t >= 10000]                               = 'b_calientes'
    cats[(t >= 6500) & (t < 10000) & (g >= 3.5)]  = 'af_sp'
    cats[(t >= 4000) & (t < 6500)  & (g >= 3.5)]  = 'kg_enanas'
    cats[(t >= 4000) & (t < 6500)  & (g < 3.5)]   = 'kg_gigantes'
    cats[t < 4000]                                  = 'm_frias'
    cats[m < -1.0]                                  = 'halo'
    return cats


# =============================================================================
# PANEL 1 — CATALOG DIAGNOSTICS (8 subpanels)
# =============================================================================

def panel_diagnostico(df: pd.DataFrame, manifest: dict, carpeta: Path):
    """
    Full catalog diagnostic panel with 8 subpanels:
      Row 1: Teff distribution | HR diagram | Metallicity [M/H] | RUWE
      Row 2: EW H-alpha        | Completeness by field | Category distribution | Spectrum availability
    """
    _apply_dark_style()

    total = len(df)
    run_date = manifest.get('run_date', 'unknown date')
    db_folder_name = manifest.get('output_folder', '')

    # Derived quantities
    df = df.copy()
    df['parallax_snr'] = df['parallax'] / df['parallax_error']
    mask_reliable = (df['parallax'] > 0) & (df['parallax_snr'] > 5)

    ebp = df['ebpminrp_gspphot'].fillna(0.0)
    df['abs_mag_g'] = (
        df['phot_g_mean_mag']
        + 5
        + 5 * np.log10(df['parallax'].clip(lower=1e-6) / 1000)
        - 2.74 * ebp
    )
    df_hr = df[mask_reliable].copy()

    # Category assignment
    df['_category'] = classify_stars(df)

    fig = plt.figure(figsize=(22, 12), facecolor=DARK_BG)
    fig.suptitle(
        f'DIAGNÓSTICO DEL CATÁLOGO  —  Gaia DR3  |  {total} estrellas  |  {run_date}',
        fontsize=13, color=TEXT_COLOR, y=0.99,
        fontweight='bold', fontfamily='monospace'
    )

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.48, wspace=0.38,
        left=0.06, right=0.98, top=0.94, bottom=0.08
    )

    # ---- 1. Teff distribution ----
    ax1 = fig.add_subplot(gs[0, 0])
    teff = df['teff_gspphot'].dropna()
    ax1.hist(teff, bins=40, color=ACCENT1, alpha=0.85,
             edgecolor=DARK_BG, linewidth=0.4)
    ax1.set_title('Distribución de T_eff', fontweight='bold')
    ax1.set_xlabel('T_eff  [K]')
    ax1.set_ylabel('N estrellas')
    ax1.grid(True, alpha=0.4)
    for t_boundary, label, color in [
        (3500, 'M', ACCENT2), (5200, 'K', ACCENT4),
        (6000, 'G', ACCENT3), (7500, 'A', ACCENT1), (10000, 'B', ACCENT5)
    ]:
        if teff.min() <= t_boundary <= teff.max():
            ax1.axvline(t_boundary, color=color, linewidth=0.8,
                        linestyle='--', alpha=0.7)
            ax1.text(t_boundary + 100, ax1.get_ylim()[1] * 0.88,
                     label, color=color, fontsize=8)

    # ---- 2. HR diagram ----
    ax2 = fig.add_subplot(gs[0, 1])
    giant_mask  = (df_hr['abs_mag_g'] < 3.0) & (df_hr['teff_gspphot'] < 7000)
    normal_mask = ~giant_mask

    ax2.scatter(
        df_hr.loc[normal_mask, 'teff_gspphot'],
        df_hr.loc[normal_mask, 'abs_mag_g'],
        s=4, alpha=0.6, color=ACCENT1,
        label=f'Enanas ({normal_mask.sum()})'
    )
    ax2.scatter(
        df_hr.loc[giant_mask, 'teff_gspphot'],
        df_hr.loc[giant_mask, 'abs_mag_g'],
        s=12, alpha=0.85, color=ACCENT4, marker='*',
        label=f'Gigantes ({giant_mask.sum()})'
    )
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.set_title('Diagrama HR  (paralaje confiable)', fontweight='bold')
    ax2.set_xlabel('T_eff  [K]')
    ax2.set_ylabel('M_G  [mag]')
    ax2.legend(fontsize=8, markerscale=2)
    ax2.grid(True, alpha=0.4)
    ax2.axhline(3.0, color=DIM_TEXT, linewidth=0.6, linestyle=':')
    ax2.text(
        ax2.get_xlim()[0] * 0.99, 3.2,
        'is_giant threshold', color=DIM_TEXT, fontsize=7, ha='right'
    )

    # ---- 3. Metallicity ----
    ax3 = fig.add_subplot(gs[0, 2])
    mh = df['mh_gspphot'].dropna()
    ax3.hist(mh, bins=40, color=ACCENT3, alpha=0.85,
             edgecolor=DARK_BG, linewidth=0.4)
    ax3.axvline(-1.0, color=ACCENT2, linewidth=1.0,
                linestyle='--', label='[M/H] = -1.0  (halo)')
    ax3.set_title('Metalicidad  [M/H]', fontweight='bold')
    ax3.set_xlabel('[M/H]  [dex]')
    ax3.set_ylabel('N estrellas')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.4)
    n_poor = (mh < -1.0).sum()
    ax3.text(0.97, 0.92, f'Halo: {n_poor}',
             transform=ax3.transAxes, ha='right', color=ACCENT2, fontsize=8)

    # ---- 4. RUWE ----
    ax4 = fig.add_subplot(gs[0, 3])
    ruwe = df['ruwe'].dropna()
    ax4.hist(ruwe.clip(upper=5.0), bins=50, color=ACCENT4, alpha=0.85,
             edgecolor=DARK_BG, linewidth=0.4)
    ax4.axvline(1.4, color=ACCENT2, linewidth=1.2,
                linestyle='--', label='RUWE = 1.4  (binaria candidata)')
    ax4.set_title('Distribución de RUWE', fontweight='bold')
    ax4.set_xlabel('RUWE  (recortado a 5.0)')
    ax4.set_ylabel('N estrellas')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.4)
    n_bin = (ruwe > 1.4).sum()
    ax4.text(0.97, 0.92, f'RUWE > 1.4: {n_bin}',
             transform=ax4.transAxes, ha='right', color=ACCENT2, fontsize=8)

    # ---- 5. EW H-alpha ----
    ax5 = fig.add_subplot(gs[1, 0])
    flag_col = 'ew_espels_halpha_flag'
    ew_col   = 'ew_espels_halpha'
    if flag_col in df.columns and ew_col in df.columns:
        ew_ok  = df[df[flag_col] == 0][ew_col].dropna()
        ew_abs = ew_ok[ew_ok >= 0]
        ew_emi = ew_ok[ew_ok < 0]
        ax5.hist(ew_abs, bins=35, color=ACCENT1, alpha=0.8,
                 edgecolor=DARK_BG, linewidth=0.4,
                 label=f'Absorción ({len(ew_abs)})')
        ax5.hist(ew_emi, bins=10, color=ACCENT2, alpha=0.85,
                 edgecolor=DARK_BG, linewidth=0.4,
                 label=f'Emisión ({len(ew_emi)})')
        ax5.axvline(0, color=TEXT_COLOR, linewidth=0.8, linestyle=':')
        ax5.text(0.03, 0.92, 'Convención DR3:\n+ = absorción\n− = emisión',
                 transform=ax5.transAxes, color=DIM_TEXT, fontsize=7.5)
    else:
        ax5.text(0.5, 0.5, 'EW H-alpha no disponible',
                 transform=ax5.transAxes, ha='center', va='center',
                 color=DIM_TEXT, fontsize=10)
    ax5.set_title('EW H-alpha  (flag = 0)', fontweight='bold')
    ax5.set_xlabel('EW_Halpha  [Å]')
    ax5.set_ylabel('N estrellas')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.4)

    # ---- 6. Completeness by field ----
    ax6 = fig.add_subplot(gs[1, 1])
    fields = {
        'teff_gspphot':          'T_eff',
        'logg_gspphot':          'log g',
        'mh_gspphot':            '[M/H]',
        'alphafe_gspspec':       '[alpha/Fe]',
        'fem_gspspec':           '[Fe/M]',
        'ebpminrp_gspphot':      'E(BP-RP)',
        'ew_espels_halpha':      'EW H-alpha',
        'radial_velocity_error': 'RV error',
        'nss_solution_type':     'NSS type',
    }
    labels    = list(fields.values())
    pct_avail = [(1 - df[c].isna().mean()) * 100
                 for c in fields if c in df.columns]
    labels    = [v for c, v in fields.items() if c in df.columns]
    bar_colors = [
        ACCENT3 if v >= 80 else ACCENT4 if v >= 40 else ACCENT2
        for v in pct_avail
    ]
    bars = ax6.barh(labels, pct_avail, color=bar_colors,
                    edgecolor=DARK_BG, linewidth=0.4, height=0.65)
    ax6.set_xlim(0, 115)
    ax6.set_title('Completitud por campo  [%]', fontweight='bold')
    ax6.set_xlabel('% valores disponibles')
    ax6.grid(True, axis='x', alpha=0.4)
    ax6.axvline(100, color=DIM_TEXT, linewidth=0.5, linestyle=':')
    for bar, val in zip(bars, pct_avail):
        ax6.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.0f}%', va='center', fontsize=8, color=TEXT_COLOR)

    # ---- 7. Category distribution vs. target ----
    ax7 = fig.add_subplot(gs[1, 2])
    category_counts = df['_category'].value_counts()

    # Retrieve targets from manifest if available
    manifest_targets = manifest.get('parameters', {})
    target_total     = manifest_targets.get('total_requested', total)

    # Recompute targets using the same proportions as the builder
    proportions = {
        'kg_enanas': 0.30, 'af_sp': 0.20, 'kg_gigantes': 0.16,
        'b_calientes': 0.12, 'm_frias': 0.12, 'halo': 0.10,
    }
    cat_keys   = list(CATEGORY_LABELS.keys())
    cat_labels = [CATEGORY_LABELS[k] for k in cat_keys]
    actuals    = [category_counts.get(k, 0) for k in cat_keys]
    targets_n  = [int(target_total * proportions[k]) for k in cat_keys]
    colors_bar = [CATEGORY_COLORS[k] for k in cat_keys]

    y_pos = np.arange(len(cat_keys))
    ax7.barh(y_pos - 0.18, targets_n, height=0.32,
             color=DIM_TEXT, alpha=0.5, label='Objetivo')
    ax7.barh(y_pos + 0.18, actuals,   height=0.32,
             color=colors_bar, alpha=0.9, label='Real')
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(cat_labels, fontsize=8)
    ax7.set_title('Distribución por Categoría', fontweight='bold')
    ax7.set_xlabel('N estrellas')
    ax7.legend(fontsize=8)
    ax7.grid(True, axis='x', alpha=0.4)
    for i, (a, t) in enumerate(zip(actuals, targets_n)):
        color = ACCENT3 if a >= int(t * 0.9) else ACCENT2
        ax7.text(a + 0.5, i + 0.18, str(a),
                 va='center', fontsize=7.5, color=color)

    # ---- 8. Spectrum availability ----
    ax8 = fig.add_subplot(gs[1, 3])

    has_bprp = df.get('has_bprp_spectrum', pd.Series(False, index=df.index))
    has_rvs  = df.get('has_rvs_spectrum',  pd.Series(False, index=df.index))

    n_both   = int((has_bprp & has_rvs).sum())
    n_bprp   = int((has_bprp & ~has_rvs).sum())
    n_rvs    = int((~has_bprp & has_rvs).sum())
    n_none   = int((~has_bprp & ~has_rvs).sum())

    categories_spec = ['BP/RP + RVS', 'Solo BP/RP', 'Solo RVS', 'Sin espectro']
    counts_spec     = [n_both, n_bprp, n_rvs, n_none]
    colors_spec     = [ACCENT3, ACCENT1, ACCENT4, ACCENT2]

    bars8 = ax8.bar(categories_spec, counts_spec,
                    color=colors_spec, edgecolor=DARK_BG,
                    linewidth=0.4, width=0.6)
    ax8.set_title('Disponibilidad de Espectros', fontweight='bold')
    ax8.set_ylabel('N estrellas')
    ax8.grid(True, axis='y', alpha=0.4)
    for bar, val in zip(bars8, counts_spec):
        pct = val / total * 100
        ax8.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val}\n({pct:.0f}%)',
            ha='center', va='bottom', fontsize=8, color=TEXT_COLOR
        )
    ax8.tick_params(axis='x', labelsize=7.5)

    ruta = carpeta / 'diagnostico_catalogo.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f"  Panel guardado: {ruta}")
    plt.close(fig)


# =============================================================================
# PANEL 2 — PARALLEL SPECTRA (BP/RP + RVS) FOR A SINGLE STAR
# =============================================================================

def panel_espectros_paralelo(
    source_id: int,
    fila_catalogo: pd.Series,
    wavelength_bprp: Optional[np.ndarray],
    flux_bprp: Optional[np.ndarray],
    wavelength_rvs: Optional[np.ndarray],
    flux_rvs: Optional[np.ndarray],
    carpeta: Path
):
    """
    Two-row panel for a single star:
      Row 1: Full BP/RP spectrum (336–1020 nm)
      Row 2: RVS zoom (845–872 nm) with Ca II triplet
    Includes a physical parameters sidebar.
    """
    _apply_dark_style()

    # Extract parameters from catalog row
    teff  = fila_catalogo.get('teff_gspphot', np.nan)
    logg  = fila_catalogo.get('logg_gspphot', np.nan)
    mh    = fila_catalogo.get('mh_gspphot', np.nan)
    ruwe  = fila_catalogo.get('ruwe', np.nan)
    plx   = fila_catalogo.get('parallax', np.nan)
    plx_e = fila_catalogo.get('parallax_error', np.nan)
    ew_ha = fila_catalogo.get('ew_espels_halpha', np.nan)
    nss   = fila_catalogo.get('nss_solution_type', None)
    has_bprp = bool(fila_catalogo.get('has_bprp_spectrum', False))
    has_rvs  = bool(fila_catalogo.get('has_rvs_spectrum', False))

    def _fmt(val, dec=2, suffix=''):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        return f'{val:.{dec}f}{suffix}'

    # Compute M_G if parallax is reliable
    abs_mag_str = 'N/A'
    abs_mag_val = np.nan
    if (not np.isnan(plx) and not np.isnan(plx_e)
            and plx > 0 and (plx / plx_e) > 5):
        ebp = fila_catalogo.get('ebpminrp_gspphot', 0.0)
        ebp = 0.0 if pd.isna(ebp) else float(ebp)
        abs_mag_val = (
            float(fila_catalogo.get('phot_g_mean_mag', np.nan))
            + 5 + 5 * np.log10(plx / 1000) - 2.74 * ebp
        )
        abs_mag_str = _fmt(abs_mag_val, 2, ' mag')

    fig = plt.figure(figsize=(16, 9), facecolor=DARK_BG)
    fig.suptitle(
        f'ESPECTROS GAIA DR3  —  source_id: {source_id}',
        fontsize=13, color=TEXT_COLOR, y=0.99,
        fontweight='bold', fontfamily='monospace'
    )

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        hspace=0.45,
        left=0.07, right=0.78,
        top=0.93, bottom=0.08
    )

    # ---- Row 1: Full BP/RP spectrum ----
    ax_bp = fig.add_subplot(gs[0])

    if wavelength_bprp is not None and flux_bprp is not None:
        ax_bp.plot(wavelength_bprp, flux_bprp,
                   color=ACCENT1, linewidth=0.9, alpha=0.9,
                   label='BP/RP calibrado')
        ax_bp.fill_between(wavelength_bprp, flux_bprp,
                           alpha=0.08, color=ACCENT1)
        ax_bp.axvspan(330,  680, color='#1a3a5c', alpha=0.25,
                      label='BP (330–680 nm)')
        ax_bp.axvspan(640, 1050, color='#3a1a1a', alpha=0.25,
                      label='RP (640–1050 nm)')

        spectral_lines = [
            (656.3, 'H-alpha', ACCENT3),
            (849.8, 'Ca II',   ACCENT2),
            (854.2, 'Ca II',   ACCENT2),
            (866.2, 'Ca II',   ACCENT2),
        ]
        labeled = set()
        for lam, name, color in spectral_lines:
            if wavelength_bprp[0] <= lam <= wavelength_bprp[-1]:
                lbl = name if name not in labeled else None
                ax_bp.axvline(lam, color=color, linewidth=0.9,
                              linestyle='--', alpha=0.8, label=lbl)
                labeled.add(name)

        ax_bp.set_title('Espectro BP/RP  (336–1020 nm)',
                        fontweight='bold', fontsize=10)
        ax_bp.set_xlabel('Longitud de onda [nm]')
        ax_bp.set_ylabel('Flujo [W/m²/nm]')
        ax_bp.legend(fontsize=8, ncol=3, loc='upper right')
        ax_bp.grid(True, alpha=0.35)
    else:
        ax_bp.text(0.5, 0.5, 'Espectro BP/RP no disponible',
                   transform=ax_bp.transAxes,
                   ha='center', va='center',
                   color=DIM_TEXT, fontsize=11)
        ax_bp.set_title('Espectro BP/RP', fontweight='bold', fontsize=10)

    # ---- Row 2: RVS zoom (Ca II triplet) ----
    ax_rv = fig.add_subplot(gs[1])

    if wavelength_rvs is not None and flux_rvs is not None:
        ax_rv.plot(wavelength_rvs, flux_rvs,
                   color=ACCENT4, linewidth=0.9, alpha=0.9,
                   label='RVS crudo')
        ax_rv.fill_between(wavelength_rvs, flux_rvs,
                           alpha=0.08, color=ACCENT4)

        # ContinuumAgent pseudo-continuum windows
        continuum_windows = [
            (847.0, 849.0), (851.0, 853.5),
            (856.0, 860.0), (869.5, 874.0)
        ]
        for i, (w0, w1) in enumerate(continuum_windows):
            ax_rv.axvspan(w0, w1, color=ACCENT3, alpha=0.10,
                          label='Ventana pseudo-continuo' if i == 0 else None)

        for lam, label in [
            (849.8, 'Ca II 849.8'),
            (854.2, 'Ca II 854.2'),
            (866.2, 'Ca II 866.2'),
        ]:
            ax_rv.axvline(lam, color=ACCENT2, linewidth=1.1,
                          linestyle='--', alpha=0.85, label=label)

        ax_rv.set_xlim(845, 872)
        ax_rv.set_title('Espectro RVS  (zoom Ca II Triplete, 845–872 nm)',
                        fontweight='bold', fontsize=10)
        ax_rv.set_xlabel('Longitud de onda [nm]')
        ax_rv.set_ylabel('Flujo [u.a.]')
        ax_rv.legend(fontsize=8, ncol=3, loc='upper right')
        ax_rv.grid(True, alpha=0.35)
        ax_rv.text(0.02, 0.05,
                   'R ~ 11,500  |  Δλ ~ 0.075 nm',
                   transform=ax_rv.transAxes,
                   color=DIM_TEXT, fontsize=7.5)
    else:
        ax_rv.text(0.5, 0.5, 'Espectro RVS no disponible',
                   transform=ax_rv.transAxes,
                   ha='center', va='center',
                   color=DIM_TEXT, fontsize=11)
        ax_rv.set_title('Espectro RVS', fontweight='bold', fontsize=10)

    # ---- Physical parameters sidebar ----
    ax_info = fig.add_axes([0.80, 0.09, 0.19, 0.83], facecolor=PANEL_BG)
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    def _nss_str(v):
        if v is None or str(v).strip().lower() in ('nan', 'none', '<na>', ''):
            return 'None'
        return str(v)

    ruwe_color = ACCENT5 if (not np.isnan(ruwe) and ruwe > 1.4) else ACCENT3
    ew_color   = ACCENT2 if (not np.isnan(ew_ha) and ew_ha < 0) else ACCENT3
    nss_color  = ACCENT5 if _nss_str(nss) != 'None' else DIM_TEXT

    sidebar_rows = [
        ('PARÁMETROS FÍSICOS', None, TEXT_COLOR,  10, True),
        ('',                   None, DIM_TEXT,      8, False),
        ('T_eff',  _fmt(teff, 0, ' K'),   ACCENT1, 9, False),
        ('log g',  _fmt(logg, 2, ' dex'), ACCENT1, 9, False),
        ('[M/H]',  _fmt(mh,   2, ' dex'), ACCENT1, 9, False),
        ('M_G',    abs_mag_str,            ACCENT1, 9, False),
        ('',                   None, DIM_TEXT,      8, False),
        ('CALIDAD',            None, TEXT_COLOR,  10, True),
        ('',                   None, DIM_TEXT,      8, False),
        ('RUWE',   _fmt(ruwe, 3),          ruwe_color, 9, False),
        ('Plx SNR',
         _fmt(plx / plx_e if not (np.isnan(plx) or np.isnan(plx_e)) else np.nan, 1),
         ACCENT1, 9, False),
        ('',                   None, DIM_TEXT,      8, False),
        ('LÍNEAS',             None, TEXT_COLOR,  10, True),
        ('',                   None, DIM_TEXT,      8, False),
        ('EW H-alpha', _fmt(ew_ha, 3, ' Å'), ew_color, 9, False),
        ('',                   None, DIM_TEXT,      8, False),
        ('BINARIEDAD',         None, TEXT_COLOR,  10, True),
        ('',                   None, DIM_TEXT,      8, False),
        ('NSS type', _nss_str(nss),             nss_color,  9, False),
        ('RUWE > 1.4',
         'SÍ' if (not np.isnan(ruwe) and ruwe > 1.4) else 'No',
         ruwe_color, 9, False),
        ('',                   None, DIM_TEXT,      8, False),
        ('ESPECTROS',          None, TEXT_COLOR,  10, True),
        ('',                   None, DIM_TEXT,      8, False),
        ('BP/RP', 'Sí' if has_bprp else 'No',
         ACCENT3 if has_bprp else ACCENT2, 9, False),
        ('RVS',   'Sí' if has_rvs  else 'No',
         ACCENT3 if has_rvs  else ACCENT2, 9, False),
    ]

    y_pos = 0.97
    for label, value, color, size, bold in sidebar_rows:
        if value is None:
            ax_info.text(
                0.05, y_pos, label,
                color=color, fontsize=size,
                fontweight='bold' if bold else 'normal',
                transform=ax_info.transAxes,
                fontfamily='monospace'
            )
        else:
            ax_info.text(0.05, y_pos, f'{label}:',
                         color=DIM_TEXT, fontsize=size,
                         transform=ax_info.transAxes,
                         fontfamily='monospace')
            ax_info.text(0.95, y_pos, value,
                         color=color, fontsize=size, ha='right',
                         transform=ax_info.transAxes,
                         fontfamily='monospace')
        y_pos -= 0.042

    # Vertical separator line
    fig.add_axes(
        [0.785, 0.09, 0.002, 0.83], facecolor=GRID_COLOR
    ).axis('off')

    ruta = carpeta / f'espectros_{source_id}.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f"  Panel guardado: {ruta}")
    plt.close(fig)


# =============================================================================
# STAR SELECTION FOR SPECTRAL PANEL
# =============================================================================

def select_demo_star(df: pd.DataFrame, db_folder: Path) -> Optional[pd.Series]:
    """
    Selects the best star for the spectral panel using this priority:
        1. has_bprp_spectrum AND has_rvs_spectrum (both available from disk)
        2. Reliable parallax (SNR > 5)
        3. Clean astrometry (RUWE < 1.4)
        4. Physical parameters available (teff, mh)

    Falls back to any star with at least BP/RP if no star has both.
    """
    df = df.copy()
    df['_plx_snr'] = df['parallax'] / df['parallax_error']

    has_bprp = df.get('has_bprp_spectrum', pd.Series(False, index=df.index))
    has_rvs  = df.get('has_rvs_spectrum',  pd.Series(False, index=df.index))

    # Priority 1: both spectra + clean parameters
    mask_ideal = (
        has_bprp & has_rvs
        & df['teff_gspphot'].notna()
        & df['mh_gspphot'].notna()
        & (df['parallax'] > 0)
        & (df['_plx_snr'] > 5)
        & (df['ruwe'] < 1.4)
    )
    if mask_ideal.any():
        return df[mask_ideal].iloc[0]

    # Fallback: at least BP/RP
    mask_bprp = has_bprp & df['teff_gspphot'].notna()
    if mask_bprp.any():
        logger.warning("No star with both spectra found. Using BP/RP-only star.")
        return df[mask_bprp].iloc[0]

    # Last resort: first row
    logger.warning("No ideal star found. Using first row in catalog.")
    return df.iloc[0]


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("GAIA DR3 — VISUALIZATION & DIAGNOSTICS")
    print("=" * 60)

    # Resolve input DB folder
    try:
        db_folder = resolve_db_folder(args.db)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    print(f"\n  Source DB folder : {db_folder}")

    # Create output VIZ folder
    viz_folder = resolve_viz_output_folder()
    print(f"  Output VIZ folder: {viz_folder}")

    # Load catalog and manifest
    print("\nLoading catalog...")
    try:
        df, manifest = load_db(db_folder)
        print(f"  {len(df)} stars loaded.")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # Panel 1: Catalog diagnostics
    print("\nGenerating catalog diagnostic panel...")
    panel_diagnostico(df, manifest, carpeta=viz_folder)

    # Select demo star for spectral panel
    fila = select_demo_star(df, db_folder)
    sid  = int(fila['source_id'])
    print(f"\nSelected star for spectral panel: {sid}")
    print(f"  T_eff = {fila.get('teff_gspphot', 'N/A')} K  |  "
          f"[M/H] = {fila.get('mh_gspphot', 'N/A')}  |  "
          f"RUWE = {fila.get('ruwe', 'N/A')}")

    # Load spectra from disk (no re-download)
    wave_bprp, flux_bprp = load_bprp_spectrum(db_folder, sid)
    wave_rvs,  flux_rvs  = load_rvs_spectrum(db_folder, sid)

    if wave_bprp is not None:
        print(f"  BP/RP: {len(wave_bprp)} points, "
              f"{wave_bprp[0]:.0f}–{wave_bprp[-1]:.0f} nm")
    else:
        print("  BP/RP: not available on disk.")

    if wave_rvs is not None:
        print(f"  RVS:   {len(wave_rvs)} points, "
              f"{wave_rvs[0]:.1f}–{wave_rvs[-1]:.1f} nm")
    else:
        print("  RVS:   not available on disk.")

    # Panel 2: Parallel spectra
    print("\nGenerating spectral panel...")
    panel_espectros_paralelo(
        source_id=sid,
        fila_catalogo=fila,
        wavelength_bprp=wave_bprp,
        flux_bprp=flux_bprp,
        wavelength_rvs=wave_rvs,
        flux_rvs=flux_rvs,
        carpeta=viz_folder
    )

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"  diagnostico_catalogo.png  →  {viz_folder}")
    print(f"  espectros_{sid}.png       →  {viz_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()