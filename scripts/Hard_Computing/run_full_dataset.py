"""
Full dataset run — HC Pipeline v2 sobre el corpus completo.
===========================================================
Lee automáticamente la carpeta DB-N más reciente bajo data/raw/ (o la que
se especifique con --db-folder) y genera los contratos JSON en la carpeta
versionada JSON-hc-prueba-{N+1}-{YYYYMMDD} del mismo directorio.

Uso básico (usa la carpeta DB más reciente):
    python run_full_dataset.py

Seleccionar una carpeta específica:
    python run_full_dataset.py --db-folder data/raw/DB-1-2024-01-01

VERSIÓN MEJORADA con Quality Score, Timestamps y Estadísticas Ca II
        
Opciones adicionales:
    --max-stars N        Limitar a N estrellas (útil para pruebas rápidas)
    --require-rvs        Procesar solo estrellas con espectro RVS disponible
    --require-bprp       Procesar solo estrellas con espectro BP/RP disponible
    --workers N          Número de procesos paralelos (default: 1)
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
 
import pandas as pd
 
# ---------------------------------------------------------------------------
# Importar el orquestador v2
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from scripts.Hard_Computing.hc_pipeline_orchestrator_v2 import HCPipelineOrchestrator  # noqa: E402
 
# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/cesar/Documentos/Tesis-cimat/Estancia/data/raw")
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_full_dataset")
 
 
# ===========================================================================
# MEJORA 1: QUALITY SCORE COMPUTATION
# ===========================================================================
def compute_quality_score(contract: dict) -> float:
    """
    Calcula score de calidad global [0.0 - 1.0].
    
    Factores:
    - Paralaje confiable: crítico (0.0 si falla)
    - BP/RP alta calidad: importante (0.5 penalización)
    - RVS estable: deseable (0.7 penalización)
    """
    score = 1.0
    
    # Factor 1: Paralaje
    if not contract['logical_flags'].get('is_reliable_parallax', False):
        score *= 0.0
    
    # Factor 2: BP/RP
    bprp = contract['spectral_summary'].get('bprp_continuum', {})
    if bprp.get('status') != 'success':
        score *= 0.3
    elif not bprp.get('high_snr', False):
        score *= 0.5
    
    # Factor 3: RVS
    rvs = contract['spectral_summary'].get('rvs_continuum', {})
    if rvs.get('status') != 'success':
        score *= 0.7
    elif not rvs.get('stable', False):
        score *= 0.85
    
    # Bonus por calidad espectral excepcional
    if (bprp.get('high_snr', False) and 
        rvs.get('high_snr', False) and 
        rvs.get('stable', False)):
        score = min(1.0, score * 1.1)
    
    return round(score, 4)
 
 
def add_quality_scores(contracts: list) -> list:
    """
    MEJORA 1 + 2: Añade quality_score y processing_timestamp a cada contrato.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    for contract in contracts:
        contract['quality_score'] = compute_quality_score(contract)
        contract['processing_timestamp'] = timestamp
    
    logger.info(f"Quality scores computed for {len(contracts)} contracts")
    return contracts
 
 
# ===========================================================================
# MEJORA 3: CA II TRIPLET STATISTICS
# ===========================================================================
def compute_cat_statistics(contracts: list) -> dict:
    """
    Calcula estadísticas detalladas del Ca II Triplet.
    """
    stats = {
        'total_stars': len(contracts),
        'stars_with_cat_data': 0,
        'high_quality_detections': 0,
        'per_line_stats': {
            '849.8nm': {'attempted': 0, 'success': 0, 'high_quality': 0},
            '854.2nm': {'attempted': 0, 'success': 0, 'high_quality': 0},
            '866.2nm': {'attempted': 0, 'success': 0, 'high_quality': 0},
        }
    }
    
    line_wavelengths = {0: '849.8nm', 1: '854.2nm', 2: '866.2nm'}
    
    for contract in contracts:
        cat_triplet = contract.get('spectral_summary', {}).get('cat_triplet', [])
        
        if not cat_triplet:
            continue
        
        stats['stars_with_cat_data'] += 1
        has_high_quality = False
        
        for idx, line in enumerate(cat_triplet):
            line_key = line_wavelengths.get(idx)
            if not line_key:
                continue
            
            stats['per_line_stats'][line_key]['attempted'] += 1
            
            if line.get('status') != 'failed':
                stats['per_line_stats'][line_key]['success'] += 1
            
            if line.get('high_quality_fit', False):
                stats['per_line_stats'][line_key]['high_quality'] += 1
                has_high_quality = True
        
        if has_high_quality:
            stats['high_quality_detections'] += 1
    
    # Calcular porcentajes
    for line_key in stats['per_line_stats']:
        attempted = stats['per_line_stats'][line_key]['attempted']
        if attempted > 0:
            success = stats['per_line_stats'][line_key]['success']
            hq = stats['per_line_stats'][line_key]['high_quality']
            stats['per_line_stats'][line_key]['success_rate'] = round(100 * success / attempted, 1)
            stats['per_line_stats'][line_key]['hq_rate'] = round(100 * hq / attempted, 1)
    
    return stats
 
 
def print_cat_statistics(cat_stats: dict) -> None:
    """
    Imprime reporte de Ca II Triplet.
    """
    print("\n" + "=" * 64)
    print("  Ca II INFRARED TRIPLET — DETECTION STATISTICS")
    print("=" * 64)
    
    total = cat_stats['total_stars']
    with_data = cat_stats['stars_with_cat_data']
    hq_det = cat_stats['high_quality_detections']
    
    print(f"  Stars with CaII data:         {with_data}/{total} ({100*with_data/total:.1f}%)")
    print(f"  High-quality detections:      {hq_det}/{total} ({100*hq_det/total:.1f}%)")
    
    print("\n  Per-line breakdown:")
    print(f"  {'Line':12s} {'Attempted':>10s} {'Success':>10s} {'High-Q':>10s}")
    print("  " + "-" * 50)
    
    for line, data in cat_stats['per_line_stats'].items():
        att = data.get('attempted', 0)
        suc = data.get('success', 0)
        hq = data.get('high_quality', 0)
        suc_rate = data.get('success_rate', 0)
        hq_rate = data.get('hq_rate', 0)
        
        print(f"  {line:12s} {att:>10d} {suc:>5d} ({suc_rate:4.1f}%) {hq:>5d} ({hq_rate:4.1f}%)")
    
    print()
 
 
# ---------------------------------------------------------------------------
# Helpers (sin cambios)
# ---------------------------------------------------------------------------
 
def find_latest_db_folder(root: Path) -> Path:
    """
    Busca la carpeta DB-N-YYYY-MM-DD con N más alto dentro de root.
    """
    candidates = [
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("DB-")
    ]
    if not candidates:
        raise RuntimeError(
            f"No se encontró ninguna carpeta DB-* en '{root}'.\n"
            "Ejecuta primero gaia_dataset_builder.py para generar el corpus."
        )
 
    def _db_number(d: Path) -> int:
        parts = d.name.split("-")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return -1
 
    latest = max(candidates, key=_db_number)
    return latest
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HC Pipeline v2 — run completo sobre corpus Gaia DR3."
    )
    parser.add_argument(
        "--db-folder", type=Path, default=None,
        help=(
            "Carpeta DB-N-YYYY-MM-DD a procesar. "
            "Si se omite, se usa la de N más alto en data/raw/."
        ),
    )
    parser.add_argument(
        "--max-stars", type=int, default=None,
        help="Limitar el número de estrellas procesadas (útil para pruebas).",
    )
    parser.add_argument(
        "--require-rvs", action="store_true",
        help="Omitir estrellas sin archivo RVS disponible.",
    )
    parser.add_argument(
        "--require-bprp", action="store_true",
        help="Omitir estrellas sin espectro BP/RP disponible.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Número de procesos paralelos (default: 1).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DATA_ROOT,
        help=f"Directorio raíz para las carpetas JSON-hc-prueba-* (default: {DATA_ROOT}).",
    )
    return parser.parse_args()
 
 
def print_summary(summary: pd.DataFrame, output_path: Path, cat_stats: dict = None) -> None:
    """
    Imprime resumen mejorado con quality scores y Ca II stats.
    """
    sep = "=" * 64
 
    print(f"\n{sep}")
    print("  CORPUS COMPLETO — DISTRIBUCIÓN DE FLAGS")
    print(sep)
 
    flag_cols = [
        "is_giant", "is_metal_poor",
        "is_binary_candidate", "is_high_velocity", "has_emission",
    ]
    for col in flag_cols:
        if col not in summary.columns:
            continue
        n_true = summary[col].sum()
        pct = 100.0 * n_true / len(summary) if len(summary) > 0 else 0.0
        print(f"  {col:28s}: {int(n_true):4d}  ({pct:.1f} %)")
 
    print(f"\n  Total procesadas: {len(summary)}")
 
    print(f"\n{sep}")
    print("  ESTADO ASTROMETRÍA")
    print(sep)
    print(summary["astrometry_status"].value_counts().to_string())
 
    print(f"\n{sep}")
    print("  DISPONIBILIDAD DE ESPECTROS")
    print(sep)
    for col, label in [
        ("bprp_snr", "BP/RP  con ajuste de continuo"),
        ("rvs_snr",  "RVS    con ajuste de continuo"),
    ]:
        if col in summary.columns:
            n = summary[col].notna().sum()
            print(f"  {label}: {n}")
 
    # ===== MEJORA: QUALITY SCORE DISTRIBUTION =====
    if 'quality_score' in summary.columns:
        print(f"\n{sep}")
        print("  QUALITY SCORE DISTRIBUTION")
        print(sep)
        
        perfect = (summary['quality_score'] >= 0.95).sum()
        good = ((summary['quality_score'] >= 0.7) & (summary['quality_score'] < 0.95)).sum()
        fair = ((summary['quality_score'] >= 0.4) & (summary['quality_score'] < 0.7)).sum()
        poor = (summary['quality_score'] < 0.4).sum()
        
        print(f"  Perfect (≥0.95): {perfect:4d} ({100*perfect/len(summary):.1f}%)")
        print(f"  Good (0.7-0.95): {good:4d} ({100*good/len(summary):.1f}%)")
        print(f"  Fair (0.4-0.70): {fair:4d} ({100*fair/len(summary):.1f}%)")
        print(f"  Poor (<0.40):    {poor:4d} ({100*poor/len(summary):.1f}%)")
        print(f"\n  Mean quality score: {summary['quality_score'].mean():.3f}")
        print(f"  Median quality score: {summary['quality_score'].median():.3f}")
 
    print(f"\n{sep}")
    print("  ESTRELLAS CON EMISIÓN H-alpha")
    print(sep)
    emitters = summary[summary["has_emission"] == True]  # noqa: E712
    print(f"  Total: {len(emitters)}")
    if len(emitters) > 0:
        print(
            emitters[["source_id", "teff_k", "abs_mag", "is_giant"]]
            .to_string(index=False)
        )
 
    print(f"\n{sep}")
    print("  DISTRIBUCIÓN DE CLASES (gigante × metal-pobre × binaria × alta-vel)")
    print(sep)
    group_cols = [c for c in flag_cols[:4] if c in summary.columns]
    if group_cols:
        print(summary.groupby(group_cols).size().to_string())
 
    # ===== MEJORA 3: CA II STATISTICS =====
    if cat_stats:
        print_cat_statistics(cat_stats)
 
    print(f"\nJSON guardado en: {output_path}\n")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    args = parse_args()
 
    # --- Resolver carpeta DB ---
    if args.db_folder is not None:
        db_folder = args.db_folder
        if not db_folder.exists():
            logger.error(f"La carpeta especificada no existe: {db_folder}")
            sys.exit(1)
    else:
        try:
            db_folder = find_latest_db_folder(DATA_ROOT)
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
 
    logger.info(f"Carpeta DB seleccionada: {db_folder}")
 
    # Verificar archivos mínimos
    missing = [
        f for f in ("catalog.csv", "spectra_bprp.npy", "spectra_bprp_ids.npy",
                    "sampling_bprp.npy")
        if not (db_folder / f).exists()
    ]
    if missing:
        logger.warning(
            f"Archivos faltantes en '{db_folder}': {missing}. "
            "Algunos espectros pueden no estar disponibles."
        )
 
    rvs_dir = db_folder / "rvs"
    if not rvs_dir.exists():
        logger.warning(f"Directorio RVS no encontrado: {rvs_dir}")
 
    # --- Instanciar orquestador ---
    orchestrator = HCPipelineOrchestrator(
        db_folder=db_folder,
        n_workers=args.workers,
    )
 
    # --- Ejecutar pipeline ---
    contracts = orchestrator.run(
        max_stars=args.max_stars,
        require_bprp=args.require_bprp,
        require_rvs=args.require_rvs,
    )
 
    if not contracts:
        logger.error("El pipeline no produjo ningún contrato. Revisa los datos de entrada.")
        sys.exit(1)
 
    # ===== APLICAR MEJORAS 1 Y 2 =====
    contracts = add_quality_scores(contracts)
    
    # ===== CALCULAR MEJORA 3 =====
    cat_stats = compute_cat_statistics(contracts)
 
    # --- Guardar resultados ---
    output_path = HCPipelineOrchestrator.save_results(
        contracts, base_dir=args.output_root
    )
 
    # --- Resumen en consola ---
    summary = orchestrator.summarize(contracts)
 
    # Añadir columnas de SNR y quality_score al summary
    summary['bprp_snr'] = [
        c.get('spectral_summary', {}).get('bprp_continuum', {}).get('snr')
        for c in contracts
    ]
    summary['rvs_snr'] = [
        c.get('spectral_summary', {}).get('rvs_continuum', {}).get('snr')
        for c in contracts
    ]
    summary['quality_score'] = [
        c.get('quality_score', 0.0)
        for c in contracts
    ]
    
    # Añadir flag de detección Ca II
    summary['has_cat_detection'] = [
        any(line.get('high_quality_fit', False) 
            for line in c.get('spectral_summary', {}).get('cat_triplet', []))
        for c in contracts
    ]
 
    # ===== IMPRIMIR RESUMEN MEJORADO =====
    print_summary(summary, output_path, cat_stats)
 
 
if __name__ == "__main__":
    main()
