# -*- coding: utf-8 -*-
"""
HC Pipeline Orchestrator v2
============================
Versión actualizada del orquestador HC para trabajar con los datos generados
por gaia_dataset_builder.py.

Layout de datos esperado (carpeta DB-N-YYYY-MM-DD):
    catalog.csv               — catálogo balanceado con metadatos tabulares
    spectra_bprp.npy          — matriz de flujos BP/RP  (n_estrellas × n_longitudes)
    spectra_bprp_ids.npy      — source_id alineados con las filas de la matriz
    sampling_bprp.npy         — grid de longitudes de onda en nm  (n_longitudes,)
    run_manifest.json
    rvs/
        {source_id}.npz       — claves: wavelength_nm, flux, flux_error

Uso básico:
    orch = HCPipelineOrchestrator(
        db_folder=Path("/home/cesar/.../data/raw/DB-1-2024-01-01")
    )
    contracts = orch.run(max_stars=200, require_rvs=True)
    HCPipelineOrchestrator.save_results(contracts)
"""

import json
import logging
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.modeling import fitting, models
from astropy.utils.exceptions import AstropyUserWarning
from scipy.interpolate import splev, splrep

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HC_Orchestrator_v2")


# ===========================================================================
# AGENTE 1: AstrometryAgent
# ===========================================================================

K_G_DEFAULT = 2.74  # coeficiente de extinción banda G — Wang & Chen (2019)


class AstrometryAgent:
    """
    Deriva propiedades intrínsecas estelares a partir de la astrometría y
    fotometría de Gaia DR3.

    Notas:
        - pmra en Gaia DR3 ya viene como mu_alpha* (corregido por cos δ).
        - Magnitudes absolutas y velocidades tangenciales se suprimen cuando
          el SNR de paralaje < 5.
    """

    def __init__(self, extinction_coeff: float = K_G_DEFAULT):
        self.k_g = extinction_coeff

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        source_id = data.get("source_id", "UNKNOWN")

        res: Dict[str, Any] = {
            "module": "AstrometryAgent",
            "status": "failed",
            "measurements": {
                "absolute_magnitude_g": None,
                "tangential_velocity_km_s": None,
                "extinction_ag": None,
                "teff_k": None,
            },
            "logical_flags": {
                "is_reliable_parallax": False,
                "is_giant": False,
                "is_high_velocity": False,
                "is_binary_candidate": False,
            },
        }

        try:
            p = float(data["parallax"])
            p_err = float(data["parallax_error"])
            g_mag = float(data["phot_g_mean_mag"])
            ebp_rp = data["ebpminrp_gspphot"]
            teff = data["teff_gspphot"]
            ruwe = float(data["ruwe"])
            pmra = float(data["pmra"])
            pmdec = float(data["pmdec"])

            teff_val = float(teff) if not _is_nan(teff) else None
            res["measurements"]["teff_k"] = int(teff_val) if teff_val is not None else None

            if p <= 0:
                logger.warning(
                    f"[{source_id}] Paralaje negativo/cero ({p:.4f} mas). "
                    "Cantidades dependientes de distancia no calculadas."
                )
                res["logical_flags"]["is_binary_candidate"] = bool(ruwe > 1.4)
                res["status"] = "partial"
                return res

            if p_err == 0:
                raise ZeroDivisionError("parallax_error es cero.")

            is_reliable = bool((p / p_err) > 5)

            if _is_nan(ebp_rp):
                logger.warning(f"[{source_id}] ebpminrp_gspphot es NaN. A_G = 0.0.")
                ebp_rp = 0.0

            # A_G = k_G · E(BP-RP)
            a_g = self.k_g * float(ebp_rp)
            # M_G = G + 5 + 5·log10(π_mas / 1000) − A_G
            abs_mag = g_mag + 5 + 5 * math.log10(p / 1000) - a_g
            # V_tan = 4.74 · μ_total / π   [km/s]
            pm_total = math.sqrt(pmra**2 + pmdec**2)
            v_tan = 4.74 * pm_total / p

            res["measurements"]["extinction_ag"] = round(float(a_g), 4)

            if is_reliable:
                res["measurements"]["absolute_magnitude_g"] = round(float(abs_mag), 4)
                res["measurements"]["tangential_velocity_km_s"] = round(float(v_tan), 4)
            else:
                logger.info(
                    f"[{source_id}] SNR paralaje = {p/p_err:.2f} < 5. "
                    "abs_mag y v_tan suprimidas."
                )

            res["logical_flags"] = {
                "is_reliable_parallax": is_reliable,
                "is_giant": bool(
                    abs_mag < 3.0 and teff_val is not None
                    and teff_val < 7000 and is_reliable
                ),
                "is_high_velocity": bool(v_tan > 200 and is_reliable),
                "is_binary_candidate": bool(ruwe > 1.4),
            }
            res["status"] = "success"

        except KeyError as e:
            logger.error(f"[{source_id}] AstrometryAgent — campo faltante: {e}")
        except ZeroDivisionError as e:
            logger.error(f"[{source_id}] AstrometryAgent — {e}")
        except Exception as e:
            logger.error(f"[{source_id}] AstrometryAgent — error inesperado: {e}")

        return res


# ===========================================================================
# AGENTE 2: ContinuumAgent
# ===========================================================================

# Ventanas de pseudo-continuo RVS (nm) — conservadoras para evitar las alas del CaT
RVS_CONTINUUM_WINDOWS: List[Tuple[float, float]] = [
    (847.0, 849.0),
    (851.0, 853.5),
    (856.0, 860.0),
    (869.5, 874.0),
]


class ContinuumAgent:
    """
    Normaliza espectros estelares mediante splines cúbicos con sigma-clipping
    iterativo.  Usa un polinomio de Chebyshev como respaldo cuando el spline
    diverge (oscilaciones de Runge en los bordes o grandes gaps de absorción).
    """

    def __init__(
        self,
        sigma_lower: float = 2.0,
        sigma_upper: float = 3.0,
        iterations: int = 5,
        rvs_mode: bool = False,
    ):
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.iterations = iterations
        self.rvs_mode = rvs_mode

    def _build_rvs_window_mask(self, wave: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(wave), dtype=bool)
        for w_min, w_max in RVS_CONTINUUM_WINDOWS:
            mask |= (wave >= w_min) & (wave <= w_max)
        return mask

    @staticmethod
    def _chebyshev_fallback_fit(
        wave: np.ndarray,
        flux: np.ndarray,
        mask: np.ndarray,
        degree: int = 3,
    ) -> np.ndarray:
        """
        Polinomio de Chebyshev grado 3 evaluado en el dominio [-1, 1].
        Numéricamente superior a la base de potencias puras cerca de los bordes.
        """
        w_min, w_max = wave[mask].min(), wave[mask].max()
        w_scaled_all = 2.0 * (wave - w_min) / (w_max - w_min) - 1.0
        w_scaled_masked = w_scaled_all[mask]
        coeffs = np.polynomial.chebyshev.chebfit(w_scaled_masked, flux[mask], deg=degree)
        return np.polynomial.chebyshev.chebval(w_scaled_all, coeffs)

    def _sigma_clip_fit(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        initial_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Devuelve (continuum_model, máscara_final, método_usado)."""
        mask = initial_mask.copy() if initial_mask is not None else np.ones(len(flux), dtype=bool)
        continuum = np.ones_like(flux)
        spline_ok = False

        for _ in range(self.iterations):
            if np.sum(mask) < 10:
                break

            residuals_current = flux[mask] - np.mean(flux[mask])
            sigma_est = np.std(residuals_current)
            s_factor = np.sum(mask) * (sigma_est**2)

            try:
                tck = splrep(wave[mask], flux[mask], s=s_factor)
                candidate = splev(wave, tck)
                if np.any(candidate <= 0) or np.any(~np.isfinite(candidate)):
                    raise ValueError("Spline produjo valores no físicos.")
                continuum = candidate
                spline_ok = True
            except Exception as e:
                logger.debug(f"Iteración de spline fallida: {e}. Usando Chebyshev.")
                break

            residuals = flux - continuum
            std = np.std(residuals[mask])
            if std == 0:
                break

            new_mask = (
                (residuals > -self.sigma_lower * std)
                & (residuals < self.sigma_upper * std)
            )
            if initial_mask is not None:
                new_mask &= initial_mask

            if np.array_equal(mask, new_mask):
                break
            mask = new_mask

        if not spline_ok or np.any(continuum <= 0) or np.any(~np.isfinite(continuum)):
            try:
                continuum = self._chebyshev_fallback_fit(wave, flux, mask)
                return continuum, mask, "chebyshev"
            except Exception as e:
                logger.error(f"Chebyshev fallback también falló: {e}. Continuo plano.")
                continuum = np.full_like(flux, np.mean(flux[mask]))
                return continuum, mask, "flat_mean"

        return continuum, mask, "spline"

    def process(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        source_id: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "module": "ContinuumAgent",
            "status": "failed",
            "measurements": {
                "normalized_flux": None,
                "continuum_model": None,
                "mean_snr": 0.0,
            },
            "logical_flags": {
                "continuum_is_stable": False,
                "high_snr_continuum": False,
                "fit_diverged": False,
            },
        }

        try:
            initial_mask = (
                self._build_rvs_window_mask(wavelength) if self.rvs_mode else None
            )
            continuum_model, continuum_mask, fit_method = self._sigma_clip_fit(
                wavelength, flux, initial_mask
            )
            if fit_method != "spline":
                logger.warning(
                    f"[{source_id}] ContinuumAgent usó método de respaldo: '{fit_method}'."
                )

            if np.sum(continuum_mask) < 10:
                logger.error(f"[{source_id}] Máscara final < 10 puntos. Ajuste abortado.")
                res["logical_flags"]["fit_diverged"] = True
                return res

            if np.any(continuum_model <= 0):
                logger.error(f"[{source_id}] Modelo de continuo con valores no positivos.")
                res["logical_flags"]["fit_diverged"] = True
                return res

            norm_flux = flux / continuum_model
            signal = np.mean(flux[continuum_mask])
            noise = np.std((flux - continuum_model)[continuum_mask])
            snr = float(signal / noise) if noise > 0 else 0.0
            stability_std = float(np.std(norm_flux[continuum_mask]))

            res["measurements"] = {
                "normalized_flux": norm_flux.tolist(),
                "continuum_model": continuum_model.tolist(),
                "mean_snr": round(snr, 2),
                "fit_method": fit_method,
            }
            res["logical_flags"] = {
                "continuum_is_stable": bool(stability_std < 0.05),
                "high_snr_continuum": bool(snr > 20),
                "fit_diverged": False,
                "used_fallback": fit_method != "spline",
            }
            res["status"] = "success"

        except Exception as e:
            logger.error(f"[{source_id}] ContinuumAgent error crítico: {e}")

        return res


# ===========================================================================
# AGENTE 3: LineAgent
# ===========================================================================

FWHM_INSTRUMENTAL_RVS_NM = 0.075  # R~11500 a 860 nm → Δλ ~ 0.075 nm


class LineAgent:
    """
    Ajusta un perfil de Voigt a una línea espectral en un espectro normalizado.

    La ventana de extracción se pre-desplaza por efecto Doppler (Δλ = λ₀·v/c)
    cuando se proporciona una velocidad radial, evitando que estrellas de alta
    velocidad queden fuera de la ventana estática.
    """

    _C_KM_S = 299792.458

    def __init__(
        self,
        rest_wavelength: float,
        window_size_nm: float = 1.5,
        radial_velocity: Optional[float] = None,
    ):
        self.rest_wavelength = rest_wavelength
        self.window_size_nm = window_size_nm
        self.radial_velocity = radial_velocity
        self.fitter = fitting.LevMarLSQFitter()

    def _doppler_shifted_center(self) -> float:
        if self.radial_velocity is None or _is_nan(self.radial_velocity):
            return self.rest_wavelength
        delta_lambda = self.rest_wavelength * (self.radial_velocity / self._C_KM_S)
        shifted = self.rest_wavelength + delta_lambda
        logger.debug(
            f"Doppler: λ_rest={self.rest_wavelength:.3f} nm, "
            f"v_r={self.radial_velocity:.1f} km/s → λ_obs={shifted:.4f} nm"
        )
        return shifted

    def _isolate_line_window(
        self, wave: np.ndarray, flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        center = self._doppler_shifted_center()
        mask = (
            (wave >= center - self.window_size_nm)
            & (wave <= center + self.window_size_nm)
        )
        return wave[mask], flux[mask]

    def process(
        self,
        wavelength: np.ndarray,
        normalized_flux: np.ndarray,
        source_id: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "module": "LineAgent",
            "status": "failed",
            "target_line_nm": self.rest_wavelength,
            "doppler_shift_applied_nm": (
                round(self._doppler_shifted_center() - self.rest_wavelength, 4)
                if self.radial_velocity is not None else None
            ),
            "measurements": {
                "equivalent_width_aa": np.nan,
                "fwhm_total_nm": np.nan,
                "centroid_shift_nm": np.nan,
            },
            "logical_flags": {
                "high_quality_fit": False,
                "is_broad_line": False,
                "has_emission_core": False,
            },
        }

        try:
            x_win, y_win = self._isolate_line_window(wavelength, normalized_flux)

            if len(x_win) < 7:
                logger.error(
                    f"[{source_id}] Puntos insuficientes ({len(x_win)}) "
                    f"para línea {self.rest_wavelength} nm."
                )
                return res

            # Invertir espectro: líneas en absorción → picos positivos para Voigt
            y_inv = 1.0 - y_win
            amp_guess = max(float(np.max(y_inv)), 0.05)

            voigt_init = models.Voigt1D(
                x_0=self.rest_wavelength,
                amplitude_L=amp_guess,
                fwhm_L=0.05,
                fwhm_G=FWHM_INSTRUMENTAL_RVS_NM,
            )
            voigt_init.x_0.bounds = (self.rest_wavelength - 1.5, self.rest_wavelength + 1.5)
            voigt_init.amplitude_L.bounds = (-0.5, 1.2)
            voigt_init.fwhm_G.bounds = (FWHM_INSTRUMENTAL_RVS_NM, None)
            voigt_init.fwhm_L.bounds = (0.0, None)

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", AstropyUserWarning)
                fitted_model = self.fitter(voigt_init, x_win, y_inv)
                for w in caught_warnings:
                    logger.debug(
                        f"[{source_id}] Astropy warning en {self.rest_wavelength} nm: {w.message}"
                    )

            if self.fitter.fit_info["ierr"] not in [1, 2, 3, 4]:
                logger.warning(
                    f"[{source_id}] LevMarLSQ no convergió para línea {self.rest_wavelength} nm."
                )
                return res

            x_fine = np.linspace(self.rest_wavelength - 3.0, self.rest_wavelength + 3.0, 1000)
            fitted_fine = np.clip(fitted_model(x_fine), 0.0, None)
            ew_nm = float(np.trapz(fitted_fine, x_fine))
            ew_aa = ew_nm * 10.0  # nm → Å

            # FWHM total — aproximación de Thompson et al. (1987)
            f_l = fitted_model.fwhm_L.value
            f_g = fitted_model.fwhm_G.value
            fwhm_total = float(0.5346 * f_l + math.sqrt(0.2166 * f_l**2 + f_g**2))
            centroid_shift = float(fitted_model.x_0.value - self.rest_wavelength)
            peak_value = float(fitted_model(fitted_model.x_0.value))

            res["measurements"] = {
                "equivalent_width_aa": round(ew_aa, 3),
                "fwhm_total_nm": round(fwhm_total, 4),
                "centroid_shift_nm": round(centroid_shift, 4),
            }
            res["logical_flags"] = {
                "high_quality_fit": True,
                "is_broad_line": bool(fwhm_total > 0.5),
                "has_emission_core": bool(peak_value < 0),
            }
            res["status"] = "success"

        except Exception as e:
            logger.error(f"[{source_id}] LineAgent error crítico: {e}")

        return res


# ===========================================================================
# AGENTE 4: BinaryDetectorAgent
# ===========================================================================


class BinaryDetectorAgent:
    """
    Detecta sistemas múltiples no resueltos mediante RUWE, variabilidad de RV
    y la clasificación del catálogo NSS de Gaia DR3.
    """

    def __init__(self, ruwe_threshold: float = 1.4, min_rv_transits: int = 5):
        self.ruwe_threshold = ruwe_threshold
        self.min_rv_transits = min_rv_transits
        self.default_rv_error_threshold = 5.0

    def _check_nss_validity(self, nss_type: Any) -> bool:
        if _is_nan(nss_type) or nss_type is None:
            return False
        return str(nss_type).strip().lower() not in ("", "none", "nan", "null")

    def _get_adaptive_rv_threshold(self, teff: Any) -> float:
        """
        Estrellas calientes (≥7000 K): umbral 10 km/s (líneas anchas → RV error mayor).
        Estrellas M (<4000 K): umbral 2 km/s (jitter cromosférico intrínseco).
        """
        if _is_nan(teff):
            return self.default_rv_error_threshold
        teff_f = float(teff)
        if teff_f >= 7000:
            return 10.0
        if teff_f < 4000:
            return 2.0
        return self.default_rv_error_threshold

    def process(self, data: Dict[str, Any], source_id: str = "UNKNOWN") -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "module": "BinaryDetectorAgent",
            "status": "failed",
            "measurements": {
                "ruwe": None,
                "rv_error_km_s": None,
                "rv_nb_transits": None,
                "nss_solution": None,
                "adaptive_rv_threshold": None,
            },
            "logical_flags": {
                "is_astrometric_binary": False,
                "is_rv_variable": False,
                "is_confirmed_nss": False,
                "is_binary_candidate": False,
            },
        }

        try:
            ruwe = data.get("ruwe", np.nan)
            rv_err = data.get("radial_velocity_error", np.nan)
            rv_transits_raw = data.get("rv_nb_transits", 0)
            nss_type = data.get("nss_solution_type", None)
            teff = data.get("teff_gspphot", np.nan)

            try:
                rv_transits = int(rv_transits_raw) if not _is_nan(rv_transits_raw) else 0
            except (ValueError, TypeError):
                rv_transits = 0

            rv_threshold = self._get_adaptive_rv_threshold(teff)

            is_astro_binary = bool(not _is_nan(ruwe) and float(ruwe) > self.ruwe_threshold)
            is_rv_var = bool(
                not _is_nan(rv_err)
                and rv_transits >= self.min_rv_transits
                and float(rv_err) > rv_threshold
            )
            is_nss = self._check_nss_validity(nss_type)
            master_flag = is_astro_binary or is_rv_var or is_nss

            res["measurements"] = {
                "ruwe": float(ruwe) if not _is_nan(ruwe) else None,
                "rv_error_km_s": float(rv_err) if not _is_nan(rv_err) else None,
                "rv_nb_transits": rv_transits,
                "nss_solution": str(nss_type) if is_nss else None,
                "adaptive_rv_threshold": rv_threshold,
            }
            res["logical_flags"] = {
                "is_astrometric_binary": is_astro_binary,
                "is_rv_variable": is_rv_var,
                "is_confirmed_nss": is_nss,
                "is_binary_candidate": master_flag,
            }
            res["status"] = "success"

        except Exception as e:
            logger.error(f"[{source_id}] BinaryDetectorAgent error crítico: {e}")

        return res


# ===========================================================================
# HC PIPELINE ORCHESTRATOR v2
# ===========================================================================


class HCPipelineOrchestrator:
    """
    Orquesta los cuatro agentes HC sobre una carpeta DB-N-YYYY-MM-DD generada
    por gaia_dataset_builder.py.

    La matriz BP/RP se carga una sola vez en memoria (mmap) al construir el
    objeto; los archivos RVS .npz se abren por demanda, estrella a estrella.

    Args:
        db_folder:  Carpeta DB-N-YYYY-MM-DD que contiene catalog.csv,
                    spectra_bprp.npy, spectra_bprp_ids.npy, sampling_bprp.npy
                    y el subdirectorio rvs/.
        n_workers:  Número de procesos paralelos (1 = ejecución serial).
    """

    # Tripleta infrarroja del Ca II (nm) — diagnóstico principal en ventana RVS
    CAT_LINES_NM = [849.802, 854.209, 866.214]
    # H-alpha (nm) — a partir del espectro BP/RP
    HALPHA_NM = 656.281

    def __init__(self, db_folder: Path, n_workers: int = 1):
        self.db_folder = Path(db_folder)
        self.catalog_path = self.db_folder / "catalog.csv"
        self.rvs_dir = self.db_folder / "rvs"
        self.n_workers = max(1, n_workers)

        # Matriz BP/RP — cargada una vez con mmap para seguridad multiproceso
        self._bprp_matrix: Optional[np.ndarray] = None
        self._bprp_sampling: Optional[np.ndarray] = None
        self._bprp_id_to_idx: Dict[int, int] = {}
        self._load_bprp_matrix()

        self.astrometry_agent = AstrometryAgent()
        self.binary_agent = BinaryDetectorAgent()

    # ------------------------------------------------------------------
    # Carga de espectros
    # ------------------------------------------------------------------

    def _load_bprp_matrix(self) -> None:
        """Pre-carga la matriz BP/RP compartida como array memory-mapped."""
        matrix_path = self.db_folder / "spectra_bprp.npy"
        ids_path = self.db_folder / "spectra_bprp_ids.npy"
        sampling_path = self.db_folder / "sampling_bprp.npy"

        if not (matrix_path.exists() and ids_path.exists() and sampling_path.exists()):
            logger.warning(
                f"Arrays BP/RP no encontrados en '{self.db_folder}'. "
                "Espectros BP/RP no disponibles."
            )
            return

        self._bprp_matrix = np.load(str(matrix_path), mmap_mode="r")
        bprp_ids = np.load(str(ids_path))
        self._bprp_sampling = np.load(str(sampling_path))
        self._bprp_id_to_idx = {int(sid): idx for idx, sid in enumerate(bprp_ids)}
        logger.info(
            f"Matriz BP/RP cargada: shape={self._bprp_matrix.shape}, "
            f"λ={self._bprp_sampling[0]:.0f}–{self._bprp_sampling[-1]:.0f} nm, "
            f"{len(bprp_ids)} estrellas."
        )

    def _load_bprp_spectrum(
        self, source_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Devuelve (wavelength_nm, flux) de la fila correspondiente en la
        matriz compartida, o None si la estrella no está o la fila es todo NaN.
        """
        if self._bprp_matrix is None or self._bprp_sampling is None:
            return None
        try:
            idx = self._bprp_id_to_idx.get(int(source_id))
        except (ValueError, TypeError):
            return None
        if idx is None:
            return None
        flux = np.array(self._bprp_matrix[idx], dtype=float)
        if np.all(np.isnan(flux)):
            return None
        return self._bprp_sampling.copy(), flux

    def _load_rvs_spectrum(
        self, source_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Carga wavelength_nm y flux de rvs/{source_id}.npz.
        Devuelve None si el archivo no existe o el flujo es todo NaN.
        """
        npz_path = self.rvs_dir / f"{source_id}.npz"
        if not npz_path.exists():
            return None
        try:
            data = np.load(str(npz_path))
            wave = np.array(data["wavelength_nm"], dtype=float)
            flux = np.array(data["flux"], dtype=float)
            if len(wave) == 0 or np.all(np.isnan(flux)):
                return None
            return wave, flux
        except Exception as e:
            logger.warning(f"[{source_id}] No se pudo cargar espectro RVS: {e}")
            return None

    # ------------------------------------------------------------------
    # Catálogo
    # ------------------------------------------------------------------

    def _load_catalog(self) -> pd.DataFrame:
        """Carga catalog.csv con guardas de tipo para source_id y nss_solution_type."""
        df = pd.read_csv(
            self.catalog_path,
            dtype={"source_id": str, "nss_solution_type": str},
            low_memory=False,
        )
        if "nss_solution_type" in df.columns:
            df["nss_solution_type"] = df["nss_solution_type"].apply(
                lambda x: None
                if _is_nan(x) or str(x).strip().lower() in ("nan", "")
                else str(x).strip()
            )
        logger.info(f"Catálogo cargado: {len(df)} filas, {len(df.columns)} columnas.")
        return df

    # ------------------------------------------------------------------
    # Procesamiento espectral
    # ------------------------------------------------------------------

    def _run_spectral_agents(
        self,
        source_id: str,
        row: Dict[str, Any],
        teff: Optional[float],
    ) -> Dict[str, Any]:
        """
        Ejecuta ContinuumAgent + LineAgent sobre los espectros disponibles
        de la estrella.  La VR se reenvía a cada LineAgent para pre-desplazar
        la ventana de extracción en estrellas de alta velocidad.
        """
        spectral_contracts: Dict[str, Any] = {}
        radial_velocity: Optional[float] = _safe_float(row.get("radial_velocity"))

        # --- Ruta BP/RP ---
        bprp = self._load_bprp_spectrum(source_id)
        if bprp is not None:
            wave_bprp, flux_bprp = bprp
            cont_result = ContinuumAgent(rvs_mode=False).process(
                wave_bprp, flux_bprp, source_id
            )
            spectral_contracts["bprp_continuum"] = cont_result

            if cont_result["status"] == "success":
                norm_flux = np.array(cont_result["measurements"]["normalized_flux"])
                spectral_contracts["halpha_fit"] = LineAgent(
                    rest_wavelength=self.HALPHA_NM,
                    radial_velocity=radial_velocity,
                ).process(wave_bprp, norm_flux, source_id)

        # --- Ruta RVS ---
        rvs = self._load_rvs_spectrum(source_id)
        if rvs is not None:
            wave_rvs, flux_rvs = rvs
            cont_result_rvs = ContinuumAgent(rvs_mode=True).process(
                wave_rvs, flux_rvs, source_id
            )
            spectral_contracts["rvs_continuum"] = cont_result_rvs

            if cont_result_rvs["status"] == "success":
                norm_flux_rvs = np.array(
                    cont_result_rvs["measurements"]["normalized_flux"]
                )
                spectral_contracts["cat_fits"] = [
                    LineAgent(
                        rest_wavelength=line_nm,
                        radial_velocity=radial_velocity,
                    ).process(wave_rvs, norm_flux_rvs, source_id)
                    for line_nm in self.CAT_LINES_NM
                ]

        return spectral_contracts

    # ------------------------------------------------------------------
    # Ensamblaje del contrato
    # ------------------------------------------------------------------

    def _merge_to_sc_contract(
        self,
        source_id: str,
        row: Dict[str, Any],
        astro_result: Dict[str, Any],
        binary_result: Dict[str, Any],
        spectral_contracts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fusiona todos los contratos de agentes en el JSON SC-ready canónico."""
        astro_meas = astro_result.get("measurements", {})
        astro_flags = astro_result.get("logical_flags", {})
        bin_flags = binary_result.get("logical_flags", {})

        physical_vector: Dict[str, Any] = {
            "abs_mag": astro_meas.get("absolute_magnitude_g"),
            "teff_k": astro_meas.get("teff_k"),
            "metallicity": _safe_float(row.get("mh_gspphot")),
            "fe_h": _safe_float(row.get("fem_gspspec")),
            "alpha_fe": _safe_float(row.get("alphafe_gspspec")),
            "logg": _safe_float(row.get("logg_gspphot")),
            "v_tan": astro_meas.get("tangential_velocity_km_s"),
            "extinction_ag": astro_meas.get("extinction_ag"),
        }

        # Flag de emisión H-alpha — el valor del catálogo tiene prioridad
        # Convenio ESP-ELS: EW > 0 = absorción, EW < 0 = emisión
        ew_halpha_raw = row.get("ew_espels_halpha")
        halpha_flag_raw = row.get("ew_espels_halpha_flag")
        has_catalog_halpha = (
            not _is_nan(ew_halpha_raw)
            and not _is_nan(halpha_flag_raw)
            and float(halpha_flag_raw) == 0.0
        )
        has_emission_catalog = bool(has_catalog_halpha and float(ew_halpha_raw) < 0)

        halpha_fit_contract = spectral_contracts.get("halpha_fit", {})
        has_emission_fit = bool(
            halpha_fit_contract.get("logical_flags", {}).get("has_emission_core", False)
        )

        logical_flags: Dict[str, Any] = {
            "is_reliable_parallax": astro_flags.get("is_reliable_parallax", False),
            "is_giant": astro_flags.get("is_giant", False),
            "is_metal_poor": bool(
                physical_vector["metallicity"] is not None
                and physical_vector["metallicity"] < -1.0
            ),
            "is_binary_candidate": bin_flags.get("is_binary_candidate", False),
            "is_high_velocity": astro_flags.get("is_high_velocity", False),
            "has_emission": has_emission_catalog or has_emission_fit,
            "has_emission_source": (
                "catalog" if has_emission_catalog
                else ("spectral_fit" if has_emission_fit else "none")
            ),
        }

        spectral_summary: Dict[str, Any] = {}

        if has_catalog_halpha:
            spectral_summary["halpha_catalog"] = {
                "ew_aa": round(float(ew_halpha_raw) * 10.0, 3),
                "flag": int(halpha_flag_raw),
            }

        cat_fits = spectral_contracts.get("cat_fits", [])
        if cat_fits:
            spectral_summary["cat_triplet"] = [
                {
                    "line_nm": f["target_line_nm"],
                    "ew_aa": f["measurements"].get("equivalent_width_aa"),
                    "fwhm_nm": f["measurements"].get("fwhm_total_nm"),
                    "status": f["status"],
                    "high_quality_fit": f["logical_flags"].get("high_quality_fit", False),
                }
                for f in cat_fits
            ]

        for key in ("bprp_continuum", "rvs_continuum"):
            cont = spectral_contracts.get(key)
            if cont:
                spectral_summary[key] = {
                    "status": cont["status"],
                    "snr": cont["measurements"].get("mean_snr"),
                    "stable": cont["logical_flags"].get("continuum_is_stable"),
                    "high_snr": cont["logical_flags"].get("high_snr_continuum"),
                    "fit_diverged": cont["logical_flags"].get("fit_diverged"),
                }

        return {
            "source_id": source_id,
            "pipeline_version": "HC-2.0",
            "astrometry_status": astro_result.get("status"),
            "binary_status": binary_result.get("status"),
            "physical_vector": physical_vector,
            "logical_flags": logical_flags,
            "binary_diagnostics": binary_result.get("measurements", {}),
            "spectral_summary": spectral_summary,
        }

    # ------------------------------------------------------------------
    # Punto de entrada por estrella (picklable para ProcessPoolExecutor)
    # ------------------------------------------------------------------

    def run_single(self, row: Dict[str, Any]) -> Dict[str, Any]:
        source_id = str(row.get("source_id", "UNKNOWN"))
        astro_result = self.astrometry_agent.process(row)
        binary_result = self.binary_agent.process(row, source_id=source_id)
        teff_raw = row.get("teff_gspphot")
        teff = float(teff_raw) if not _is_nan(teff_raw) else None
        spectral_contracts = self._run_spectral_agents(source_id, row, teff)
        return self._merge_to_sc_contract(
            source_id, row, astro_result, binary_result, spectral_contracts
        )

    # ------------------------------------------------------------------
    # Worker estático para procesamiento paralelo
    # ------------------------------------------------------------------

    @staticmethod
    def _process_chunk(rows: List[Dict[str, Any]], db_folder: Path) -> List[Dict[str, Any]]:
        """
        Worker para ProcessPoolExecutor.  Instancia un orquestador fresco en
        cada proceso hijo para evitar hazards de estado compartido.
        """
        worker = HCPipelineOrchestrator(db_folder=db_folder, n_workers=1)
        results: List[Dict[str, Any]] = []
        for row_dict in rows:
            sid = str(row_dict.get("source_id", "UNKNOWN"))
            try:
                results.append(worker.run_single(row_dict))
            except Exception as e:
                logger.error(f"[{sid}] Worker process falló: {e}")
        return results

    # ------------------------------------------------------------------
    # Punto de entrada principal del pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        max_stars: Optional[int] = None,
        require_bprp: bool = False,
        require_rvs: bool = False,
        chunk_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta el pipeline HC completo sobre el catálogo.

        Args:
            max_stars:    Límite de estrellas a procesar.
            require_bprp: Omite estrellas sin espectro BP/RP disponible.
            require_rvs:  Omite estrellas sin archivo rvs/{source_id}.npz.
            chunk_size:   Filas por chunk en modo paralelo.

        Returns:
            Lista de contratos JSON SC-ready.
        """
        df = self._load_catalog()

        all_rows: List[Dict[str, Any]] = []
        for row_dict in df.to_dict("records"):
            if max_stars is not None and len(all_rows) >= max_stars:
                break
            sid = str(row_dict.get("source_id", "UNKNOWN"))
            if require_bprp and self._load_bprp_spectrum(sid) is None:
                continue
            if require_rvs and self._load_rvs_spectrum(sid) is None:
                continue
            all_rows.append(row_dict)

        logger.info(
            f"Pipeline iniciando: {len(all_rows)} estrellas "
            f"(workers={self.n_workers}, chunk_size={chunk_size})."
        )

        contracts: List[Dict[str, Any]] = []

        if self.n_workers == 1:
            for row_dict in all_rows:
                sid = str(row_dict.get("source_id", "UNKNOWN"))
                try:
                    contract = self.run_single(row_dict)
                    contracts.append(contract)
                    logger.info(
                        f"[{sid}] OK | "
                        f"astro={contract['astrometry_status']} | "
                        f"binary={contract['binary_status']} | "
                        f"giant={contract['logical_flags']['is_giant']} | "
                        f"binary_cand={contract['logical_flags']['is_binary_candidate']}"
                    )
                except Exception as e:
                    logger.error(f"[{sid}] Orquestador falló: {e}")
        else:
            chunks = [
                all_rows[i: i + chunk_size]
                for i in range(0, len(all_rows), chunk_size)
            ]
            logger.info(f"Enviando {len(chunks)} chunks a {self.n_workers} workers.")
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(
                        HCPipelineOrchestrator._process_chunk,
                        chunk,
                        self.db_folder,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        chunk_contracts = future.result()
                        contracts.extend(chunk_contracts)
                        logger.info(
                            f"Chunk {chunk_idx + 1}/{len(chunks)} completado "
                            f"({len(chunk_contracts)} contratos)."
                        )
                    except Exception as e:
                        logger.error(f"Chunk {chunk_idx} lanzó excepción: {e}")

        logger.info(
            f"Pipeline completo. {len(contracts)} contratos de {len(all_rows)} estrellas."
        )
        return contracts

    # ------------------------------------------------------------------
    # Helpers de salida
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_run_output_path(base_dir: Path) -> Path:
        from datetime import datetime

        base_dir.mkdir(parents=True, exist_ok=True)
        max_n = 0
        for folder in base_dir.iterdir():
            if not folder.is_dir() or not folder.name.startswith("JSON-hc-prueba-"):
                continue
            try:
                n = int(folder.name.split("-")[3])
                max_n = max(max_n, n)
            except (IndexError, ValueError):
                continue

        fecha = datetime.now().strftime("%Y%m%d")
        return base_dir / f"JSON-hc-prueba-{max_n + 1}-{fecha}" / "hc_contracts.json"

    @staticmethod
    def save_results(
        contracts: List[Dict[str, Any]],
        base_dir: Path = Path("/home/cesar/Documentos/Tesis-cimat/Estancia/data/raw"),
    ) -> Path:
        """
        Guarda los contratos en una carpeta versionada dentro de base_dir.

        Convención de nombre: JSON-hc-prueba-{N+1}-{YYYYMMDD}/hc_contracts.json

        Returns:
            Path del archivo escrito.
        """
        output_path = HCPipelineOrchestrator._resolve_run_output_path(base_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(contracts, f, indent=2, default=_json_default)
        logger.info(f"Resultados escritos en {output_path} ({len(contracts)} contratos).")
        return output_path

    @staticmethod
    def summarize(contracts: List[Dict[str, Any]]) -> pd.DataFrame:
        """DataFrame plano de resumen para validación física rápida."""
        rows = []
        for c in contracts:
            pv = c.get("physical_vector", {})
            lf = c.get("logical_flags", {})
            bd = c.get("binary_diagnostics", {})
            rows.append({
                "source_id": c["source_id"],
                "astrometry_status": c["astrometry_status"],
                "teff_k": pv.get("teff_k"),
                "abs_mag": pv.get("abs_mag"),
                "metallicity": pv.get("metallicity"),
                "fe_h": pv.get("fe_h"),
                "alpha_fe": pv.get("alpha_fe"),
                "logg": pv.get("logg"),
                "v_tan": pv.get("v_tan"),
                "is_reliable_parallax": lf.get("is_reliable_parallax"),
                "is_giant": lf.get("is_giant"),
                "is_metal_poor": lf.get("is_metal_poor"),
                "is_binary_candidate": lf.get("is_binary_candidate"),
                "is_high_velocity": lf.get("is_high_velocity"),
                "has_emission": lf.get("has_emission"),
                "ruwe": bd.get("ruwe"),
            })
        return pd.DataFrame(rows)


# ===========================================================================
# UTILIDADES
# ===========================================================================


def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any) -> Optional[float]:
    if _is_nan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Tipo {type(obj)} no es serializable a JSON")


# ===========================================================================
# SMOKE TEST — solo filas sintéticas, no requiere archivos en disco
# ===========================================================================

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    SAMPLE_ROWS = [
        {
            "source_id": "5853498713190528",
            "ra": 120.5, "dec": -15.2,
            "parallax": 12.45, "parallax_error": 0.02,
            "pmra": -15.3, "pmdec": 4.2,
            "ruwe": 1.02,
            "phot_g_mean_mag": 10.2,
            "ebpminrp_gspphot": 0.05,
            "teff_gspphot": 5700.0,
            "logg_gspphot": 4.44,
            "mh_gspphot": 0.01,
            "alphafe_gspspec": 0.02,
            "fem_gspspec": -0.01,
            "ew_espels_halpha": 2.5,
            "ew_espels_halpha_flag": 0.0,
            "radial_velocity": 28.4,
            "radial_velocity_error": 1.2,
            "rv_nb_transits": 12.0,
            "nss_solution_type": None,
        },
        {
            "source_id": "5853498713199999",
            "ra": 121.1, "dec": -15.8,
            "parallax": -0.15, "parallax_error": 0.50,
            "pmra": 2.1, "pmdec": -1.1,
            "ruwe": 2.80,
            "phot_g_mean_mag": 18.5,
            "ebpminrp_gspphot": float("nan"),
            "teff_gspphot": 3200.0,
            "logg_gspphot": 4.80,
            "mh_gspphot": float("nan"),
            "alphafe_gspspec": float("nan"),
            "fem_gspspec": float("nan"),
            "ew_espels_halpha": float("nan"),
            "ew_espels_halpha_flag": float("nan"),
            "radial_velocity": float("nan"),
            "radial_velocity_error": float("nan"),
            "rv_nb_transits": float("nan"),
            "nss_solution_type": None,
        },
        {
            "source_id": "5853498713194444",
            "ra": 122.3, "dec": -14.9,
            "parallax": 4.20, "parallax_error": 0.05,
            "pmra": 12.0, "pmdec": 8.5,
            "ruwe": 1.10,
            "phot_g_mean_mag": 12.1,
            "ebpminrp_gspphot": 0.12,
            "teff_gspphot": 6100.0,
            "logg_gspphot": 4.20,
            "mh_gspphot": -0.50,
            "alphafe_gspspec": 0.15,
            "fem_gspspec": -0.45,
            "ew_espels_halpha": 1.2,
            "ew_espels_halpha_flag": 0.0,
            "radial_velocity": -45.0,
            "radial_velocity_error": 8.5,
            "rv_nb_transits": 25.0,
            "nss_solution_type": "Orbital",
        },
        {
            "source_id": "9999000000000001",
            "ra": 200.0, "dec": 45.0,
            "parallax": 2.10, "parallax_error": 0.10,
            "pmra": 80.0, "pmdec": -60.0,
            "ruwe": 1.05,
            "phot_g_mean_mag": 13.8,
            "ebpminrp_gspphot": 0.02,
            "teff_gspphot": 5200.0,
            "logg_gspphot": 4.30,
            "mh_gspphot": -1.85,
            "alphafe_gspspec": 0.38,
            "fem_gspspec": -1.80,
            "ew_espels_halpha": 1.8,
            "ew_espels_halpha_flag": 0.0,
            "radial_velocity": -312.0,
            "radial_velocity_error": 2.1,
            "rv_nb_transits": 18.0,
            "nss_solution_type": None,
        },
        {
            "source_id": "7777000000000001",
            "ra": 55.0, "dec": 20.0,
            "parallax": 3.50, "parallax_error": 0.08,
            "pmra": -5.0, "pmdec": 2.0,
            "ruwe": 1.15,
            "phot_g_mean_mag": 9.5,
            "ebpminrp_gspphot": 0.08,
            "teff_gspphot": 22000.0,
            "logg_gspphot": 3.8,
            "mh_gspphot": -0.10,
            "alphafe_gspspec": 0.0,
            "fem_gspspec": float("nan"),
            "ew_espels_halpha": -12.5,
            "ew_espels_halpha_flag": 0.0,
            "radial_velocity": 18.0,
            "radial_velocity_error": 15.0,
            "rv_nb_transits": 8.0,
            "nss_solution_type": None,
        },
    ]

    print("\n" + "=" * 72)
    print("  HC PIPELINE ORCHESTRATOR v2.0 — SMOKE TEST")
    print("=" * 72 + "\n")

    # Instancia mínima sin carpeta real (solo pruebas tabulares)
    dummy_orch = object.__new__(HCPipelineOrchestrator)
    dummy_orch._bprp_matrix = None
    dummy_orch._bprp_sampling = None
    dummy_orch._bprp_id_to_idx = {}
    dummy_orch.rvs_dir = Path("/nonexistent")
    dummy_orch.astrometry_agent = AstrometryAgent()
    dummy_orch.binary_agent = BinaryDetectorAgent()

    all_contracts = []
    for row_dict in SAMPLE_ROWS:
        sid = row_dict["source_id"]
        print(f"--- Procesando source_id: {sid} ---")
        astro_result = dummy_orch.astrometry_agent.process(row_dict)
        binary_result = dummy_orch.binary_agent.process(row_dict, source_id=sid)
        contract = dummy_orch._merge_to_sc_contract(
            sid, row_dict, astro_result, binary_result, {}
        )
        all_contracts.append(contract)
        print(json.dumps(contract, indent=2, default=_json_default))
        print()

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------

    def assert_check(condition: bool, label: str) -> None:
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")

    print("\n" + "=" * 72)
    print("  TESTS DE CARACTERÍSTICAS")
    print("=" * 72)

    print("\n  [Ventana Doppler]")
    agent_rv = LineAgent(rest_wavelength=656.281, radial_velocity=-312.0)
    shifted = agent_rv._doppler_shifted_center()
    expected = 656.281 + 656.281 * (-312.0 / LineAgent._C_KM_S)
    assert_check(abs(shifted - expected) < 1e-6,
                 f"Estrella halo: H-alpha centrada en {shifted:.4f} nm")
    assert_check(
        LineAgent(rest_wavelength=656.281, radial_velocity=None)._doppler_shifted_center() == 656.281,
        "Sin VR: ventana en longitud de onda en reposo"
    )
    assert_check(
        LineAgent(rest_wavelength=656.281, radial_velocity=float("nan"))._doppler_shifted_center() == 656.281,
        "VR=NaN: guardia NaN dispara correctamente"
    )

    print("\n  [Fallback Chebyshev]")
    wave_t = np.linspace(336, 1020, 343)
    flux_t = np.exp((wave_t - 336) / 300)
    flux_t[:5] *= 0.1
    flux_t[-5:] *= 0.1
    cr = ContinuumAgent().process(wave_t, flux_t, source_id="RUNGE_TEST")
    assert_check(cr["status"] == "success", "Espectro Runge-prone alcanza 'success'")
    if cr["status"] == "success":
        method = cr["measurements"].get("fit_method", "desconocido")
        assert_check(method in ("spline", "chebyshev", "flat_mean"),
                     f"fit_method en contrato: '{method}'")

    print("\n  [Regresión: flags físicos]")
    r = [c["logical_flags"] for c in all_contracts]
    assert_check(r[0]["is_reliable_parallax"], "Fuente 0: paralaje confiable")
    assert_check(not r[0]["is_giant"], "Fuente 0: no es gigante (enana Teff=5700)")
    assert_check(not r[0]["has_emission"], "Fuente 0: sin emisión (EW positivo)")
    assert_check(not r[1]["is_reliable_parallax"], "Fuente 1: no confiable (paralaje negativo)")
    assert_check(all_contracts[1]["binary_diagnostics"]["ruwe"] == 2.80, "Fuente 1: RUWE=2.80")
    assert_check(r[2]["is_binary_candidate"], "Fuente 2: candidata binaria (NSS Orbital)")
    assert_check(r[3]["is_metal_poor"], "Fuente 3: metal-pobre ([M/H]=-1.85)")
    assert_check(r[3]["is_high_velocity"], "Fuente 3: alta velocidad")
    assert_check(r[4]["has_emission"], "Fuente 4: emisión detectada (EW negativo)")
    assert_check(not r[4]["is_giant"], "Fuente 4: no gigante (estrella Be, Teff=22000)")

    print("\n  [Loaders de espectro — sin crash en datos ausentes]")
    assert_check(dummy_orch._load_bprp_spectrum("999") is None,
                 "_load_bprp_spectrum: None cuando matriz no cargada")
    assert_check(dummy_orch._load_rvs_spectrum("999") is None,
                 "_load_rvs_spectrum: None cuando archivo ausente")

    print("\nSmoke test completo.\n")
