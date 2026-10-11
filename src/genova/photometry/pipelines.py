"""
These functions achieve two workflows:
1. one physical case with many predictive samples;
2. many cases combined into one light curve distribution.
"""

from collections.abc import Sequence

import numpy as np

from genova.constants import REFERENCE_DISTANCE_MPC, TIME_SPLINE_DEGREE
from genova.photometry.backends.sncosmo_backend import (
    ab_magnitudes_from_spectral_timeseries,
)
from genova.photometry.frames import rest_to_observer_frame
from genova.photometry.lightcurves import (
    build_common_time_grid,
    ensemble_quantiles,
    interpolate_magnitudes,
    lightcurve_quantiles,
)


def observer_ab_magnitudes_from_rest_spectra(
    *,
    time_rest: np.ndarray,
    wave_rest: np.ndarray,
    spectra_rest_at_10pc: np.ndarray,
    redshift: float,
    luminosity_distance_mpc: float,
    bands: Sequence[str],
    time_spline_degree: int = TIME_SPLINE_DEGREE,
) -> np.ndarray:
    """Compute observer frame apparent AB magnitudes from rest frame spectra.

    Parameters
    ----------
    time_rest:
        Spectral time grid in rest frame, shape (n_times,).
    wave_rest:
        Wavelength grid in Angstrom in rest frame, shape (n_wavelengths,).
    spectra_rest_at_10pc:
        Spectra (F_lambda) at 10 pc, shape (..., n_times, n_wavelengths).
        Leading axes are preserved. For example, one physical case with many
        predicted samples can be passed as (n_samples, n_times, n_wavelengths)
    redshift:
        Redshift applied to all spectra.
    luminosity_distance_mpc:
        Luminosity distance in Mpc applied to all spectra.
    bands:
        Registered sncosmo bandpass names.
    time_spline_degree:
        Passed to the sncosmo backend.

    Returns
    -------
    Apparent AB magnitudes with shape:
        (..., n_times, n_bands)

    Notes
    -----
    Apparent magnitude is only calculated from spectra after
    both distance and frame transformations.
    """
    # rest to observer frame
    time_obs, wave_obs, spectra_obs_at_10pc = rest_to_observer_frame(
        time_rest=time_rest,
        wave_rest=wave_rest,
        spectra_rest=spectra_rest_at_10pc,
        redshift=redshift,
    )

    # Luminosity distance scale, ~1/dL^2
    distance_scale = (REFERENCE_DISTANCE_MPC / float(luminosity_distance_mpc)) ** 2
    spectra_obs_at_dl = spectra_obs_at_10pc * distance_scale

    # Magnitude calculation
    mags_obs = ab_magnitudes_from_spectral_timeseries(
        spectra_time=time_obs,
        wave=wave_obs,
        spectra=spectra_obs_at_dl,
        bands=bands,
        time_spline_degree=time_spline_degree,
    )
    return mags_obs


def observer_ab_magnitudes_per_case_from_rest_spectra(
    *,
    time_rest: np.ndarray,
    wave_rest: np.ndarray,
    spectra_rest_at_10pc: np.ndarray,
    redshift: np.ndarray,
    luminosity_distance_mpc: np.ndarray,
    bands: Sequence[str],
    time_spline_degree: int = TIME_SPLINE_DEGREE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute apparent AB magnitudes for multiple physical cases.

    Each case has its own redshift and luminosity distance. The sample axis is
    explicit so that the output matches the flow prediction.

    Parameters
    ----------
    time_rest:
        Shared time grid in rest frame, (n_times,).
    wave_rest:
        Shared wavelength grid in rest frame, (n_wavelengths,).
    spectra_rest_at_10pc:
        Spectra at 10 pc in rest frame, (n_cases, n_samples, n_times, n_wavelengths).
    redshift:
        Redshift for each case, (n_cases,).
    luminosity_distance_mpc:
        Luminosity distance for each case in Mpc, (n_cases,).
    bands:
        Bandpass names.
    time_spline_degree:
        Time spline degree for interpolation passed to sncosmo.

    Returns
    -------
    time_obs_per_case:
        Time grid in observer frame for each case, (n_cases, n_times).
    magnitudes_per_case:
        Apparent magnitude samples, (n_cases, n_samples, n_times, n_bands).
    """
    time_rest = np.asarray(time_rest)
    wave_rest = np.asarray(wave_rest)
    spectra_rest_at_10pc = np.asarray(spectra_rest_at_10pc)
    redshift = np.asarray(redshift)
    luminosity_distance_mpc = np.asarray(luminosity_distance_mpc)

    # Check shape
    if spectra_rest_at_10pc.ndim != 4:
        raise ValueError(
            "spectra_rest_at_10pc must have four dimensions: "
            "(n_cases, n_samples, n_times, n_wavelengths)"
        )

    n_cases, n_samples, n_times, n_wavelengths = spectra_rest_at_10pc.shape
    n_bands = len(bands)

    if redshift.shape != (n_cases,):
        raise ValueError(f"redshift must have shape ({n_cases},), got {redshift.shape}")
    if luminosity_distance_mpc.shape != (n_cases,):
        raise ValueError(
            "luminosity_distance_mpc must have shape "
            f"({n_cases},), got {luminosity_distance_mpc.shape}"
        )

    time_obs_per_case = time_rest[None, :] * (1.0 + redshift[:, None])

    # Initialise array for magnitude samples
    magnitudes_per_case = np.empty((n_cases, n_samples, n_times, n_bands), dtype=float)

    for case_index in range(n_cases):
        magnitudes_per_case[case_index] = observer_ab_magnitudes_from_rest_spectra(
            time_rest=time_rest,
            wave_rest=wave_rest,
            spectra_rest_at_10pc=spectra_rest_at_10pc[case_index],
            redshift=float(redshift[case_index]),
            luminosity_distance_mpc=float(luminosity_distance_mpc[case_index]),
            bands=bands,
            time_spline_degree=time_spline_degree,
        )

    return time_obs_per_case, magnitudes_per_case


def predict_case_lightcurve(
    *,
    time_rest: np.ndarray,
    wave_rest: np.ndarray,
    spectra_rest_at_10pc: np.ndarray,
    redshift: float,
    luminosity_distance_mpc: float,
    bands: Sequence[str],
    quantiles: np.ndarray | list[float],
    time_spline_degree: int = TIME_SPLINE_DEGREE,
    return_magnitudes: bool = False,
) -> tuple[np.ndarray, ...]:
    """Predict light curve quantiles for one physical case.

    Parameters
    ----------
    time_rest:
        Time grid in rest frame, (n_times,).
    wave_rest:
        Wavelength grid in Angstrom in rest frame, (n_wavelengths,).
    spectra_rest_at_10pc:
        Spectral samples at 10 pc in rest frame,
        (n_samples, n_times, n_wavelengths).
    redshift:
        Redshift of the source.
    luminosity_distance_mpc:
        Luminosity distance in Mpc.
    bands:
        Registered sncosmo bandpass names.
    quantiles:
        Quantile levels to compute over the predictive samples, (n_quantiles,).
    time_spline_degree:
        Passed to the sncosmo backend.
    return_magnitudes:
        If True, also return the magnitude samples used for the quantiles.

    Returns
    -------
    time_obs:
        Time grid in observer frame, (n_times,).
    quantile_values:
        Apparent magnitude quantiles, (n_quantiles, n_times, n_bands).
    magnitude_samples:
        Only if "return_magnitudes" is True. Apparent magnitude samples,
        (n_samples, n_times, n_bands).

    Notes
    -----
    For absolute magnitudes, pass redshift=0 and
    luminosity_distance_mpc=REFERENCE_DISTANCE_MPC (10 pc).
    """
    # Convert each predicted spectrum into apparent magnitudes.
    time_obs = np.asarray(time_rest) * (1.0 + float(redshift))

    magnitude_samples = observer_ab_magnitudes_from_rest_spectra(
        time_rest=time_rest,
        wave_rest=wave_rest,
        spectra_rest_at_10pc=spectra_rest_at_10pc,
        redshift=redshift,
        luminosity_distance_mpc=luminosity_distance_mpc,
        bands=bands,
        time_spline_degree=time_spline_degree,
    )  # (n_samples, n_times, n_bands)

    # Compute quantiles
    quantile_values = lightcurve_quantiles(
        magnitude_samples,
        quantiles,
        sample_axis=0,
    )

    if return_magnitudes:
        return time_obs, quantile_values, magnitude_samples

    return time_obs, quantile_values


def predict_ensemble_lightcurve(
    *,
    time_rest: np.ndarray,
    wave_rest: np.ndarray,
    spectra_rest_at_10pc: np.ndarray,
    redshift: np.ndarray,
    luminosity_distance_mpc: np.ndarray,
    bands: Sequence[str],
    quantiles: np.ndarray | list[float],
    time_obs_common: np.ndarray | None = None,
    time_obs_min: float | None = None,
    time_obs_max: float | None = None,
    time_spline_degree: int = TIME_SPLINE_DEGREE,
    return_magnitudes: bool = False,
) -> tuple[np.ndarray, ...]:
    """Predict light curve distribution for multiple cases.

    Parameters
    ----------
    time_rest:
        Time grid in rest frame, (n_times,).
    wave_rest:
        Wavelength grid in Angstrom in rest frame, (n_wavelengths,).
    spectra_rest_at_10pc:
        Spectral samples at 10 pc in rest frame,
        (n_cases, n_samples, n_times, n_wavelengths).
    redshift:
        Redshift for each physical case, (n_cases,).
    luminosity_distance_mpc:
        Luminosity distance for each physical case in Mpc, (n_cases,).
    bands:
        Registered sncosmo bandpass names.
    quantiles:
        Quantile levels to compute over the combined case and sample
        axes, (n_quantiles,).
    time_obs_common:
        Optional shared observer-frame time grid, (n_common_times,).
        If provided, this grid is used directly.
    time_obs_min:
        Optional lower limit for the time grid in observer frame.
        Used only when "time_obs_common" is not provided.
    time_obs_max:
        Optional upper limit for the time grid in observer frame.
        Used only when "time_obs_common" is not provided.
    time_spline_degree:
        Passed to the sncosmo backend.
    return_magnitudes:
        If True, also return the magnitude samples used for the quantiles.

    Returns
    -------
    time_obs_grid:
        Shared time grid in observer frame, (n_common_times,).
    quantile_values:
        Combined apparent magnitude quantiles,
        (n_quantiles, n_common_times, n_bands).
    valid_draws:
        Number of finite magnitude draws contributing at each time and in each
        band, (n_common_times, n_bands).
    magnitude_samples:
        Only if "return_magnitudes" is True. Apparent magnitude samples on the
        shared time grid, (n_cases, n_samples, n_common_times, n_bands).
        "NaN" where a case does not cover a grid time.
    """

    # Convert every spectrum to apparent magnitudes.
    time_obs_per_case, magnitudes_per_case = (
        observer_ab_magnitudes_per_case_from_rest_spectra(
            time_rest=time_rest,
            wave_rest=wave_rest,
            spectra_rest_at_10pc=spectra_rest_at_10pc,
            redshift=redshift,
            luminosity_distance_mpc=luminosity_distance_mpc,
            bands=bands,
            time_spline_degree=time_spline_degree,
        )
    )

    # Use provided time grid when supplied. Otherwise, generate
    # a new grid over the interval covered by every case in observer frame.
    if time_obs_common is None:
        time_obs_grid = build_common_time_grid(
            time_rest,
            redshift,
            time_obs_min=time_obs_min,
            time_obs_max=time_obs_max,
        )
    else:
        time_obs_grid = np.asarray(time_obs_common, dtype=np.float64)

        if time_obs_grid.ndim != 1 or time_obs_grid.size < 2:
            raise ValueError(
                "time_obs_common must be a 1D array with at least two values"
            )

        if np.any(np.diff(time_obs_grid) <= 0.0):
            raise ValueError("time_obs_common must be strictly increasing")

    # Interpolate all light curves before combining their distributions.
    magnitudes_common = interpolate_magnitudes(
        time_obs_per_case,
        magnitudes_per_case,
        time_obs_grid,
    )

    # Combine all posterior cases and predicted samples into one ensemble.
    quantile_values, valid_draws = ensemble_quantiles(
        magnitudes_common,
        quantiles,
    )

    if return_magnitudes:
        return time_obs_grid, quantile_values, valid_draws, magnitudes_common

    return time_obs_grid, quantile_values, valid_draws
