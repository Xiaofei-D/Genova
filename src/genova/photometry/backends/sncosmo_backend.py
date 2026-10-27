"""AB synthetic photometry from spectral time series using sncosmo."""

from collections.abc import Sequence

import numpy as np
from sncosmo import TimeSeriesSource
from sncosmo.bandpasses import get_bandpass
from sncosmo.magsystems import get_magsystem
from sncosmo.utils import integration_grid

from genova.constants import HC_ERG_AA, TIME_SPLINE_DEGREE

# Wavelength step in Angstrom of the integration grid.
FLUX_SPACING_AA = 5.0

# Flux floor factor
SPECTRA_FLOOR = 1.0e-30


def _clean_spectra(spectra: np.ndarray) -> np.ndarray:
    """Replace zero and non-finite flux with a tiny non-zero floor."""
    values = np.asarray(spectra, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    valid_values = values[valid]

    # If all values are invalid, use a floor of 1e-30.
    mean_value = float(np.mean(valid_values)) if valid_values.size else 1.0
    if not np.isfinite(mean_value) or mean_value == 0.0:
        mean_value = 1.0

    replacement = SPECTRA_FLOOR * mean_value
    return np.where(valid, values, replacement)


def _ab_magnitudes_for_band(
    source: TimeSeriesSource,
    band_name: str,
    phase: np.ndarray,
) -> np.ndarray:
    """Integrate interpolated spectra through one bandpass in the AB system."""
    band = get_bandpass(band_name)

    if band.minwave() < source.minwave() or band.maxwave() > source.maxwave():
        raise ValueError(
            f"bandpass {band.name!r} [{band.minwave():.6g}, .., "
            f"{band.maxwave():.6g}] outside spectral range "
            f"[{source.minwave():.6g}, .., {source.maxwave():.6g}]"
        )

    integration_wave, dwave = integration_grid(
        band.minwave(), band.maxwave(), FLUX_SPACING_AA
    )

    transmission = band(integration_wave)

    # Interpolate first, then reflect negative spline values before integration.
    interpolated_flux = np.abs(source.flux(phase, integration_wave))

    photon_flux = (
        np.sum(integration_wave * transmission * interpolated_flux, axis=1)
        * dwave
        / HC_ERG_AA
    )

    zero_point_flux = get_magsystem("ab").zpbandflux(band)

    return -2.5 * np.log10(photon_flux / zero_point_flux)


def ab_magnitudes_from_spectral_timeseries(
    *,
    spectra_time: np.ndarray,
    wave: np.ndarray,
    spectra: np.ndarray,
    bands: Sequence[str],
    time_spline_degree: int = TIME_SPLINE_DEGREE,
) -> np.ndarray:
    """Convert same frame spectral time series into AB magnitudes.

    Parameters
    ----------
    spectra_time:
        Spectral time grid in days, shape (n_times,).
    wave:
        Wavelength grid in Angstrom, shape (n_wavelengths,).
    spectra:
        F_lambda spectra with shape (..., n_times, n_wavelengths).
    bands:
        Registered sncosmo bandpass names.
    time_spline_degree:
        Spline degree used in the time direction.

    Returns
    -------
    np.ndarray
        AB magnitudes with shape (..., n_times, n_bands). A time whose spectrum is
        exactly zero at every wavelength has zero flux: its magnitudes are +inf.
    """
    time_arr = np.asarray(spectra_time, dtype=float)
    wave_arr = np.asarray(wave, dtype=float)
    spectra_arr = np.asarray(spectra, dtype=float)
    band_names = tuple(bands)

    if time_arr.ndim != 1 or wave_arr.ndim != 1:
        raise ValueError("spectra_time and wavelengths must be one dimensional")

    if spectra_arr.shape[-2:] != (time_arr.size, wave_arr.size):
        raise ValueError("spectra final axes must have shape (n_times, n_wavelengths)")

    if not band_names:
        raise ValueError("bands must contain at least one bandpass name")

    # Ignore the zero flux spectra
    zero_rows = np.all(spectra_arr == 0.0, axis=-1).reshape(-1, time_arr.size)

    # sncosmo requires at least two time points for linear interpolation.
    # Duplicate a spectrum only to perform the interpolation.
    if time_arr.size == 1:
        delta_time = max(abs(float(time_arr[0])) * 1.0e-6, 1.0e-6)
        source_time = np.array([time_arr[0], time_arr[0] + delta_time])
        source_spectra = np.repeat(spectra_arr, 2, axis=-2)
        spline_degree = 1
    else:
        source_time = time_arr
        source_spectra = spectra_arr
        spline_degree = max(1, min(time_spline_degree, time_arr.size - 1))

    leading_shape = source_spectra.shape[:-2]
    flat_spectra = source_spectra.reshape(
        -1,
        source_time.size,
        wave_arr.size,
    )
    flat_magnitudes = np.empty(
        (flat_spectra.shape[0], time_arr.size, len(band_names)),
        dtype=float,
    )  # (n_cases, n_times, n_bands)

    for spectra_index, spectral_timeseries in enumerate(flat_spectra):
        source = TimeSeriesSource(
            phase=source_time,
            wave=wave_arr,
            flux=_clean_spectra(spectral_timeseries),
            time_spline_degree=spline_degree,
        )

        for band_index, band_name in enumerate(band_names):
            flat_magnitudes[spectra_index, :, band_index] = _ab_magnitudes_for_band(
                source,
                band_name,
                time_arr,
            )

    # Zero flux is infinitely faint.
    flat_magnitudes[zero_rows] = np.inf

    return flat_magnitudes.reshape(*leading_shape, time_arr.size, len(band_names))
