"""
These helpers apply frame and distance corrections only. Run them before calculating
synthetic photometry when observer frame magnitudes are needed.
"""

import numpy as np


def distance_modulus_from_mpc(
    luminosity_distance_mpc: float | np.ndarray,
) -> np.float64 | np.ndarray:
    """Compute distance modulus from luminosity distance in Mpc.

    Parameters
    ----------
    luminosity_distance_mpc:
        Luminosity distance in Mpc.

    Returns
    -------
    np.ndarray
        Distance modulus,
        mu = 5 log10(D_L / Mpc) + 25.
    """
    luminosity_distance_mpc = np.asarray(luminosity_distance_mpc, dtype=np.float64)

    return 5.0 * np.log10(luminosity_distance_mpc) + 25.0


def rest_to_observer_frame(
    *,
    time_rest: np.ndarray,
    wave_rest: np.ndarray,
    spectra_rest: np.ndarray,
    redshift: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert rest frame spectral time series to the observer frame.

    Parameters
    ----------
    time_rest:
        Rest frame time grid, shape (n_times,).

    wave_rest:
        Rest frame wavelength grid in Angstrom, shape (n_wavelengths,).

    spectra_rest:
        Rest frame spectral flux density, shape:
            (..., n_times, n_wavelengths)

    redshift:
        Scalar redshift (or a single-element array).

    Returns
    -------
    time_obs, wave_obs, spectra_obs with:
        time_obs = time_rest * (1 + z)
        wave_obs = wave_rest * (1 + z)
        spectra_obs = spectra_rest / (1 + z)

    Notes
    -----
    This function only applies frame conversion. It does not add distance
    modulus. Use distance_modulus_from_mpc for including distance.
    """
    time_rest = np.asarray(time_rest, dtype=np.float64)
    wave_rest = np.asarray(wave_rest, dtype=np.float64)
    spectra_rest = np.asarray(spectra_rest, dtype=np.float64)
    redshift = np.asarray(redshift, dtype=np.float64)

    time_obs = time_rest * (1.0 + float(redshift))
    wave_obs = wave_rest * (1.0 + float(redshift))
    spectra_obs = spectra_rest / (1.0 + float(redshift))

    return time_obs, wave_obs, spectra_obs
