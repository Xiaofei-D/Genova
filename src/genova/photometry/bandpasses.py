import numpy as np
import sncosmo


def validate_bandpass_curve(
    band_wave: np.ndarray,
    transmission: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the shape of a photometric bandpass transmission curve.

    Convention:
        band_wave.shape == transmission.shape == (n_band_wave,)

    Parameters
    ----------
    band_wave:
        Bandpass wavelength grid in Angstrom.
    transmission:
        Bandpass transmission at each wavelength.

    Returns
    -------
    The same arrays as input, but converted to numpy arrays.
    """
    band_wave_arr = np.asarray(band_wave)
    transmission_arr = np.asarray(transmission)

    # Check array shapes
    if band_wave_arr.ndim != 1:
        raise ValueError(
            f"band_wave must be a 1D array. Got shape {band_wave_arr.shape}."
        )

    if transmission_arr.ndim != 1:
        raise ValueError(
            f"transmission must be a 1D array. Got shape {transmission_arr.shape}."
        )

    if band_wave_arr.shape != transmission_arr.shape:
        raise ValueError(
            "band_wave and transmission must have the same shape. "
            f"Got {band_wave_arr.shape} and {transmission_arr.shape}."
        )

    return band_wave_arr, transmission_arr


def register_bandpass(
    band_wave: np.ndarray,
    transmission: np.ndarray,
    name: str,
    *,
    force: bool = False,
) -> sncosmo.Bandpass:
    """Register a custom transmission curve as a named sncosmo bandpass.

    After registration, "name" can be passed in any "bands" argument, e.g. of
    :func:`genova.photometry.predict_case_lightcurve`.

    Parameters
    ----------
    band_wave:
        Bandpass wavelength grid in Angstrom.
    transmission:
        Bandpass transmission at each wavelength.
    name:
        Name the bandpass is registered under.
    force:
        If "True", overwrite an existing registration under "name".
        Otherwise raises if "name" is already registered.

    Returns
    -------
    sncosmo.Bandpass
        The registered bandpass object.
    """
    wave, transmission = validate_bandpass_curve(band_wave, transmission)

    # Drop non-finite values
    mask = np.isfinite(wave) & np.isfinite(transmission)
    wave_arr = wave[mask]
    transmission_arr = transmission[mask]

    # Keep wavelength in ascending order
    order = np.argsort(wave_arr)
    wave_arr = wave_arr[order]
    transmission_arr = transmission_arr[order]
    bandpass = sncosmo.Bandpass(
        wave_arr, transmission_arr, name=name, wave_unit="angstrom"
    )
    sncosmo.register(bandpass, name=name, force=force)

    return bandpass
