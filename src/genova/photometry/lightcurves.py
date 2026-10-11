"""
Generate light curve distribution.
"""

import warnings

import numpy as np


def build_common_time_grid(
    time_rest: np.ndarray,
    redshift: np.ndarray,
    *,
    time_obs_min: float | None = None,
    time_obs_max: float | None = None,
) -> np.ndarray:
    """Build an observer frame common time grid covered by every case.

    The shared observer frame time interval is the intersection of the time
    covered by all cases:

        shared start = max(time_rest[0] * (1 + redshift))
        shared end   = min(time_rest[-1] * (1 + redshift))

    A new observer frame time grid is generated with the same number of points and
    the same sampling style as the input rest frame grid.

    Parameters
    ----------
    time_rest:
        Original rest frame time grid, shape (n_times,).
    redshift:
        Redshift values for all cases, shape (n_cases,).
    time_obs_min: optional
        Requested minimum time in observer frame.
    time_obs_max: optional
        Requested maximum time in observer frame.

    Returns
    -------
    np.ndarray
        Shared time grid in observer frame, shape (n_times,).
    """
    time_rest_arr = np.asarray(time_rest, dtype=np.float64)
    redshift_arr = np.asarray(redshift, dtype=np.float64)

    if time_rest_arr.ndim != 1:
        raise ValueError("time_rest must be a 1D array")

    # Full time interval covered by every input case.
    shared_time_obs_min = time_rest_arr[0] * (1.0 + np.max(redshift_arr))
    shared_time_obs_max = time_rest_arr[-1] * (1.0 + np.min(redshift_arr))

    if shared_time_obs_min >= shared_time_obs_max:
        raise ValueError("The cases do not share a common observer frame time interval")

    # Requested observer frame time interval.
    requested_time_obs_min = (
        shared_time_obs_min if time_obs_min is None else float(time_obs_min)
    )
    requested_time_obs_max = (
        shared_time_obs_max if time_obs_max is None else float(time_obs_max)
    )

    # Clip requested limits to the shared interval. This avoids extrapolation.
    grid_time_obs_min = max(requested_time_obs_min, shared_time_obs_min)
    grid_time_obs_max = min(requested_time_obs_max, shared_time_obs_max)

    if grid_time_obs_min != requested_time_obs_min:
        warnings.warn(
            "time_obs_min is outside the time range covered by all "
            f"cases; input {requested_time_obs_min:.6g} is clipped to "
            f"{grid_time_obs_min:.6g}.",
            UserWarning,
            stacklevel=2,
        )

    if grid_time_obs_max != requested_time_obs_max:
        warnings.warn(
            "time_obs_max is outside the time range covered by all "
            f"cases; input {requested_time_obs_max:.6g} is clipped to "
            f"{grid_time_obs_max:.6g}.",
            UserWarning,
            stacklevel=2,
        )

    if grid_time_obs_min >= grid_time_obs_max:
        raise ValueError("time_obs_min must be smaller than time_obs_max")

    # Preserve the sampling style of the input grid.
    time_steps = np.diff(time_rest_arr)
    is_linear = np.allclose(
        time_steps,
        time_steps[0],
        rtol=1.0e-5,
        atol=1.0e-12,
    )  # ∣a−b∣ ≤ atol + rtol∣b∣.

    is_geometric = np.all(time_rest_arr > 0.0) and np.allclose(
        np.diff(np.log(time_rest_arr)),
        np.diff(np.log(time_rest_arr))[0],
        rtol=1.0e-5,
        atol=1.0e-12,
    )

    if is_geometric:
        return np.geomspace(grid_time_obs_min, grid_time_obs_max, time_rest_arr.size)

    if is_linear:
        return np.linspace(grid_time_obs_min, grid_time_obs_max, time_rest_arr.size)

    # Preserve the relative positions for a general non-uniform input grid.
    fractions = (time_rest_arr - time_rest_arr[0]) / (
        time_rest_arr[-1] - time_rest_arr[0]
    )

    return grid_time_obs_min + fractions * (grid_time_obs_max - grid_time_obs_min)


def interpolate_magnitudes(
    time_obs_per_case: np.ndarray,
    magnitudes_per_case: np.ndarray,
    time_obs_common: np.ndarray,
) -> np.ndarray:
    """Interpolate magnitude samples onto the shared time grid.

    Parameters
    ----------
    time_obs_per_case:
        Observer frame time grid for each case, shape (n_cases, n_times).
    magnitudes_per_case:
        Magnitude samples for each case,
        shape (n_cases, n_samples, n_times, n_bands).
    time_obs_common:
        Common time grid in observer frame, shape (n_common_times,).

    Returns
    -------
    np.ndarray
        Interpolated magnitude samples, shape
        (n_cases, n_samples, n_common_times, n_bands).

    Notes
    -----
    Linear interpolation is performed in magnitude space. Values outside the
    time coverage remain "NaN".
    """

    time_case_arr = np.asarray(time_obs_per_case, dtype=np.float64)
    mag_arr = np.asarray(magnitudes_per_case, dtype=np.float64)
    time_common_arr = np.asarray(time_obs_common, dtype=np.float64)

    n_cases, n_samples, n_times, n_bands = mag_arr.shape

    # np.interp expects grid to be strictly increasing.
    if np.any(np.diff(time_case_arr, axis=1) <= 0):
        raise ValueError("Each observer-frame time grid must be increasing")

    magnitudes_common = np.full(
        (n_cases, n_samples, time_common_arr.size, n_bands),
        np.nan,
        dtype=np.float64,
    )

    # Interpolate each magnitude light curve independently.
    # Missing values and unsupported time regions remain NaN.
    for case_index in range(n_cases):
        t_src = time_case_arr[case_index]

        for sample_index in range(n_samples):
            for band_index in range(n_bands):
                y_src = mag_arr[case_index, sample_index, :, band_index]
                finite = np.isfinite(y_src)

                if np.count_nonzero(finite) < 2:
                    continue

                t_min = np.min(t_src[finite])
                t_max = np.max(t_src[finite])
                inside = (time_common_arr >= t_min) & (time_common_arr <= t_max)

                if np.any(inside):
                    magnitudes_common[case_index, sample_index, inside, band_index] = (
                        np.interp(time_common_arr[inside], t_src[finite], y_src[finite])
                    )

    return magnitudes_common


def lightcurve_quantiles(
    magnitudes: np.ndarray,
    quantiles: np.ndarray | list[float],
    *,
    sample_axis: int = 0,
) -> np.ndarray:
    """Compute light curve quantiles from magnitude samples.

    Parameters
    ----------
    magnitudes:
        usually with shape (n_samples, n_times, n_bands).
    quantiles:
        Quantiles in the range [0, 1], for example [0.05, 0.5, 0.95].
    sample_axis:
        Axis corresponding to magnitude samples. By default is 0.

    Returns
    -------
    Quantile array
        Quantiles computed over the selected sample axis. An input of shape
        (n_samples, n_times, n_bands) returns an output of shape
        (n_quantiles, n_times, n_bands).
    """
    mag_arr = np.asarray(magnitudes)
    quantile_arr = np.asarray(quantiles)

    # Raise if the shape is wrong
    if mag_arr.ndim != 3:
        raise ValueError("magnitudes must have shape (n_samples, n_times, n_bands)")

    return np.quantile(mag_arr, quantile_arr, axis=sample_axis)


def ensemble_quantiles(
    magnitudes_common: np.ndarray,
    quantiles: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one combined light curve distribution over all cases.

    Parameters
    ----------
    magnitudes_common:
        Apparent magnitude samples on the shared time grid, with shape
        (n_cases, n_samples, n_common_times, n_bands).
    quantiles:
        Quantile levels to compute over the combined axis, with shape
        (n_quantiles,).

    Returns
    -------
    quantiles:
        Magnitude quantiles, shape
        (n_quantiles, n_common_times, n_bands).
    valid_draws:
        Number of magnitude draws available at each time and in each band,
        shape (n_common_times, n_bands).

    Notes
    -----
    "NaN" values outside the common coverage of individual light curves are ignored
    when computing the quantiles. Valid_draws are returned for inspection.
    """
    mag_arr = np.asarray(magnitudes_common, dtype=np.float64)

    n_cases, n_samples, n_times, n_bands = mag_arr.shape

    # Combine posterior cases and predicted samples of each case into one ensemble.
    draws = mag_arr.reshape(n_cases * n_samples, n_times, n_bands)

    # How many light curves contribute at each time and band.
    # This is important near time boundaries where some cases may be NaN.
    valid_draws = np.count_nonzero(np.isfinite(draws), axis=0)

    # Compute final quantiles over all available draws.
    quantile_values = np.nanquantile(draws, quantiles, axis=0)

    return quantile_values, valid_draws
