import numpy as np
import sncosmo
from sncosmo.bandpasses import get_bandpass

from genova.constants import HC_ERG_AA, TIME_SPLINE_DEGREE
from genova.photometry import register_bandpass
from genova.photometry.backends.sncosmo_backend import (
    _ab_magnitudes_for_band,
    _clean_spectra,
    ab_magnitudes_from_spectral_timeseries,
)
from genova.photometry.frames import rest_to_observer_frame
from genova.photometry.lightcurves import (
    build_common_time_grid,
    ensemble_quantiles,
    interpolate_magnitudes,
    lightcurve_quantiles,
)
from genova.photometry.pipelines import (
    observer_ab_magnitudes_from_rest_spectra,
    observer_ab_magnitudes_per_case_from_rest_spectra,
    predict_case_lightcurve,
    predict_ensemble_lightcurve,
)

# --------------------
# Helpers
# --------------------


def register_test_bandpass(*, name: str, wave_min: float, wave_max: float) -> str:
    band_wave = np.linspace(wave_min, wave_max, 301, dtype=np.float64)
    transmission = np.ones_like(band_wave)

    bandpass = sncosmo.Bandpass(band_wave, transmission, name=name)
    sncosmo.registry.register(bandpass, name=name, force=True)

    return name


def make_spectral_timeseries() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra_time = np.array([0.2, 1.0, 3.0, 10.0], dtype=np.float64)
    wave = np.linspace(2800.0, 25000.0, 800, dtype=np.float64)

    time_col = spectra_time[:, None]
    wave_row = wave[None, :]

    spectra = 1.0e-15 * (
        np.exp(-time_col / 4.0) * np.exp(-0.5 * ((wave_row - 7000.0) / 2500.0) ** 2)
        + 0.05
    )

    return spectra_time, wave, spectra


# --------------------
# Bandpass register
# --------------------


def test_register_bandpass_by_name() -> None:
    band_wave = np.array([4000.0, 5000.0, 6000.0])
    transmission = np.array([0.0, 1.0, 0.0])

    register_bandpass(band_wave, transmission, "genova_test_band")

    bandpass = get_bandpass("genova_test_band")
    np.testing.assert_allclose(bandpass.wave, band_wave)
    np.testing.assert_allclose(bandpass.trans, transmission)


# --------------------
# Frame conversion
# --------------------


def test_rest_to_observer_shifts_time_wave_flux() -> None:
    time_rest = np.array([0.2, 1.0, 3.0], dtype=np.float64)
    wave_rest = np.array([3000.0, 5000.0, 9000.0, 15000.0], dtype=np.float64)
    spectra_rest = np.ones((time_rest.size, wave_rest.size), dtype=np.float64)

    redshift = 0.01

    time_obs, wave_obs, spectra_obs = rest_to_observer_frame(
        time_rest=time_rest,
        wave_rest=wave_rest,
        spectra_rest=spectra_rest,
        redshift=redshift,
    )

    np.testing.assert_allclose(time_obs, time_rest * (1.0 + redshift))
    np.testing.assert_allclose(wave_obs, wave_rest * (1.0 + redshift))
    np.testing.assert_allclose(spectra_obs, spectra_rest / (1.0 + redshift))


# ------------------------
# Synthetic photometry
# ------------------------


def test_preserves_absolute_interpolated_flux_convention() -> None:
    spectra_time = np.array([1.0, 2.0], dtype=float)
    wave = np.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0])

    # All input fluxes are positive, but the cubic wavelength spline
    # undershoots below zero between the sparse grid points.
    base_spectrum = 1.0e-15 * np.array([1.0, 1.0e-6, 1.0, 1.0e-6, 1.0])
    spectra = np.stack([base_spectrum, base_spectrum])

    band_name = register_test_bandpass(
        name="genova_absolute_interpolated_flux",
        wave_min=7100.0,
        wave_max=7550.0,
    )

    actual = ab_magnitudes_from_spectral_timeseries(
        spectra_time=spectra_time,
        wave=wave,
        spectra=spectra,
        bands=[band_name],
        time_spline_degree=1,
    )

    source = sncosmo.TimeSeriesSource(
        phase=spectra_time,
        wave=wave,
        flux=spectra,
        time_spline_degree=1,
    )
    band = sncosmo.get_bandpass(band_name)

    integration_wave, dwave = sncosmo.utils.integration_grid(
        band.minwave(), band.maxwave(), 5.0
    )

    interpolated_flux = source.flux(spectra_time, integration_wave)

    # Make sure this test actually exercises the special convention.
    assert np.any(interpolated_flux < 0.0)

    transmission = band(integration_wave)

    photon_flux = (
        np.sum(integration_wave * transmission * np.abs(interpolated_flux), axis=1)
        * dwave
        / HC_ERG_AA
    )

    zero_point_flux = sncosmo.get_magsystem("ab").zpbandflux(band)
    expected = -2.5 * np.log10(photon_flux / zero_point_flux)

    np.testing.assert_allclose(
        actual[:, 0],
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_single_time_axis_supported() -> None:
    _, wave, spectra_timeseries = make_spectral_timeseries()

    band = register_test_bandpass(
        name="genova_sncosmo_backend_single_time",
        wave_min=5000.0,
        wave_max=9000.0,
    )

    spectra_time = np.array([1.0], dtype=np.float64)
    base_spectrum = spectra_timeseries[1]

    spectra_samples = np.stack(
        [0.8 * base_spectrum, 1.0 * base_spectrum, 1.2 * base_spectrum],
        axis=0,
    )

    # (n_samples, 1, n_wavelengths), not (n_samples, n_wavelengths).
    spectra_samples = spectra_samples[:, None, :]

    mags = ab_magnitudes_from_spectral_timeseries(
        spectra_time=spectra_time,
        wave=wave,
        spectra=spectra_samples,
        bands=[band],
    )

    assert mags.shape == (3, 1, 1)
    assert np.all(np.isfinite(mags))

    assert mags[2, 0, 0] < mags[1, 0, 0] < mags[0, 0, 0]


# ---------------------------
# Light curve combination
# ---------------------------


def test_common_time_grid_shared_interval() -> None:
    """Test the shared observer frame time grid for different redshifts."""
    time_rest = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 11.0])
    redshift = np.array([0.0, 0.1, 0.2])

    time_obs_common = build_common_time_grid(time_rest, redshift)

    expected_start = time_rest[0] * (1.0 + np.max(redshift))
    expected_end = time_rest[-1] * (1.0 + np.min(redshift))

    assert time_obs_common.shape == time_rest.shape
    np.testing.assert_allclose(time_obs_common[0], expected_start)
    np.testing.assert_allclose(time_obs_common[-1], expected_end)


def test_interpolate_magnitudes_no_extrapolation() -> None:
    """Interpolate supported times and report unsupported times as NaN."""
    time_obs_per_case = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

    magnitudes_per_case = np.array(
        [[[[10.0], [20.0], [30.0]]], [[[100.0], [110.0], [120.0]]]]
    )
    # Shape:
    # (n_cases=2, n_samples=1, n_times=3, n_bands=1)

    time_obs_common = np.array([1.0, 1.5, 2.5, 3.5])

    result = interpolate_magnitudes(
        time_obs_per_case,
        magnitudes_per_case,
        time_obs_common,
    )

    expected = np.array(
        [
            [[[10.0], [15.0], [25.0], [np.nan]]],
            [[[np.nan], [100.0], [110.0], [120.0]]],
        ]
    )

    assert result.shape == (2, 1, 4, 1)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_ensemble_quantiles_combine_cases_and_samples() -> None:
    """Compute one quantile distribution over all case and sample draws.
    NaN draw is invalid and is not involved in quantile
    """
    magnitudes_common = np.array(
        [
            [
                [[10.0], [20.0]],
                [[20.0], [30.0]],
            ],
            [
                [[30.0], [40.0]],
                [[np.nan], [50.0]],
            ],
        ]
    )
    # Shape:
    # (n_cases=2, n_samples=2, n_times=2, n_bands=1)
    #
    # Draw 0: [10, 20, 30, NaN]
    # Draw 1: [20, 30, 40, 50]

    result, valid_draws = ensemble_quantiles(
        magnitudes_common,
        quantiles=[0.0, 0.5, 1.0],
    )

    expected = np.array(
        [
            [[10.0], [20.0]],
            [[20.0], [35.0]],
            [[30.0], [50.0]],
        ]
    )

    assert result.shape == (3, 2, 1)
    assert valid_draws.shape == (2, 1)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(
        valid_draws,
        np.array([[3], [4]]),
    )


# -------------------------
# Light-curve pipelines
# -------------------------


def test_predict_case_lightcurve() -> None:
    """Compute magnitude quantiles for one case with multiple samples."""
    # Register a local bandpass
    test_band = sncosmo.Bandpass(
        wave=np.array([4000.0, 5000.0, 6000.0]),
        trans=np.array([0.0, 1.0, 0.0]),
        name="test_case_band",
    )
    sncosmo.register(test_band, force=True)

    time_rest = np.array([1.0, 2.0])
    wave_rest = np.linspace(3000.0, 7000.0, 50)

    spectra_rest_at_10pc = np.full((3, 2, 50), 1.0e-15)
    spectra_rest_at_10pc[1] *= 2.0
    spectra_rest_at_10pc[2] *= 3.0
    # Shape:
    # (n_samples=3, n_times=2, n_wavelengths=50)

    quantiles = [0.0, 0.5, 1.0]

    # the workflow
    expected_magnitudes = observer_ab_magnitudes_from_rest_spectra(
        time_rest=time_rest,
        wave_rest=wave_rest,
        spectra_rest_at_10pc=spectra_rest_at_10pc,
        redshift=0.0,
        luminosity_distance_mpc=1.0e-5,
        bands=["test_case_band"],
        time_spline_degree=1,
    )

    expected_quantiles = lightcurve_quantiles(
        expected_magnitudes,
        quantiles,
        sample_axis=0,
    )

    result_time, result_quantiles = predict_case_lightcurve(
        time_rest=time_rest,
        wave_rest=wave_rest,
        spectra_rest_at_10pc=spectra_rest_at_10pc,
        redshift=0.0,
        luminosity_distance_mpc=1.0e-5,
        bands=["test_case_band"],
        quantiles=quantiles,
        time_spline_degree=1,
    )

    np.testing.assert_allclose(result_time, time_rest)
    np.testing.assert_allclose(result_quantiles, expected_quantiles)
    assert result_quantiles.shape == (3, 2, 1)

    # Optional magnitude samples
    _, quantiles_with_samples, magnitude_samples = predict_case_lightcurve(
        time_rest=time_rest,
        wave_rest=wave_rest,
        spectra_rest_at_10pc=spectra_rest_at_10pc,
        redshift=0.0,
        luminosity_distance_mpc=1.0e-5,
        bands=["test_case_band"],
        quantiles=quantiles,
        time_spline_degree=1,
        return_magnitudes=True,
    )

    assert magnitude_samples.shape == (3, 2, 1)
    np.testing.assert_array_equal(magnitude_samples, expected_magnitudes)
    np.testing.assert_array_equal(quantiles_with_samples, result_quantiles)


def test_predict_ensemble_lightcurve() -> None:
    # Register a local bandpass
    test_band = sncosmo.Bandpass(
        wave=np.array([4000.0, 5000.0, 6000.0]),
        trans=np.array([0.0, 1.0, 0.0]),
        name="test_ensemble_band",
    )
    sncosmo.register(test_band, force=True)

    time_rest = np.array([1.0, 2.0, 3.0])
    wave_rest = np.linspace(3000.0, 7000.0, 50)

    spectra_rest_at_10pc = np.full((2, 2, 3, 50), 1.0e-15)
    spectra_rest_at_10pc[:, 1] *= 2.0
    # Shape:
    # (n_cases=2, n_samples=2, n_times=3, n_wavelengths=50)

    redshift = np.array([0.0, 0.2])
    luminosity_distance_mpc = np.array([40.0, 50.0])
    bands = ["test_ensemble_band"]
    quantiles = [0.05, 0.5, 0.95]

    time_obs_min = 1.5
    time_obs_max = 2.8

    # the workflow
    time_obs_per_case, magnitudes_per_case = (
        observer_ab_magnitudes_per_case_from_rest_spectra(
            time_rest=time_rest,
            wave_rest=wave_rest,
            spectra_rest_at_10pc=spectra_rest_at_10pc,
            redshift=redshift,
            luminosity_distance_mpc=luminosity_distance_mpc,
            bands=bands,
            time_spline_degree=1,
        )
    )

    expected_time = build_common_time_grid(
        time_rest,
        redshift,
        time_obs_min=time_obs_min,
        time_obs_max=time_obs_max,
    )

    magnitudes_common = interpolate_magnitudes(
        time_obs_per_case,
        magnitudes_per_case,
        expected_time,
    )

    expected_quantiles, expected_valid_draws = ensemble_quantiles(
        magnitudes_common,
        quantiles,
    )

    # The high-level pipeline should reproduce the same calculation.
    result_time, result_quantiles, result_valid_draws = predict_ensemble_lightcurve(
        time_rest=time_rest,
        wave_rest=wave_rest,
        spectra_rest_at_10pc=spectra_rest_at_10pc,
        redshift=redshift,
        luminosity_distance_mpc=luminosity_distance_mpc,
        bands=bands,
        quantiles=quantiles,
        time_obs_min=time_obs_min,
        time_obs_max=time_obs_max,
        time_spline_degree=1,
    )

    np.testing.assert_allclose(result_time, expected_time)
    np.testing.assert_allclose(result_quantiles, expected_quantiles)
    np.testing.assert_array_equal(result_valid_draws, expected_valid_draws)

    # Optional magnitude samples
    *_, magnitude_samples = predict_ensemble_lightcurve(
        time_rest=time_rest,
        wave_rest=wave_rest,
        spectra_rest_at_10pc=spectra_rest_at_10pc,
        redshift=redshift,
        luminosity_distance_mpc=luminosity_distance_mpc,
        bands=bands,
        quantiles=quantiles,
        time_obs_min=time_obs_min,
        time_obs_max=time_obs_max,
        time_spline_degree=1,
        return_magnitudes=True,
    )

    # (n_cases, n_samples, n_times, n_bands)
    assert magnitude_samples.shape == (2, 2, 3, 1)
    np.testing.assert_array_equal(magnitude_samples, magnitudes_common)


# --------------------
# Exact zero flux
# --------------------


def general_path_magnitudes(spectra_time, wave, spectra, bands):
    """Magnitudes from the backend's cleaning, time spline and integration steps."""
    source = sncosmo.TimeSeriesSource(
        phase=spectra_time,
        wave=wave,
        flux=_clean_spectra(spectra),
        time_spline_degree=TIME_SPLINE_DEGREE,
    )
    return np.stack(
        [_ab_magnitudes_for_band(source, band, spectra_time) for band in bands], axis=-1
    )


def test_all_zero_time_row_gives_infinite_magnitudes() -> None:
    spectra_time, wave, spectra = make_spectral_timeseries()
    spectra[1] = 0.0
    bands = (
        register_test_bandpass(
            name="zero_flux_test_band",
            wave_min=4000.0,
            wave_max=9000.0,
        ),
    )

    magnitudes = ab_magnitudes_from_spectral_timeseries(
        spectra_time=spectra_time, wave=wave, spectra=spectra, bands=bands
    )

    assert np.all(np.isposinf(magnitudes[1]))
    # Other rows are unchanged; the zero row stays in the time spline as before.
    expected = general_path_magnitudes(spectra_time, wave, spectra, bands)
    np.testing.assert_array_equal(magnitudes[[0, 2, 3]], expected[[0, 2, 3]])


def test_near_zero_spectra_keep_finite_magnitudes() -> None:
    spectra_time, wave, spectra = make_spectral_timeseries()
    spectra[1] = 1.0e-300
    spectra[2] = 0.0
    spectra[2, 400] = 1.0e-20  # one non-zero wavelength
    bands = (
        register_test_bandpass(
            name="near_zero_flux_test_band",
            wave_min=4000.0,
            wave_max=9000.0,
        ),
    )

    magnitudes = ab_magnitudes_from_spectral_timeseries(
        spectra_time=spectra_time, wave=wave, spectra=spectra, bands=bands
    )

    np.testing.assert_array_equal(
        magnitudes, general_path_magnitudes(spectra_time, wave, spectra, bands)
    )
    assert np.all(np.isfinite(magnitudes))
