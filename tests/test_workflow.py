"""Tests for event-level light-curve workflows and the command line interface."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from genova import __version__
from genova.cli import build_parser, main
from genova.constants import (
    BANDS,
    CASE_CONDITION_NAMES,
    DEVICE,
    N_SAMPLES,
    QUANTILES,
    REFERENCE_DISTANCE_MPC,
    TIME_CONDITION_NAME,
    time_rest_grid,
)
from genova.photometry import register_bandpass
from genova.prediction import FlowMetadata
from genova.workflows.event_lightcurves import (
    EventLightcurveModel,
    validate_posterior_table,
    validate_time_support,
)

CONDITION_NAMES = (TIME_CONDITION_NAME, *CASE_CONDITION_NAMES)

TIME_REST = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)

TEST_QUANTILES = np.array([0.16, 0.50, 0.84], dtype=float)

TEST_BAND = "genova_workflow_test_band"


# ----------------
# Test helpers
# ----------------


class FakeFlow:
    """Return deterministic positive spectra from the scaled time condition."""

    def sample(self, *, conditional: torch.Tensor, num_samples: int) -> torch.Tensor:
        assert conditional.ndim == 2
        assert num_samples == conditional.shape[0]

        return conditional[:, :1].repeat(1, 5) * 100.0


def make_fake_metadata() -> FlowMetadata:
    return FlowMetadata(
        path=Path("fake_flow_config.json"),
        n_inputs=5,
        n_conditional_inputs=len(CONDITION_NAMES),
        n_transforms=3,
        n_neurons=32,
        batch_norm_between_transforms=True,
        linear_transform=None,
        wavelengths=np.array([3000.0, 4000.0, 5000.0, 6000.0, 7500.0]),
        condition_names=CONDITION_NAMES,
        condition_min=np.array([0.0, 1.0, 1.0, 0.0, 0.0, -1.0], dtype=np.float32),
        condition_max=np.array([20.0, 2.0, 2.0, 1000.0, 1000.0, 1.0], dtype=np.float32),
        scale_factor=100.0,
    )


def make_posterior_table(n_rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "m1": np.linspace(1.30, 1.40, n_rows),
            "m2": np.linspace(1.20, 1.30, n_rows),
            "lambda1": np.linspace(380.0, 420.0, n_rows),
            "lambda2": np.linspace(420.0, 460.0, n_rows),
            "cos_theta_obs": np.linspace(0.2, 0.8, n_rows),
            "redshift": np.full(n_rows, 0.01),
            "luminosity_distance_Mpc": np.full(n_rows, 40.0),
        }
    )


def make_posterior_csv(tmp_path: Path, *, n_rows: int = 2) -> Path:
    event_dir = tmp_path / "GW170817"
    event_dir.mkdir()

    csv_path = event_dir / "GW170817_posteriors.csv"

    make_posterior_table(n_rows).to_csv(csv_path, index=False)

    return csv_path


def patch_surrogate_loader(monkeypatch) -> None:
    metadata = make_fake_metadata()
    flow_model = FakeFlow()

    def fake_load_flow_surrogate(*, surrogate_dir=None, device=None):
        return metadata, flow_model

    monkeypatch.setattr(
        "genova.workflows.event_lightcurves.load_flow_surrogate",
        fake_load_flow_surrogate,
    )


def register_test_bandpass() -> None:
    wave = np.linspace(4500.0, 6500.0, 201)
    transmission = np.ones_like(wave)

    register_bandpass(wave, transmission, TEST_BAND, force=True)


# ------------------------------------------------
# CSV → spectra → observer-frame light curve
# ------------------------------------------------


def test_from_csv_predicts_observer_lightcurve(monkeypatch, tmp_path: Path) -> None:
    patch_surrogate_loader(monkeypatch)
    register_test_bandpass()

    csv_path = make_posterior_csv(tmp_path, n_rows=2)

    model = EventLightcurveModel.from_csv(
        csv_path,
        device="cpu",
        time_rest=TIME_REST,
        bands=(TEST_BAND,),
        quantiles=TEST_QUANTILES,
        n_samples=2,
    )

    output_path = tmp_path / "prediction.npz"

    prediction = model.predict_lightcurve_quantiles(
        frame="obs",
        save_npz=True,
        output_path=output_path,
    )

    assert model.event == "GW170817"
    assert model.case_conditions.shape == (2, len(CASE_CONDITION_NAMES))

    assert prediction["frame"] == "obs"
    assert "time_obs" in prediction
    assert "time_rest_lc" not in prediction

    assert prediction["magnitude_quantiles"].shape[0] == TEST_QUANTILES.size
    assert prediction["magnitude_quantiles"].shape[-1] == 1

    assert (
        prediction["valid_draws"].shape == prediction["magnitude_quantiles"].shape[1:]
    )

    assert np.all(np.isfinite(prediction["magnitude_quantiles"]))
    assert np.all(prediction["valid_draws"] > 0)

    assert output_path.is_file()
    assert prediction["saved_npz"] == str(output_path)

    # Full magnitude samples are opt-in only.
    assert "magnitude_samples" not in prediction

    with np.load(output_path, allow_pickle=True) as saved:
        np.testing.assert_allclose(
            saved["magnitude_quantiles"], prediction["magnitude_quantiles"]
        )
        assert "magnitude_samples" not in saved


def test_return_magnitudes_matches_quantiles(monkeypatch, tmp_path: Path) -> None:
    patch_surrogate_loader(monkeypatch)
    register_test_bandpass()

    model = EventLightcurveModel.from_csv(
        make_posterior_csv(tmp_path, n_rows=2),
        device="cpu",
        time_rest=TIME_REST,
        bands=(TEST_BAND,),
        quantiles=TEST_QUANTILES,
        n_samples=2,
    )

    default = model.predict_lightcurve_quantiles(frame="obs")

    output_path = tmp_path / "prediction.npz"
    prediction = model.predict_lightcurve_quantiles(
        frame="obs",
        save_npz=True,
        output_path=output_path,
        return_magnitudes=True,
    )

    # Requesting the samples does not change the quantiles.
    np.testing.assert_array_equal(
        prediction["magnitude_quantiles"], default["magnitude_quantiles"]
    )

    # (n_cases, n_samples, n_times, n_bands)
    samples = prediction["magnitude_samples"]
    assert samples.shape == (2, 2, TIME_REST.size, 1)
    assert samples.dtype == np.float64

    # The quantiles are computed from exactly these samples.
    np.testing.assert_array_equal(
        np.nanquantile(samples.reshape(4, TIME_REST.size, 1), TEST_QUANTILES, axis=0),
        prediction["magnitude_quantiles"],
    )

    with np.load(output_path, allow_pickle=True) as saved:
        np.testing.assert_array_equal(saved["magnitude_samples"], samples)


# ---------------------
# Rest-frame meaning
# ---------------------


def test_rest_frame_matches_10pc_reference(monkeypatch) -> None:
    patch_surrogate_loader(monkeypatch)
    register_test_bandpass()

    table = make_posterior_table(n_rows=2)

    case_conditions = table.loc[:, list(CASE_CONDITION_NAMES)].to_numpy(
        dtype=np.float32
    )

    redshift = table["redshift"].to_numpy(dtype=float)

    luminosity_distance_mpc = table["luminosity_distance_Mpc"].to_numpy(dtype=float)

    rest_model = EventLightcurveModel.from_conditions(
        case_conditions=case_conditions,
        redshift=redshift,
        luminosity_distance_mpc=luminosity_distance_mpc,
        event="test_event",
        device="cpu",
        time_rest=TIME_REST,
        bands=(TEST_BAND,),
        quantiles=TEST_QUANTILES,
        n_samples=2,
    )

    reference_model = EventLightcurveModel.from_conditions(
        case_conditions=case_conditions,
        redshift=np.zeros(2),
        luminosity_distance_mpc=np.full(2, REFERENCE_DISTANCE_MPC),
        event="test_event",
        device="cpu",
        time_rest=TIME_REST,
        bands=(TEST_BAND,),
        quantiles=TEST_QUANTILES,
        n_samples=2,
    )

    rest_prediction = rest_model.predict_lightcurve_quantiles(frame="rest")
    reference_prediction = reference_model.predict_lightcurve_quantiles(frame="obs")

    np.testing.assert_allclose(
        rest_prediction["time_rest_lc"], reference_prediction["time_obs"]
    )
    np.testing.assert_allclose(
        rest_prediction["magnitude_quantiles"],
        reference_prediction["magnitude_quantiles"],
    )

    np.testing.assert_array_equal(
        rest_prediction["valid_draws"], reference_prediction["valid_draws"]
    )


# ----------------------------
# Posterior input contract
# ----------------------------


def test_posterior_table_rejects_missing_columns() -> None:
    table = make_posterior_table(n_rows=1).drop(columns="lambda1")

    with pytest.raises(ValueError, match="Missing required posterior columns"):
        validate_posterior_table(table)


# -------------------
# Training support
# -------------------


def test_time_support_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="outside the training support"):
        validate_time_support(
            np.array([1.0, 50.0], dtype=np.float32), make_fake_metadata()
        )


# ---------------------------
# Command line interface
# ---------------------------


def test_cli_parses_options() -> None:
    args = build_parser().parse_args(
        [
            "--posterior-csv", "posterior.csv",
            "--frame", "rest",
            "--output", "out.npz",
            "--magnitudes",
            "--event", "GW_test",
            "--n-samples", "4",
            "--bands", "ps1::g", "ps1::r",
            "--quantiles", "0.1", "0.9",
            "--time-min", "1",
            "--time-max", "5",
            "--n-times", "10",
            "--device", "cpu",
            "--surrogate-dir", "my_surrogate",
        ]
    )  # fmt: skip

    assert args.posterior_csv == Path("posterior.csv")
    assert args.frame == "rest"
    assert args.output == Path("out.npz")
    assert args.magnitudes is True
    assert args.event == "GW_test"
    assert args.n_samples == 4
    assert args.bands == ["ps1::g", "ps1::r"]
    assert args.quantiles == [0.1, 0.9]
    assert (args.time_min, args.time_max, args.n_times) == (1.0, 5.0, 10)
    assert args.device == "cpu"
    assert args.surrogate_dir == Path("my_surrogate")


def test_cli_defaults_match_python_api() -> None:
    args = build_parser().parse_args(["--posterior-csv", "posterior.csv"])
    from_csv = inspect.signature(EventLightcurveModel.from_csv).parameters
    predict = inspect.signature(EventLightcurveModel.predict_lightcurve_quantiles)

    assert args.magnitudes is False
    assert args.frame == predict.parameters["frame"].default
    assert args.n_samples == N_SAMPLES == from_csv["n_samples"].default
    assert tuple(args.bands) == BANDS == from_csv["bands"].default
    assert args.device == DEVICE == from_csv["device"].default
    assert tuple(args.quantiles) == QUANTILES
    np.testing.assert_array_equal(
        time_rest_grid(args.time_min, args.time_max, args.n_times), time_rest_grid()
    )


def test_cli_filters_hull_and_saves_magnitudes_on_request(
    monkeypatch, tmp_path: Path
) -> None:
    patch_surrogate_loader(monkeypatch)
    register_test_bandpass()

    # Two rows inside the bundled training domain, one far outside.
    csv_path = make_posterior_csv(tmp_path, n_rows=3)
    table = pd.read_csv(csv_path)
    table[["m1", "m2", "lambda1", "lambda2"]] = [
        [1.5026, 1.2357, 131.16, 1451.84],
        [1.5026, 1.2357, 131.16, 1451.84],
        [5.0, 4.0, 5.0, 5.0],
    ]
    table.to_csv(csv_path, index=False)

    output_dir = tmp_path / "out"
    common = [
        "--posterior-csv", str(csv_path),
        "--device", "cpu",
        "--bands", TEST_BAND,
        "--time-min", "1",
        "--time-max", "5",
        "--n-times", "4",
        "--n-samples", "2",
    ]  # fmt: skip

    default_path = output_dir / "default.npz"
    assert main([*common, "--output", str(default_path)]) == 0

    # The filtered posterior is saved next to the output automatically.
    filtered = pd.read_csv(output_dir / "GW170817_posteriors_in_convex_hull.csv")
    assert len(filtered) == 2

    with np.load(default_path, allow_pickle=True) as saved:
        assert str(saved["event"]) == "GW170817"
        assert "magnitude_quantiles" in saved
        assert "magnitude_samples" not in saved

    samples_path = output_dir / "samples.npz"
    assert main([*common, "--output", str(samples_path), "--magnitudes"]) == 0

    with np.load(samples_path, allow_pickle=True) as saved:
        assert "magnitude_quantiles" in saved
        # (n_kept_cases, n_samples, n_times, n_bands)
        assert saved["magnitude_samples"].shape == (2, 2, 4, 1)


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
