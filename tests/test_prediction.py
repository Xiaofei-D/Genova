"""Tests for surrogate discovery and loading"""

from pathlib import Path

import numpy as np
import pytest
import torch

from genova.constants import CASE_CONDITION_NAMES, TIME_CONDITION_NAME
from genova.prediction import FlowMetadata, load_flow_surrogate, predict_spectra

CONDITION_NAMES = (TIME_CONDITION_NAME, *CASE_CONDITION_NAMES)


class FakeFlow:
    """Return one deterministic spectrum for every conditional row."""

    def sample(self, *, conditional: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Check the one-row-to-one-spectrum Glasflow sampling contract."""
        assert conditional.ndim == 2

        # Each conditional row must generate exactly one stochastic spectrum.
        assert num_samples == conditional.shape[0]

        # Use scaled time as the synthetic spectrum amplitude. Multiplying by
        # 100 mimics the spectra scale factor used during training.
        return conditional[:, :1].repeat(1, 3) * 100.0


def make_metadata() -> FlowMetadata:
    return FlowMetadata(
        path=Path("fake_flow_config.json"),
        n_inputs=3,
        n_conditional_inputs=len(CONDITION_NAMES),
        n_transforms=3,
        n_neurons=32,
        batch_norm_between_transforms=True,
        linear_transform=None,
        wavelengths=np.array([3000.0, 5000.0, 7000.0]),
        condition_names=CONDITION_NAMES,
        condition_min=np.zeros(len(CONDITION_NAMES), dtype=np.float32),
        condition_max=np.ones(len(CONDITION_NAMES), dtype=np.float32),
        scale_factor=100.0,
    )


# ---------------------------------------------------------------------
# Bundled surrogate predictions
# ---------------------------------------------------------------------


def test_bundled_surrogate_predicts_spectra(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    metadata, flow_model = load_flow_surrogate(device="cpu")

    assert metadata.condition_names == (TIME_CONDITION_NAME, *CASE_CONDITION_NAMES)
    assert metadata.wavelengths.size == metadata.n_inputs
    assert flow_model.training is False

    case_conditions = np.array([[1.35, 1.30, 400.0, 450.0, 0.5]], dtype=np.float32)
    time_rest = np.array([1.0, 2.0, 5.0], dtype=np.float32)

    spectra = predict_spectra(
        flow_model,
        metadata,
        case_conditions=case_conditions,
        time_rest=time_rest,
        n_samples=2,
        device="cpu",
    )

    assert spectra.shape == (1, 2, 3, metadata.wavelengths.size)
    assert np.all(np.isfinite(spectra))
    assert np.all(spectra >= 0.0)


# Prediction behaviour
def test_predict_spectra_output() -> None:
    metadata = make_metadata()

    # (n_cases=2, n_case_conditions=5)
    case_conditions = np.array(
        [
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.7, 0.6, 0.5, 0.4, 0.3],
        ]
    )
    # (n_times=3,)
    time_rest = np.array([0.1, 0.5, 0.9], dtype=np.float32)

    spectra = predict_spectra(
        FakeFlow(),
        metadata,
        case_conditions=case_conditions,
        time_rest=time_rest,
        n_samples=2,
        chunk_size=5,
    )

    # 2 cases x 3 times = 6 condition sets, x 2 samples = 12 conditional rows,
    # sampled in chunks of 5, 5 and 2.
    #
    # Output shape: (n_cases, n_samples, n_times, n_wavelengths)
    assert spectra.shape == (2, 2, 3, 3)

    expected = np.broadcast_to(time_rest[None, None, :, None], (2, 2, 3, 3))
    np.testing.assert_allclose(spectra, expected)


# Input contract
def test_predict_spectra_rejects_wrong_condition_count() -> None:
    metadata = make_metadata()

    with pytest.raises(ValueError, match="case_conditions must have shape"):
        predict_spectra(
            FakeFlow(),
            metadata,
            case_conditions=np.zeros((2, 3)),
            time_rest=np.array([0.1, 0.5], dtype=np.float32),
        )
