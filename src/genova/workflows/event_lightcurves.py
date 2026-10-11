"""Propagate multiple physical cases into event-level light curve quantiles.

Cases usually come from gravitational wave posteriors, one row per posterior
sample, but any table of physical conditions works.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from genova.constants import (
    BANDS,
    CASE_CONDITION_NAMES,
    CHUNK_SIZE,
    DEVICE,
    LIGHTCURVE_NPZ_TEMPLATE,
    N_SAMPLES,
    OUTPUT_DIR,
    POSTERIOR_REQUIRED_COLUMNS,
    QUANTILES,
    REFERENCE_DISTANCE_MPC,
    TIME_CONDITION_NAME,
    time_rest_grid,
)
from genova.photometry.pipelines import predict_ensemble_lightcurve
from genova.prediction import FlowMetadata, load_flow_surrogate
from genova.prediction import predict_spectra as predict_flow_spectra

# Both frames are supported
_FRAME_TIME_KEYS = {
    "obs": "time_obs",
    "rest": "time_rest_lc",
}


def _check_frame(frame: str) -> None:
    """Raise if "frame" is not a supported frame name."""
    if frame not in _FRAME_TIME_KEYS:
        raise ValueError(
            f"frame must be one of {sorted(_FRAME_TIME_KEYS)}, got {frame!r}."
        )


def default_output_path(
    event: str,
    *,
    frame: str = "obs",
    output_dir: str | Path = OUTPUT_DIR,
) -> Path:
    """Return the output path for one event. Observer frame by default."""
    _check_frame(frame)

    return Path(output_dir) / event / LIGHTCURVE_NPZ_TEMPLATE.format(frame=frame)


def validate_posterior_table(posterior: pd.DataFrame) -> None:
    """Check that a posterior table includes the columns and values that are needed."""
    missing_columns = sorted(set(POSTERIOR_REQUIRED_COLUMNS) - set(posterior.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required posterior columns: {missing_columns}. "
            f"Expected {list(POSTERIOR_REQUIRED_COLUMNS)}."
        )

    # Check Nans and Infs
    values = posterior.loc[:, list(POSTERIOR_REQUIRED_COLUMNS)].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Posteriors contain non-finite (NaN or inf) values.")

    # Posteriors may contain negative cos_theta_obs values,
    # which are not applicable for the flow model.
    cos_theta_obs = posterior["cos_theta_obs"].to_numpy(dtype=float)
    if np.any((cos_theta_obs < 0.0) | (cos_theta_obs > 1.0)):
        raise ValueError("cos_theta_obs must be in [0, 1].")


def validate_time_support(time_rest: np.ndarray, metadata: FlowMetadata) -> None:
    """Check if the requested time grid lies outside training support."""
    time_index = metadata.condition_names.index(TIME_CONDITION_NAME)
    metadata_tmin = metadata.condition_min[time_index]
    metadata_tmax = metadata.condition_max[time_index]

    requested_tmin = np.min(time_rest)
    requested_tmax = np.max(time_rest)

    if requested_tmin < metadata_tmin or requested_tmax > metadata_tmax:
        raise ValueError(
            "Time grid lies outside the training support: "
            f"requested [{requested_tmin:.3f}, {requested_tmax:.3f}] days, "
            f"allowed [{metadata_tmin:.3f}, {metadata_tmax:.3f}] days."
        )


def _save_npz(output_path: str | Path, prediction: dict[str, Any]) -> Path:
    """Save light curve predictions as a compressed NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_path, **prediction)

    return output_path


@dataclass
class EventLightcurveModel:
    """Spectra and light curve prediction for one event.

    Attributes
    ----------
    event:
        Event name, used for output file naming.
    posterior_csv:
        Posterior table, or "None" when conditions were passed directly
        as arrays.
    case_conditions:
        Physical conditions excluding time, (n_cases, 5), in the order of
        m1, m2, lambda1, lambda2, cos_theta_obs.
    redshift:
        Redshift per case, (n_cases,).
    luminosity_distance_mpc:
        Luminosity distance per case in Mpc, (n_cases,).
    time_rest:
        Time grid in days in rest frame, (n_times,).
    spectra_rest_at_10pc:
        Cached spectra in the rest frame at 10 pc,
        (n_cases, n_samples, n_times, n_wavelengths).
    """

    event: str
    metadata: FlowMetadata
    flow_model: Any
    case_conditions: np.ndarray
    redshift: np.ndarray
    luminosity_distance_mpc: np.ndarray
    posterior_csv: Path | None = None
    posterior: pd.DataFrame | None = None
    device: str | torch.device | None = DEVICE
    time_rest: np.ndarray = field(default_factory=time_rest_grid)
    bands: tuple[str, ...] = BANDS
    quantiles: np.ndarray = field(
        default_factory=lambda: np.asarray(QUANTILES, dtype=float)
    )
    n_samples: int = N_SAMPLES
    chunk_size: int = CHUNK_SIZE
    spectra_rest_at_10pc: np.ndarray | None = None

    @classmethod
    def from_conditions(
        cls,
        case_conditions: np.ndarray,
        redshift: np.ndarray,
        luminosity_distance_mpc: np.ndarray,
        *,
        event: str = "event",
        surrogate_dir: str | Path | None = None,
        device: str | torch.device | None = DEVICE,
        time_rest: np.ndarray | None = None,
        bands: tuple[str, ...] = BANDS,
        quantiles: np.ndarray | None = None,
        n_samples: int = N_SAMPLES,
        chunk_size: int = CHUNK_SIZE,
        posterior_csv: str | Path | None = None,
        posterior: pd.DataFrame | None = None,
    ) -> "EventLightcurveModel":
        """Create a model from arrays of physical conditions.

        Parameters
        ----------
        case_conditions:
            Shape (n_cases, 5), column order
            m1, m2, lambda1, lambda2, cos_theta_obs.
        redshift:
            Shape (n_cases,).
        luminosity_distance_mpc:
            Shape (n_cases,), in Mpc.
        event:
            Event name, used for output file naming.
        surrogate_dir:
            Directory of a surrogate bundle. Defaults to the bundled model.
        device:
            Torch device; see :func:`genova.prediction.resolve_device`.
        time_rest:
            Rest frame times in days. Defaults to
            :func:`genova.constants.time_rest_grid`. Must lie inside the
            surrogate's training time range.
        bands:
            Registered sncosmo bandpass names.
        quantiles:
            Quantile levels in [0, 1]. Defaults to ``QUANTILES``.
        n_samples:
            Surrogate draws per case.
        chunk_size:
            Maximum number of conditioning rows per flow call.
        posterior_csv, posterior:
            Optional source table, stored for reference only.
        """
        case_conditions = np.asarray(case_conditions, dtype=np.float32)
        redshift = np.asarray(redshift, dtype=float)
        luminosity_distance_mpc = np.asarray(luminosity_distance_mpc, dtype=float)

        # Validate shapes and column order, this is important
        if case_conditions.ndim != 2 or case_conditions.shape[1] != len(
            CASE_CONDITION_NAMES
        ):
            raise ValueError(
                "case_conditions must have shape "
                f"(n_cases, {len(CASE_CONDITION_NAMES)}) with column order "
                f"{CASE_CONDITION_NAMES}. Got shape {case_conditions.shape}."
            )

        n_cases = case_conditions.shape[0]
        for name, values in (
            ("redshift", redshift),
            ("luminosity_distance_mpc", luminosity_distance_mpc),
        ):
            if values.shape != (n_cases,):
                raise ValueError(
                    f"{name} must have shape ({n_cases},) to match "
                    f"case_conditions, got {values.shape}."
                )

        metadata, flow_model = load_flow_surrogate(
            surrogate_dir=surrogate_dir,
            device=device,
        )

        if time_rest is None:
            time_rest = time_rest_grid()
        else:
            time_rest = np.asarray(time_rest, dtype=np.float32)

        if quantiles is None:
            quantiles = np.asarray(QUANTILES, dtype=float)
        else:
            quantiles = np.asarray(quantiles, dtype=float)

        validate_time_support(time_rest, metadata)

        return cls(
            event=event,
            metadata=metadata,
            flow_model=flow_model,
            case_conditions=case_conditions,
            redshift=redshift,
            luminosity_distance_mpc=luminosity_distance_mpc,
            posterior_csv=Path(posterior_csv) if posterior_csv is not None else None,
            posterior=posterior,
            device=device,
            time_rest=time_rest,
            bands=bands,
            quantiles=quantiles,
            n_samples=n_samples,
            chunk_size=chunk_size,
        )

    @classmethod
    def from_csv(
        cls,
        posterior_csv: str | Path,
        *,
        event: str | None = None,
        surrogate_dir: str | Path | None = None,
        device: str | torch.device | None = DEVICE,
        time_rest: np.ndarray | None = None,
        bands: tuple[str, ...] = BANDS,
        quantiles: np.ndarray | None = None,
        n_samples: int = N_SAMPLES,
        chunk_size: int = CHUNK_SIZE,
    ) -> "EventLightcurveModel":
        """Create a model from one prepared posterior CSV.

        The CSV must contain the columns
        m1, m2, lambda1, lambda2, cos_theta_obs, redshift, luminosity_distance_Mpc.

        Parameters
        ----------
        posterior_csv:
            Path to the prepared posterior CSV file.
        event:
            Optional event name, defaults to the dir that holds the csv file.
        surrogate_dir:
            Optional directory containing the trained flow model and its
            configuration.
        device:
            PyTorch device on which to run the flow model, e.g. "cpu",
            "cuda:0", or a torch.device instance.
        time_rest:
            Time grid in days in the rest frame.
        bands:
            Registered sncosmo bandpass names.
        quantiles:
            Quantile levels to compute, with shape (n_quantiles,).
            If not provided, the default quantiles are used.
        n_samples:
            Number of predicted samples for each physical case.
        chunk_size:
            Maximum number of conditioning rows per flow call.

        Returns
        -------
        EventLightcurveModel
            An event-level model.
        """
        posterior_csv = Path(posterior_csv)

        if event is None:
            event = posterior_csv.parent.name  # Event name

        posterior = pd.read_csv(posterior_csv)
        validate_posterior_table(posterior)

        return cls.from_conditions(
            case_conditions=posterior.loc[:, list(CASE_CONDITION_NAMES)].to_numpy(
                dtype=np.float32
            ),
            redshift=posterior["redshift"].to_numpy(dtype=float),
            luminosity_distance_mpc=posterior["luminosity_distance_Mpc"].to_numpy(
                dtype=float
            ),
            event=event,
            surrogate_dir=surrogate_dir,
            device=device,
            time_rest=time_rest,
            bands=bands,
            quantiles=quantiles,
            n_samples=n_samples,
            chunk_size=chunk_size,
            posterior_csv=posterior_csv,
            posterior=posterior,
        )

    def predict_spectra(self, *, force: bool = False) -> np.ndarray:
        """Predict rest frame spectra at 10 pc from scratch or return cached spectra.

        Returns
        -------
        np.ndarray
            Shape (n_cases, n_samples, n_times, n_wavelengths).

        Notes
        -----
        "force=True" will recompute the spectra even if they are cached.
        """
        if self.spectra_rest_at_10pc is not None and not force:
            return self.spectra_rest_at_10pc

        spectra_rest_at_10pc = predict_flow_spectra(
            flow_model=self.flow_model,
            metadata=self.metadata,
            case_conditions=self.case_conditions,
            time_rest=self.time_rest,
            n_samples=self.n_samples,
            chunk_size=self.chunk_size,
            device=self.device,
        )

        self.spectra_rest_at_10pc = spectra_rest_at_10pc

        return spectra_rest_at_10pc

    def predict_lightcurve_quantiles(
        self,
        *,
        frame: str = "obs",
        save_npz: bool = False,
        output_path: str | Path | None = None,
        force_spectra: bool = False,
        return_magnitudes: bool = False,
    ) -> dict[str, Any]:
        """Predict light curve quantiles for this event.

        Parameters
        ----------
        frame:
            "obs" for observer frame, "rest" for rest frame. Defaults to "obs".
        save_npz:
            If True, save the returned dictionary to a compressed NPZ file.
        output_path:
            NPZ output path. Defaults to
            ``default_output_path(event, frame=frame)``.
        force_spectra:
            Rerun the spectra prediction, ignore the cached spectra.
        return_magnitudes:
            If True, also return (and save) the full magnitude samples that the
            quantiles are computed from. Off by default because the array can
            be large.

        Returns
        -------
        dict
            Light curve quantiles and the associated metadata, including:

            - "time_obs" (frame="obs") or "time_rest_lc" (frame="rest"):
              light curve time grid, (n_times,).
            - "magnitude_quantiles": (n_quantiles, n_times, n_bands).
            - "valid_draws": finite draws per time and band, (n_times, n_bands).
            - "magnitude_samples": only if return_magnitudes=True,
              (n_cases, n_samples, n_times, n_bands), "NaN" where a case does
              not cover a grid time.
            - "saved_npz": only if save_npz=True, path of the saved file.
        """
        _check_frame(frame)

        spectra_rest_at_10pc = self.predict_spectra(force=force_spectra)
        n_cases = self.case_conditions.shape[0]

        if frame == "obs":
            redshift = self.redshift
            luminosity_distance_mpc = self.luminosity_distance_mpc
        else:
            # redshift = 0.0 for absolute magnitudes
            redshift = np.zeros(n_cases, dtype=float)
            luminosity_distance_mpc = np.full(
                n_cases, REFERENCE_DISTANCE_MPC, dtype=float
            )

        ensemble = predict_ensemble_lightcurve(
            time_rest=self.time_rest,
            wave_rest=self.metadata.wavelengths,
            spectra_rest_at_10pc=spectra_rest_at_10pc,
            redshift=redshift,
            luminosity_distance_mpc=luminosity_distance_mpc,
            bands=self.bands,
            quantiles=self.quantiles,
            return_magnitudes=return_magnitudes,
        )
        time_grid, magnitude_quantiles, valid_draws = ensemble[:3]

        prediction: dict[str, Any] = {
            "event": self.event,
            "posterior_csv": str(self.posterior_csv),
            "frame": frame,
            "bands": np.asarray(self.bands),
            "quantiles": self.quantiles,
            "time_rest": self.time_rest,
            "n_posterior_cases": np.asarray(n_cases),
            "n_samples_per_case": np.asarray(self.n_samples),
            _FRAME_TIME_KEYS[frame]: time_grid,
            "magnitude_quantiles": magnitude_quantiles,
            "valid_draws": valid_draws,
        }

        if return_magnitudes:
            # The exact samples the quantiles above were computed from.
            prediction["magnitude_samples"] = ensemble[3]

        if save_npz:
            if output_path is None:
                output_path = default_output_path(self.event, frame=frame)

            prediction["saved_npz"] = str(
                _save_npz(output_path=output_path, prediction=prediction)
            )

        return prediction
