"""Load the flow model and generate rest frame spectra samples."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from glasflow import RealNVP

from genova.constants import (
    CHUNK_SIZE,
    CPU_DEVICE,
    CUDA_DEVICE,
    DEVICE,
    FLOW_CHECKPOINT_PATTERN,
    FLOW_CONFIG_PATTERN,
    N_SAMPLES,
    SURROGATE_DIR,
    TIME_CONDITION_NAME,
)


def resolve_device(device: str | torch.device | None = DEVICE) -> torch.device:
    """Return the torch device to run on.

    ``None`` selects the first visible GPU when CUDA is available and falls back
    to the CPU otherwise. Any explicit device name or ``torch.device`` is used
    as given.
    """
    if device is not None:
        return torch.device(device)

    return torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)


@dataclass(frozen=True)
class FlowMetadata:
    """Metadata required to reconstruct a flow and reproduce predictions.

    Attributes
    ----------
    path:
        Source JSON config path.
    n_inputs:
        Number of flow input dimensions, equal to the wavelength grid size.
    n_conditional_inputs:
        Number of conditionals expected by the flow.
    n_transforms:
        Number of RealNVP transforms used during training.
    n_neurons:
        Number of hidden neurons used in each transform.
    batch_norm_between_transforms:
        Whether batch normalisation was used between RealNVP transforms.
    linear_transform:
        Optional linear transform applied before each coupling transform.
    wavelengths:
        Wavelength grid in Angstrom in rest frame, shape (n_wavelengths,).
    condition_names:
        Condition names in the order of conditionals.
    condition_min:
        Physical minimum values used for condition scaling.
    condition_max:
        Physical maximum values used for condition scaling.
    asinh_scale:
        Flux scale of the spectra representation in erg s^-1 cm^-2 Angstrom^-1,
        with x = asinh(F / asinh_scale).
    x_min, x_max:
        Min-max scaling of x. The flow models y = (x - x_min) / (x_max - x_min),
        so physical spectra are recovered as
        F = asinh_scale * sinh(x_min + y * (x_max - x_min)).
    """

    path: Path
    n_inputs: int
    n_conditional_inputs: int
    n_transforms: int
    n_neurons: int
    batch_norm_between_transforms: bool
    linear_transform: str | None
    wavelengths: np.ndarray
    condition_names: tuple[str, ...]
    condition_min: np.ndarray
    condition_max: np.ndarray
    asinh_scale: float
    x_min: float
    x_max: float

    def scale_conditions(self, conditions_phys: np.ndarray) -> np.ndarray:
        """Min-max scale for physical conditions."""
        conditions_arr = np.asarray(conditions_phys, dtype=np.float32)

        if conditions_arr.shape[-1] != len(self.condition_names):
            raise ValueError(
                "Condition number mismatch. "
                f"Expected {len(self.condition_names)}, got "
                f"{conditions_arr.shape[-1]}."
            )

        # A zero span means this condition was constant during training.
        # Use 1.0 so the saved constant value maps to 0.0.
        condition_span = self.condition_max - self.condition_min
        condition_span = np.where(condition_span == 0.0, 1.0, condition_span)

        return (conditions_arr - self.condition_min) / condition_span

    def spectra_from_flow(self, y: torch.Tensor) -> torch.Tensor:
        """Physical spectra F from flow outputs y (undo min-max, then asinh)."""
        x = self.x_min + y * (self.x_max - self.x_min)
        return self.asinh_scale * torch.sinh(x)


def load_flow_config(config_path: str | Path) -> FlowMetadata:
    """
    Load flow architecture and prediction metadata from a JSON config.

    Parameters
    ----------
    config_path:
        Path to the JSON configuration file.

    Returns
    -------
    FlowMetadata
        Flow architecture and prediction metadata.
    """
    path = Path(config_path).expanduser().resolve()

    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    for key in ("model_config", "prediction_metadata"):
        if key not in config:
            raise ValueError(f"Config {path} is missing the required '{key}' section.")

    model_config = config["model_config"]
    prediction_metadata = config["prediction_metadata"]

    condition_names = tuple(
        str(name) for name in prediction_metadata["condition_names"]
    )
    condition_min = np.asarray(prediction_metadata["condition_min"], dtype=np.float32)
    condition_max = np.asarray(prediction_metadata["condition_max"], dtype=np.float32)
    wavelengths = np.asarray(
        prediction_metadata["wavelengths"], dtype=np.float64
    ).ravel()

    # Spectra representation: min-max scaled x = asinh(F / asinh_scale).
    for key in ("asinh_scale", "x_min", "x_max"):
        if key not in prediction_metadata:
            raise ValueError(
                f"Config {path} has no '{key}'. Only min-max scaled asinh "
                "models are supported; older model bundles cannot be loaded."
            )
    asinh_scale = float(prediction_metadata["asinh_scale"])
    x_min = float(prediction_metadata["x_min"])
    x_max = float(prediction_metadata["x_max"])
    if not (np.isfinite([asinh_scale, x_min, x_max]).all() and x_max > x_min):
        raise ValueError(
            f"Config {path} has invalid preprocessing: asinh_scale {asinh_scale}, "
            f"x_min {x_min}, x_max {x_max} (finite values and x_max > x_min required)."
        )

    # Cross-check the model config and metadata.
    if int(model_config["n_inputs"]) != wavelengths.size:
        raise ValueError(
            f"Config {path} is inconsistent: model_config n_inputs is "
            f"{model_config['n_inputs']} but the wavelength grid has "
            f"{wavelengths.size} points."
        )

    if int(model_config["n_conditional_inputs"]) != len(condition_names):
        raise ValueError(
            f"Config {path} is inconsistent: model_config n_conditional_inputs "
            f"is {model_config['n_conditional_inputs']} but condition_names has "
            f"{len(condition_names)} elements."
        )

    return FlowMetadata(
        path=path,
        n_inputs=int(model_config["n_inputs"]),
        n_conditional_inputs=int(model_config["n_conditional_inputs"]),
        n_transforms=int(model_config["n_transforms"]),
        n_neurons=int(model_config["n_neurons"]),
        linear_transform=model_config.get("linear_transform", None),
        batch_norm_between_transforms=bool(
            model_config.get("batch_norm_between_transforms", True)
        ),
        wavelengths=wavelengths,
        condition_names=condition_names,
        condition_min=condition_min,
        condition_max=condition_max,
        asinh_scale=asinh_scale,
        x_min=x_min,
        x_max=x_max,
    )


def _find_surrogate_files(
    surrogate_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Locate the checkpoint and JSON config of one model bundle.

    Defaults to the model bundled with the package.
    """
    root = Path(surrogate_dir) if surrogate_dir is not None else SURROGATE_DIR
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Surrogate directory does not exist: {root}")

    checkpoints = sorted(root.glob(FLOW_CHECKPOINT_PATTERN))
    configs = sorted(root.glob(FLOW_CONFIG_PATTERN))

    missing = []
    if not checkpoints:
        missing.append(FLOW_CHECKPOINT_PATTERN)
    if not configs:
        missing.append(FLOW_CONFIG_PATTERN)

    if missing:
        raise FileNotFoundError(
            f"No flow model found in {root}. Expected a file matching "
            f"{' and '.join(missing)}."
        )

    return checkpoints[0].resolve(), configs[0].resolve()


def load_flow_model(
    checkpoint_path: str | Path,
    metadata: FlowMetadata,
    *,
    device: str | torch.device | None = DEVICE,
) -> torch.nn.Module:
    """Load the flow model for prediction.

    Parameters
    ----------
    checkpoint_path:
        Path to a checkpoint (.pt) holding a state dict, optionally under a
        ``"model_state_dict"`` entry.
    metadata:
        Metadata used to recover the flow dimensions.
    device:
        Torch device; see :func:`resolve_device`.

    Returns
    -------
    torch.nn.Module
        Flow model in evaluation mode.
    """
    device_obj = resolve_device(device)

    # Rebuild the same RealNVP architecture that was used during training.
    flow = RealNVP(
        n_inputs=metadata.n_inputs,
        n_transforms=metadata.n_transforms,
        n_conditional_inputs=metadata.n_conditional_inputs,
        n_neurons=metadata.n_neurons,
        linear_transform=metadata.linear_transform,
        batch_norm_between_transforms=metadata.batch_norm_between_transforms,
    ).to(device_obj)

    # weights_only=True for state dictionary checkpoints.
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device_obj,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    flow.load_state_dict(checkpoint)
    flow.eval()

    return flow


def load_flow_surrogate(
    *,
    surrogate_dir: str | Path | None = None,
    device: str | torch.device | None = DEVICE,
) -> tuple[FlowMetadata, torch.nn.Module]:
    """Load a flow surrogate bundle (model and JSON config).

    Parameters
    ----------
    surrogate_dir:
        Directory holding one model bundle. Defaults to the bundle installed
        with the package.
    device:
        Torch device; see :func:`resolve_device`.

    Returns
    -------
    tuple[FlowMetadata, torch.nn.Module]
        The prediction metadata and the loaded flow in evaluation mode.
    """
    checkpoint_path, config_path = _find_surrogate_files(surrogate_dir=surrogate_dir)

    metadata = load_flow_config(config_path)
    flow_model = load_flow_model(checkpoint_path, metadata, device=device)

    return metadata, flow_model


def predict_spectra(
    flow_model: torch.nn.Module,
    metadata: FlowMetadata,
    *,
    case_conditions: np.ndarray,
    time_rest: np.ndarray,
    n_samples: int = N_SAMPLES,
    chunk_size: int = CHUNK_SIZE,
    device: str | torch.device | None = DEVICE,
    clip_negative: bool = True,
) -> np.ndarray:
    """Generate physical spectra samples in rest frame for one or more cases.

    Parameters
    ----------
    flow_model:
        Flow model in evaluation mode.
    metadata:
        Metadata used for condition and spectra scaling.
    case_conditions:
        Physical case conditions, shape (n_cases, 5), in the column order
        (m1, m2, lambda1, lambda2, cos_theta_obs).
    time_rest:
        Prediction time grid in rest frame in days, (n_times,).
    n_samples:
        Number of spectra drawn for each condition set.
    chunk_size:
        Maximum number of conditioning rows per flow call.
    device:
        CPU or GPU for sampling.
    clip_negative:
        If True, clip negative physical fluxes to zero.

    Returns
    -------
    np.ndarray
        Rest-frame F_lambda spectra at 10 pc,
        (n_cases, n_samples, n_times, n_wavelengths).
    """
    case_arr = np.asarray(case_conditions, dtype=np.float32)
    time_arr = np.asarray(time_rest, dtype=np.float32)

    # Time is supplied as its own grid, so the remaining conditions
    # are the per-case ones without time.
    case_condition_names = tuple(
        name for name in metadata.condition_names if name != TIME_CONDITION_NAME
    )

    # Check the shape of the case conditions
    if case_arr.ndim != 2 or case_arr.shape[1] != len(case_condition_names):
        raise ValueError(
            "case_conditions must have shape "
            f"(n_cases, {len(case_condition_names)}) with column order "
            f"{case_condition_names}. Got shape {case_arr.shape}."
        )

    n_cases = case_arr.shape[0]
    n_times = time_arr.size
    n_rows = n_cases * n_times * n_samples
    n_wavelengths = metadata.wavelengths.size

    # Build each complete condition set once,
    # then repeat for the number of samples as requested.
    case_grid = np.repeat(case_arr, n_times, axis=0)
    time_grid = np.tile(time_arr, n_cases)

    condition_values = {TIME_CONDITION_NAME: time_grid}
    for column_index, name in enumerate(case_condition_names):
        condition_values[name] = case_grid[:, column_index]

    # Follow the exact condition order saved in the metadata.
    condition_sets_phys = np.column_stack(
        [condition_values[name] for name in metadata.condition_names]
    )

    # Repeat each condition row "n_samples" times before sampling.
    conditions_phys = np.repeat(condition_sets_phys, repeats=n_samples, axis=0)
    conditions_scaled = metadata.scale_conditions(conditions_phys)

    # Prepare array for spectra samples.
    spectra_rows = np.empty((n_rows, n_wavelengths), dtype=np.float64)

    device_obj = resolve_device(device)

    # Sample in chunks
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)

        conditional = torch.as_tensor(
            conditions_scaled[start:stop],
            dtype=torch.float32,
            device=device_obj,
        )

        with torch.no_grad():
            spectra_scaled = flow_model.sample(
                conditional=conditional,
                num_samples=conditional.shape[0],
            )

            # Check shape of the output
            if isinstance(spectra_scaled, (tuple, list)):
                spectra_scaled = spectra_scaled[0]

            spectra_phys = metadata.spectra_from_flow(spectra_scaled.double())

            if clip_negative:
                spectra_phys = torch.clamp(spectra_phys, min=0.0)

        spectra_rows[start:stop] = (
            spectra_phys.detach().cpu().numpy().astype(np.float64)
        )

    # Rows were generated in (case, time, sample) order;
    # We put "samples" before "times" for only calculation reasons
    spectra = spectra_rows.reshape(n_cases, n_times, n_samples, n_wavelengths)

    return spectra.transpose(0, 2, 1, 3)
