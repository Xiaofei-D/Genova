"""Shared defaults, asset locations, and physical constants."""

from pathlib import Path

import numpy as np

# ---------------------------------------
# Asset locations and discovery patterns
# ---------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent

# Against the installed package location,
# NOT the current working directory.
MODEL_DIR = PACKAGE_DIR / "models"
SURROGATE_DIR = MODEL_DIR / "reference_surrogates"

# Files needed in each surrogate bundle, for example
# N21_flow.pt, N21_flow_config.json, and N21_convex_hull.npz.
FLOW_CHECKPOINT_PATTERN = "*_flow.pt"
FLOW_CONFIG_PATTERN = "*_config.json"
CONVEX_HULL_PATTERN = "*_convex_hull.npz"

# ------------------------
# Compute defaults
# ------------------------

CUDA_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"

# None selects the first visible CUDA device, or the CPU if CUDA is unavailable.
DEVICE: str | None = None

# Maximum number of conditioning rows per flow call.
CHUNK_SIZE = 2048

# Default number of draws per condition set.
N_SAMPLES = 1

# -----------------------------
# Prediction defaults
# -----------------------------

# Rest frame time default.
# This range sits comfortably in the flow model's training support.
# Extrapolation is not allowed.
TIME_REST_MIN = 0.4
TIME_REST_MAX = 10.0
N_TIMES = 100

# Default sncosmo bandpass names, from UV to NIR
BANDS: tuple[str, ...] = (
    "sdssu",
    "ps1::g",
    "ps1::r",
    "ps1::i",
    "ps1::z",
    "2massj",
    "2massh",
    "2massks",
)

# Median plus the central 68 per cent and 90 per cent intervals.
QUANTILES: tuple[float, ...] = (0.05, 0.16, 0.50, 0.84, 0.95)

# Spline degree used in the interpolation
TIME_SPLINE_DEGREE = 3


def time_rest_grid(
    time_min: float = TIME_REST_MIN,
    time_max: float = TIME_REST_MAX,
    n_times: int = N_TIMES,
) -> np.ndarray:
    """
    Rest-frame time grid in days with logarithmic spacing, (n_times,).

    A function rather than a module-level array so that every caller gets its
    own array to modify.
    """
    return np.geomspace(
        time_min,
        time_max,
        n_times,
    ).astype(np.float32)


# -----------------------------
# Condition and column names
# -----------------------------

TIME_CONDITION_NAME = "time_rest"

# Condition names with expected order
CASE_CONDITION_NAMES: tuple[str, ...] = (
    "m1",
    "m2",
    "lambda1",
    "lambda2",
    "cos_theta_obs",
)

# Distance information needed for predictions in observer frame.
POSTERIOR_DISTANCE_COLUMNS: tuple[str, ...] = (
    "redshift",
    "luminosity_distance_Mpc",
)

POSTERIOR_REQUIRED_COLUMNS: tuple[str, ...] = (
    *CASE_CONDITION_NAMES,
    *POSTERIOR_DISTANCE_COLUMNS,
)

# Posterior columns used to select membership of the flow's training domain.
HULL_FEATURE_COLUMNS: tuple[str, ...] = ("m1", "m2", "lambda1", "lambda2")

# ---------------------
# Physical constants
# ---------------------

REFERENCE_DISTANCE_MPC = 1.0e-5
HC_ERG_AA = 1.9864458571489284e-08

# --------------------
# Output conventions
# --------------------

OUTPUT_DIR = Path("outputs")
# Formatted with the frame name, "obs" or "rest".
LIGHTCURVE_NPZ_TEMPLATE = "{frame}_lightcurve_prediction.npz"
