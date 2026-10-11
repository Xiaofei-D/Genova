"""Pick posterior samples that fall in the flow's training domain,
which is stored as a convex hull.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from genova.constants import (
    CONVEX_HULL_PATTERN,
    HULL_FEATURE_COLUMNS,
    SURROGATE_DIR,
)

# Posterior rows tested against the hull per batch, to bound memory use.
_HULL_CHUNK_SIZE = 100_000


def find_convex_hull_file(hull_dir: str | Path | None = None) -> Path:
    """Locate the convex hull file (.npz) of a surrogate bundle.

    Defaults to the model installed with the package,
    but can be overridden with a directory path.
    """
    root = Path(hull_dir) if hull_dir is not None else SURROGATE_DIR
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Surrogate directory does not exist: {root}")

    hull_files = sorted(root.glob(CONVEX_HULL_PATTERN))

    if not hull_files:
        raise FileNotFoundError(
            f"No convex-hull file found in {root}. Expecting {CONVEX_HULL_PATTERN}."
        )

    return hull_files[0].resolve()


def posterior_in_convex_hull(
    posterior_csv: str | Path,
    output_csv: str | Path,
    *,
    hull_path: str | Path | None = None,
    hull_dir: str | Path | None = None,
) -> Path:
    """Keep only the posterior samples inside the surrogate's training domain.

    Parameters
    ----------
    posterior_csv:
        Input posterior table. Must contain the columns
        ``m1, m2, lambda1, lambda2``. All other columns are carried through unchanged.
    output_csv:
        Where to write the filtered table. Parent directories are created automatically.
    hull_path:
        Explicit convex hull file (.npz).
    hull_dir:
        Directory to search for a convex hull file. Defaults to the model installed with
        the package.

    Returns
    -------
    Path
        The path the filtered result was written to.
    """
    posterior_csv = Path(posterior_csv)
    output_csv = Path(output_csv)

    if hull_path is None:
        hull_path = find_convex_hull_file(hull_dir)

    hull_path = Path(hull_path).expanduser().resolve()

    posterior = pd.read_csv(posterior_csv)

    missing_columns = [
        column for column in HULL_FEATURE_COLUMNS if column not in posterior.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Posterior {posterior_csv} is missing required columns: "
            f"{missing_columns}. Expected {list(HULL_FEATURE_COLUMNS)}."
        )

    # Load the convex hull data
    with np.load(hull_path, allow_pickle=False) as data:
        equations = data["equations"].astype(float)
        mean = data["mean"].astype(float)
        std = data["std"].astype(float)
        log_floor = float(data["log_floor"])

    masses = [
        posterior[column].to_numpy(dtype=float) for column in HULL_FEATURE_COLUMNS[:2]
    ]
    # Clip before taking the log just in case
    log_deformabilities = [
        np.log10(np.clip(posterior[column].to_numpy(dtype=float), log_floor, None))
        for column in HULL_FEATURE_COLUMNS[2:]
    ]

    # Stack and standardise the masses and lambdas before filtering
    feature_matrix = np.column_stack([*masses, *log_deformabilities])
    feature_matrix = (feature_matrix - mean) / std  # (n_samples, 4)

    normal_vectors = equations[:, :-1]  # (n_faces, 4)
    offsets = equations[:, -1]  # (n_faces,)

    keep_mask = np.empty(feature_matrix.shape[0], dtype=bool)

    # Filtering...
    for start in range(0, feature_matrix.shape[0], _HULL_CHUNK_SIZE):
        stop = min(start + _HULL_CHUNK_SIZE, feature_matrix.shape[0])

        # Signed values to every face; negative means the inner.
        signed_values = (
            normal_vectors @ feature_matrix[start:stop].T + offsets[:, None]
        )  # (n_faces, hull_chunk_size)

        # One valid sample must be inside all faces,
        # i.e. the signed values must be <= 0 for all faces.
        keep_mask[start:stop] = np.all(signed_values <= 0.0, axis=0)

    filtered = posterior.loc[keep_mask].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    return output_csv
