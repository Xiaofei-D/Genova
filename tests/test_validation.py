"""Tests for filtering a posterior against the flow's training domain."""

import numpy as np
import pandas as pd
import pytest

from genova.validation import posterior_in_convex_hull


def test_hull_filter_requires_canonical_columns(tmp_path) -> None:
    # e.g., l1/l2 are not accepted.
    csv_path = tmp_path / "posterior.csv"
    pd.DataFrame({"m1": [1.4], "m2": [1.3], "l1": [400.0], "l2": [450.0]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="missing required columns"):
        posterior_in_convex_hull(csv_path, tmp_path / "out.csv")


def test_hull_filter_keeps_inside_drops_outside(tmp_path):
    input_csv = tmp_path / "posterior.csv"

    pd.DataFrame(
        {
            "m1": [1.5026, 5.00],
            "m2": [1.2357, 4.00],
            "lambda1": [131.16, 5.0],
            "lambda2": [1451.84, 5.0],
            "extra": [1.0, 2.0],
        }
    ).to_csv(input_csv, index=False)

    output_csv = posterior_in_convex_hull(input_csv, tmp_path / "filtered.csv")

    filtered = pd.read_csv(output_csv)

    assert len(filtered) == 1
    np.testing.assert_allclose(filtered["m1"], [1.5026])
    assert "extra" in filtered.columns
