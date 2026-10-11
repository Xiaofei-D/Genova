#!/usr/bin/env python3
"""Predict event light curve (quantiles) from gravitational wave posterior samples.

Input
    examples/data/GW170817/GW170817_lowspin_posteriors_sourceframe.csv
    Source-frame samples from the GWTC-1 low-spin GW170817 posterior, with the
    columns Genova expects: m1, m2, lambda1, lambda2, cos_theta_obs, redshift,
    luminosity_distance_Mpc.

Demonstrates
    genova.validation.posterior_in_convex_hull   restrict to the training domain
    genova.workflows.EventLightcurveModel        posterior -> spectra -> light curves

Output, written to examples/outputs/<event>/
    <event>_posteriors_in_convex_hull.csv   filtered posterior
    obs_lightcurve_prediction.npz           observer-frame apparent magnitudes
    rest_lightcurve_prediction.npz          rest-frame absolute magnitudes
    <event>_lightcurve_obs.png              one panel per band
    <event>_lightcurve_rest.png             one panel per band

Array shapes
    spectra             (n_cases, n_samples, n_times, n_wavelengths)
    magnitude_quantiles (n_quantiles, n_times, n_bands)
    valid_draws         (n_times, n_bands)

Run with:
    python examples/predict_event_lightcurve.py
    python examples/predict_event_lightcurve.py --max-cases 100   # faster
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from genova.constants import QUANTILES, TIME_REST_MAX, TIME_REST_MIN
from genova.validation import posterior_in_convex_hull
from genova.workflows import EventLightcurveModel, default_output_path

# Paths are resolved from this file, so the example runs from any directory.
EXAMPLE_DIR = Path(__file__).resolve().parent

# Indices into QUANTILES: (0.05, 0.16, 0.50, 0.84, 0.95).
Q_LOW, Q_INNER_LOW, Q_MEDIAN, Q_INNER_HIGH, Q_HIGH = range(len(QUANTILES))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", default="GW170817", help="Event name.")
    parser.add_argument(
        "--posterior-csv",
        type=Path,
        default=None,
        help="Raw posterior CSV. Defaults to the bundled GW170817 samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "outputs",
        help="Directory for generated files.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Stochastic flow draws per posterior sample.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Use only the first N posterior samples. Default: all of them.",
    )
    return parser.parse_args()


def plot_band_panels(
    time: np.ndarray,
    magnitude_quantiles: np.ndarray,
    bands: tuple[str, ...],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    output_path: Path,
) -> None:
    """Plot one panel per band, with the median and two credible intervals."""
    n_columns = 4
    n_rows = int(np.ceil(len(bands) / n_columns))

    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(3.5 * n_columns, 3.5 * n_rows), sharex=True
    )
    axes = np.atleast_1d(axes).ravel()

    for band_index, (ax, band) in enumerate(zip(axes, bands)):
        ax.fill_between(
            time,
            magnitude_quantiles[Q_LOW, :, band_index],
            magnitude_quantiles[Q_HIGH, :, band_index],
            color=color,
            alpha=0.15,
            linewidth=0,
            label="5-95%",
        )
        ax.fill_between(
            time,
            magnitude_quantiles[Q_INNER_LOW, :, band_index],
            magnitude_quantiles[Q_INNER_HIGH, :, band_index],
            color=color,
            alpha=0.30,
            linewidth=0,
            label="16-84%",
        )
        ax.plot(
            time,
            magnitude_quantiles[Q_MEDIAN, :, band_index],
            color=color,
            linewidth=1.8,
            label="median",
        )

        ax.set_title(band)
        ax.set_xscale("log")
        ax.set_xlim(TIME_REST_MIN, TIME_REST_MAX)
        # Brighter sources have smaller magnitudes.
        ax.invert_yaxis()
        ax.grid(alpha=0.25)

    for ax in axes[len(bands) :]:
        ax.set_visible(False)

    axes[0].legend(loc="best", fontsize="small")
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved {output_path}")


def main() -> None:
    args = parse_args()

    raw_posterior_csv = args.posterior_csv or (
        EXAMPLE_DIR
        / "data"
        / args.event
        / f"{args.event}_lowspin_posteriors_sourceframe.csv"
    )
    output_dir = args.output_dir / args.event
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[input]  {raw_posterior_csv}")
    print(f"[output] {output_dir}")

    # The surrogate is only valid inside its training domain, so drop posterior
    # samples that fall outside it before predicting anything.
    filtered_csv = output_dir / f"{args.event}_posteriors_in_convex_hull.csv"
    posterior_in_convex_hull(
        posterior_csv=raw_posterior_csv,
        output_csv=filtered_csv,
    )

    n_raw = len(pd.read_csv(raw_posterior_csv))
    n_kept = len(pd.read_csv(filtered_csv))
    print(f"\n[convex hull] kept {n_kept} of {n_raw} posterior samples")

    prediction_csv = filtered_csv
    if args.max_cases is not None:
        prediction_csv = output_dir / f"{args.event}_posteriors_subset.csv"
        pd.read_csv(filtered_csv).head(args.max_cases).to_csv(
            prediction_csv, index=False
        )
        print(f"[subset] using the first {args.max_cases} samples")

    model = EventLightcurveModel.from_csv(
        posterior_csv=prediction_csv,
        event=args.event,
        device=args.device,
        n_samples=args.n_samples,
    )

    spectra = model.predict_spectra()
    print(f"\n[spectra] {spectra.shape}  (cases, samples, times, wavelengths)")

    # Observer-frame apparent magnitudes, using each sample's own redshift and
    # luminosity distance.
    obs = model.predict_lightcurve_quantiles(
        frame="obs",
        save_npz=True,
        output_path=default_output_path(
            args.event, frame="obs", output_dir=args.output_dir
        ),
    )
    print(f"[obs]  magnitude_quantiles {obs['magnitude_quantiles'].shape}")
    print(
        f"       valid draws per time/band: "
        f"{obs['valid_draws'].min()}-{obs['valid_draws'].max()}"
    )
    print(f"  saved {obs['saved_npz']}")

    # Rest-frame absolute magnitudes, reusing the cached spectra.
    rest = model.predict_lightcurve_quantiles(
        frame="rest",
        save_npz=True,
        output_path=default_output_path(
            args.event, frame="rest", output_dir=args.output_dir
        ),
    )
    print(f"[rest] magnitude_quantiles {rest['magnitude_quantiles'].shape}")
    print(f"  saved {rest['saved_npz']}")

    print("\n[plots]")
    bands = tuple(str(band) for band in obs["bands"])

    plot_band_panels(
        np.asarray(obs["time_obs"], dtype=float),
        np.asarray(obs["magnitude_quantiles"], dtype=float),
        bands,
        title=f"{args.event} observer-frame light curves",
        xlabel="Time [days, observer frame]",
        ylabel="Apparent AB magnitude",
        color="tab:blue",
        output_path=output_dir / f"{args.event.lower()}_lightcurve_obs.png",
    )

    plot_band_panels(
        np.asarray(rest["time_rest_lc"], dtype=float),
        np.asarray(rest["magnitude_quantiles"], dtype=float),
        bands,
        title=f"{args.event} rest-frame absolute light curves",
        xlabel="Time [days, rest frame]",
        ylabel="Absolute AB magnitude",
        color="tab:purple",
        output_path=output_dir / f"{args.event.lower()}_lightcurve_rest.png",
    )


if __name__ == "__main__":
    main()
