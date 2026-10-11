"""Command line interface: posterior CSV in, light-curve prediction NPZ out.

Example:
    genova --posterior-csv posterior.csv --frame obs --magnitudes
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from genova import __version__
from genova.constants import (
    BANDS,
    DEVICE,
    N_SAMPLES,
    N_TIMES,
    QUANTILES,
    TIME_REST_MAX,
    TIME_REST_MIN,
    time_rest_grid,
)
from genova.validation import posterior_in_convex_hull
from genova.workflows import EventLightcurveModel, default_output_path


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the "genova" command."""
    parser = argparse.ArgumentParser(
        prog="genova",
        description=(
            "Predict event-level kilonova light curves from a posterior CSV with "
            "columns m1, m2, lambda1, lambda2, cos_theta_obs, redshift, "
            "luminosity_distance_Mpc. Samples outside the surrogate's training "
            "domain are dropped first. Results are saved as a compressed NPZ file."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--posterior-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="Posterior CSV, one physical case per row.",
    )
    parser.add_argument(
        "--frame",
        choices=("obs", "rest"),
        default="obs",
        help="'obs' for apparent magnitudes, 'rest' for absolute magnitudes "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output NPZ path (default: outputs/<event>/<frame>_lightcurve_prediction.npz).",
    )
    parser.add_argument(
        "--magnitudes",
        action="store_true",
        help=(
            "Also save the full magnitude samples, "
            "(n_cases, n_samples, n_times, n_bands). Can be large."
        ),
    )
    parser.add_argument(
        "--event",
        default=None,
        metavar="NAME",
        help="Event name (default: the name of the directory holding the CSV).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        metavar="N",
        help="Surrogate draws per posterior case (default: %(default)s).",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=list(BANDS),
        metavar="BAND",
        help="sncosmo bandpass names (default: %(default)s).",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=list(QUANTILES),
        metavar="Q",
        help="Quantile levels in [0, 1] (default: %(default)s).",
    )
    parser.add_argument(
        "--time-min",
        type=float,
        default=TIME_REST_MIN,
        metavar="DAYS",
        help="Start of the log-spaced rest-frame time grid in days (default: %(default)s).",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=TIME_REST_MAX,
        metavar="DAYS",
        help="End of the log-spaced rest-frame time grid in days (default: %(default)s).",
    )
    parser.add_argument(
        "--n-times",
        type=int,
        default=N_TIMES,
        metavar="N",
        help="Number of points in the log-spaced rest-frame time grid (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Torch device, e.g. 'cpu' or 'cuda:0' (default: GPU if available, else CPU).",
    )
    parser.add_argument(
        "--surrogate-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of a custom surrogate bundle (default: the bundled model).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the "genova" command. "argv" defaults to the command line arguments."""
    args = build_parser().parse_args(argv)

    # Resolved here so the filtered CSV location does not change the event name.
    event = args.event or args.posterior_csv.parent.name
    output_path = args.output or default_output_path(event, frame=args.frame)

    # Only predict inside the surrogate's training domain.
    posterior_csv = posterior_in_convex_hull(
        args.posterior_csv,
        output_path.parent / f"{event}_posteriors_in_convex_hull.csv",
        hull_dir=args.surrogate_dir,
    )

    model = EventLightcurveModel.from_csv(
        posterior_csv,
        event=event,
        surrogate_dir=args.surrogate_dir,
        device=args.device,
        time_rest=time_rest_grid(args.time_min, args.time_max, args.n_times),
        bands=tuple(args.bands),
        quantiles=args.quantiles,
        n_samples=args.n_samples,
    )

    n_cases = model.case_conditions.shape[0]
    if n_cases == 0:
        raise ValueError(f"No posterior samples left to predict in {posterior_csv}.")

    prediction = model.predict_lightcurve_quantiles(
        frame=args.frame,
        save_npz=True,
        output_path=output_path,
        return_magnitudes=args.magnitudes,
    )

    print(f"event:               {prediction['event']}")
    print(f"convex hull:         {n_cases} samples kept, saved {posterior_csv}")
    print(f"magnitude_quantiles: {prediction['magnitude_quantiles'].shape}")
    if args.magnitudes:
        print(f"magnitude_samples:   {prediction['magnitude_samples'].shape}")
    print(f"saved:               {prediction['saved_npz']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
