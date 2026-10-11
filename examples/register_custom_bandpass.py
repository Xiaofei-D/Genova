#!/usr/bin/env python3
"""Predict a light curve through a custom photometric bandpass.

Run with:
    python examples/register_custom_bandpass.py
"""

import numpy as np

from genova.constants import QUANTILES, REFERENCE_DISTANCE_MPC
from genova.photometry import predict_case_lightcurve, register_bandpass
from genova.prediction import load_flow_surrogate, predict_spectra

# A simple triangular transmission curve peaking at 6000 A
CUSTOM_BAND_NAME = "demo_band"
CUSTOM_BAND_WAVE = np.array([5000.0, 6000.0, 7000.0])
CUSTOM_BAND_TRANSMISSION = np.array([0.0, 1.0, 0.0])

# One binary neutron star
# m1 and m2 in solar masses, lambda1 and lambda2 dimensionless tidal
# deformabilities, cos_theta_obs the cosine of the viewing angle.
CASE_CONDITIONS = np.array([[1.40, 1.30, 500.0, 700.0, 0.50]], dtype=np.float32)

# Rest-frame days at which to predict.
TIME_REST = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)

# Sample draws per case and time from the flow
N_SAMPLES = 8


def main() -> None:
    register_bandpass(CUSTOM_BAND_WAVE, CUSTOM_BAND_TRANSMISSION, CUSTOM_BAND_NAME)
    print(f"registered bandpass: {CUSTOM_BAND_NAME!r}")

    metadata, flow_model = load_flow_surrogate(device="cpu")

    spectra = predict_spectra(
        flow_model,
        metadata,
        case_conditions=CASE_CONDITIONS,
        time_rest=TIME_REST,
        n_samples=N_SAMPLES,
        device="cpu",
    )

    # light curve at 10 pc: redshift=0, distance=1e-5 Mpc.
    time_obs, quantile_values = predict_case_lightcurve(
        time_rest=TIME_REST,
        wave_rest=metadata.wavelengths,
        spectra_rest_at_10pc=spectra[0],
        redshift=0.0,
        luminosity_distance_mpc=REFERENCE_DISTANCE_MPC,
        bands=[CUSTOM_BAND_NAME],
        quantiles=np.asarray(QUANTILES, dtype=float),
    )

    # (n_quantiles, n_times, n_bands)
    print(f"\nquantile_values shape: {quantile_values.shape}")

    median_index = QUANTILES.index(0.50)
    print(f"\nmedian AB magnitude through {CUSTOM_BAND_NAME!r}:")
    for time, magnitude in zip(time_obs, quantile_values[median_index, :, 0]):
        print(f"  t={time:5.2f} d   M_abs={magnitude:6.2f} mag")


if __name__ == "__main__":
    main()
