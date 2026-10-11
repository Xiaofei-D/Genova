#!/usr/bin/env python3
"""Predict kilonova spectra for one physical case.

Input
    None. The surrogate bundled with Genova is loaded from the installed
    package, so this script needs no data files and no arguments.

Demonstrates
    genova.prediction.load_flow_surrogate
    genova.prediction.predict_spectra

Output
    Printed array shapes and a flux summary. Nothing is written to disk.

Run with:
    python examples/predict_spectra.py
"""

import numpy as np

from genova.constants import CASE_CONDITION_NAMES
from genova.prediction import load_flow_surrogate, predict_spectra

# One binary neutron star, in the canonical condition order:
# m1 and m2 in solar masses, lambda1 and lambda2 dimensionless tidal
# deformabilities, cos_theta_obs the cosine of the viewing angle.
CASE_CONDITIONS = np.array([[1.40, 1.30, 500.0, 700.0, 0.50]], dtype=np.float32)

# Rest-frame days at which to predict.
TIME_REST = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)

# Stochastic draws per case and time, which is how the surrogate expresses
# prediction uncertainty.
N_SAMPLES = 8


def main() -> None:
    metadata, flow_model = load_flow_surrogate(device="cpu")

    print(f"surrogate:   {metadata.path.name}")
    print(f"conditions:  {metadata.condition_names}")
    print(
        f"wavelengths: {metadata.wavelengths.size} points, "
        f"{metadata.wavelengths[0]:.0f}-{metadata.wavelengths[-1]:.0f} A"
    )

    spectra = predict_spectra(
        flow_model,
        metadata,
        case_conditions=CASE_CONDITIONS,
        time_rest=TIME_REST,
        n_samples=N_SAMPLES,
        device="cpu",
    )

    case = ", ".join(
        f"{name}={value:g}"
        for name, value in zip(CASE_CONDITION_NAMES, CASE_CONDITIONS[0])
    )
    print(f"\ncase:        {case}")

    # (n_cases, n_samples, n_times, n_wavelengths)
    print(f"spectra shape: {spectra.shape}")
    print(f"  n_cases       = {spectra.shape[0]}")
    print(f"  n_samples     = {spectra.shape[1]}")
    print(f"  n_times       = {spectra.shape[2]}")
    print(f"  n_wavelengths = {spectra.shape[3]}")

    # Rest-frame F_lambda in erg/s/cm^2/A, normalised to a source at 10 pc.
    median_spectrum = np.median(spectra[0], axis=0)
    print("\nmedian F_lambda at 10 pc, peak per time:")
    for time, spectrum in zip(TIME_REST, median_spectrum):
        peak_wavelength = metadata.wavelengths[np.argmax(spectrum)]
        print(
            f"  t = {time:5.2f} d, peak {spectrum.max():.3e} at {peak_wavelength:.0f} A"
        )


if __name__ == "__main__":
    main()
