"""Synthetic photometry and light-curve prediction from rest-frame spectra."""

from genova.photometry.bandpasses import register_bandpass, validate_bandpass_curve
from genova.photometry.lightcurves import lightcurve_quantiles
from genova.photometry.pipelines import (
    observer_ab_magnitudes_from_rest_spectra,
    predict_case_lightcurve,
    predict_ensemble_lightcurve,
)

__all__ = [
    "lightcurve_quantiles",
    "register_bandpass",
    "validate_bandpass_curve",
    "predict_case_lightcurve",
    "predict_ensemble_lightcurve",
    "observer_ab_magnitudes_from_rest_spectra",
]
