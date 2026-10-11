"""Genova: kilonova spectra and light-curve prediction from a flow surrogate.

The documented entry points are re-exported here:

``load_flow_surrogate`` / ``predict_spectra``
    Physical conditions in, rest-frame spectra out.
``predict_case_lightcurve`` / ``predict_ensemble_lightcurve``
    Rest-frame spectra in, light-curve quantiles (and optionally the full
    magnitude samples) out.
``EventLightcurveModel``
    Both layers wired together for one event.

The ``genova`` command runs ``EventLightcurveModel`` on a posterior CSV.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

# The version is declared once, in pyproject.toml, and read back from the
# installed package metadata. Defined before the re-exports below so submodules
# can import it.
try:
    __version__ = _installed_version("genova")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "unknown"

from genova.photometry import predict_case_lightcurve, predict_ensemble_lightcurve
from genova.prediction import load_flow_surrogate, predict_spectra
from genova.workflows import EventLightcurveModel

__all__ = [
    "__version__",
    "load_flow_surrogate",
    "predict_spectra",
    "predict_case_lightcurve",
    "predict_ensemble_lightcurve",
    "EventLightcurveModel",
]
