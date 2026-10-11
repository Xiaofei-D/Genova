"""End-to-end workflows from physical conditions to event light curves."""

from genova.workflows.event_lightcurves import (
    EventLightcurveModel,
    default_output_path,
)

__all__ = [
    "EventLightcurveModel",
    "default_output_path",
]
