"""Restrict a posterior to the surrogate's training domain."""

from genova.validation.convex_hull import (
    find_convex_hull_file,
    posterior_in_convex_hull,
)

__all__ = [
    "find_convex_hull_file",
    "posterior_in_convex_hull",
]
