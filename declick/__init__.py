"""Dependency-free click repair primitives for old audio recordings."""

from .repair import (
    DeclickError,
    clone_measure,
    declick_batch,
    detect_clicks,
    envelope_guard,
    repair_point,
    repair_span,
    second_diff,
    validate,
)

__all__ = [
    "DeclickError",
    "clone_measure",
    "declick_batch",
    "detect_clicks",
    "envelope_guard",
    "repair_point",
    "repair_span",
    "second_diff",
    "validate",
]
