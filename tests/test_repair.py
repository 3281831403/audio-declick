"""Fifteen red tests for the click-repair contracts.

Every test calls at least one public stub in ``declick.repair``.  The suite is
intentionally red before the implementation is written; it must collect and
fail only with ``NotImplementedError``.
"""

import math

import pytest

from declick.repair import (
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

CONFIG = {
    "threshold": 0.5,
    "min_gap": 1,
    "span": 1,
    "method": "linear",
    "max_db": 6.0,
}


def test_second_diff_reports_impulse_with_correct_sign():
    """Defect 1: the second difference must peak negative at an impulse."""
    samples = [0.0, 0.0, 1.0, 0.0, 0.0]

    result = second_diff(samples)

    assert result == pytest.approx([1.0, -2.0, 1.0], abs=1e-6)
    assert result[1] < 0.0


def test_detect_clicks_deduplicates_same_click():
    """Defect 2: one click must yield one index, not one per flank."""
    samples = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0]

    clicks = detect_clicks(samples, threshold=0.5, min_gap=1)

    assert clicks == [3, 7]


def test_repair_point_interpolates_both_sides():
    """Defect 3: a single point must not be filled with the previous value."""
    samples = [0.2, 0.1, 0.9, 0.3, 0.3]
    original = list(samples)

    repaired = repair_point(samples, idx=2, span=1)

    assert repaired == pytest.approx([0.2, 0.1, 0.2, 0.3, 0.3], abs=1e-6)
    assert repaired[2] != pytest.approx(samples[1], abs=1e-6)
    assert samples == original
    assert repaired is not samples


def test_repair_span_keeps_endpoints_continuous():
    """Defect 4: the repaired block must meet both exterior anchors."""
    samples = [0.0, 0.1, 0.9, 0.8, 0.2, 0.1]

    repaired = repair_span(samples, start=2, end=3, method="linear")

    assert repaired == pytest.approx(
        [0.0, 0.1, 0.1333333333, 0.1666666667, 0.2, 0.1],
        abs=1e-6,
    )
    assert repaired[1] == pytest.approx(samples[1], abs=1e-6)
    assert repaired[4] == pytest.approx(samples[4], abs=1e-6)

    ar_repaired = repair_span(samples, start=2, end=3, method="ar")
    assert len(ar_repaired) == len(samples)
    assert ar_repaired[1] == pytest.approx(samples[1], abs=1e-6)
    assert ar_repaired[4] == pytest.approx(samples[4], abs=1e-6)


def test_envelope_guard_limits_repaired_energy():
    """Defect 5: a repaired span must not become a much louder blast."""
    samples = [0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02]
    repaired = [0.02, -0.02, 1.0, -1.0, 0.02, -0.02, 0.02]

    guarded = envelope_guard(samples, repaired, max_db=6.0)

    local_rms = math.sqrt(sum(value * value for value in guarded[2:4]) / 2.0)
    allowed_rms = 0.02 * (10.0 ** (6.0 / 20.0))
    assert local_rms <= allowed_rms + 1e-6
    assert max(abs(value) for value in guarded) <= 0.05


def test_repair_span_rejects_unknown_method():
    """Defect 6: an unknown method must not silently run linear repair."""
    samples = [0.0, 0.0, 0.5, 0.0, 0.0]

    with pytest.raises(DeclickError):
        repair_span(samples, start=2, end=3, method="cubic")


def test_invalid_indices_are_reported_not_crashing():
    """Defect 7: out-of-range indices raise DeclickError and validate reports."""
    with pytest.raises(DeclickError):
        repair_point([0.0, 0.1, 0.0], idx=99, span=1)

    problems = validate([0.0] * 8, [(0, 1), (6, 9)])
    assert any("out of bounds" in problem for problem in problems)


def test_validate_flags_overlap_and_density():
    """Defect 8: dense clicks are rejected instead of flattening the track."""
    dense = validate([0.0] * 20, [(0, 9)])
    assert any("too dense" in problem for problem in dense)

    overlapping = validate([0.0] * 40, [(1, 4), (3, 6)])
    assert any("overlap" in problem for problem in overlapping)


def test_batch_keeps_per_track_detection_isolated():
    """Defect 9: one track's click indices must not leak into another track."""
    tracks = {
        "left": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        "right": [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
    }

    result = declick_batch(tracks, CONFIG)

    assert len(result["left"]) == len(tracks["left"])
    assert len(result["right"]) == len(tracks["right"])
    assert result["left"] is not result["right"]
    assert result["left"][2] == pytest.approx(0.0, abs=1e-6)
    assert result["right"][5] == pytest.approx(0.0, abs=1e-6)
    assert result["left"][5] == pytest.approx(0.0, abs=1e-6)
    assert result["right"][2] == pytest.approx(0.0, abs=1e-6)

    invalid = {"good": [0.0] * 5, "bad": [0.0, 1.5, 0.0, 0.0, 0.0]}
    with pytest.raises(DeclickError):
        declick_batch(invalid, CONFIG)
    assert invalid["bad"][1] == pytest.approx(1.5, abs=1e-6)


def test_detect_clicks_rejects_negative_threshold():
    """Defect 10: a non-positive threshold is invalid, not a global click."""
    with pytest.raises(DeclickError):
        detect_clicks([0.0] * 8, threshold=-0.1, min_gap=2)


def test_clean_track_roundtrip_is_identity():
    """Combination: with no click, output equals input within 1e-9."""
    samples = [-0.2, 0.0, 0.15, -0.05, 0.0]
    original = list(samples)

    result = declick_batch({"clean": samples}, {**CONFIG, "threshold": 1.0})

    assert result["clean"] == pytest.approx(samples, abs=1e-9)
    assert samples == original
    assert clone_measure(samples, result["clean"]) == pytest.approx(0.0, abs=1e-6)
    assert clone_measure([1.0, -1.0], [-1.0, 1.0]) == pytest.approx(1.0, abs=1e-6)


def test_threshold_one_detects_nothing():
    """Combination: a signal at the threshold is not strictly above it."""
    samples = [0.0, 0.0, 0.5, 0.0, 0.0]

    assert detect_clicks(samples, threshold=1.0, min_gap=2) == []


def test_empty_inputs_return_empty():
    """Combination: empty audio and an empty batch stay empty."""
    assert second_diff([]) == []
    assert detect_clicks([], threshold=0.5, min_gap=2) == []
    assert validate([], []) == []
    assert declick_batch({}, CONFIG) == {}


def test_zero_threshold_raises_declick_error():
    """Combination: threshold=0 is rejected before detection."""
    with pytest.raises(DeclickError):
        detect_clicks([0.0] * 8, threshold=0.0, min_gap=2)


def test_batch_success_preserves_track_order():
    """Combination: successful batches keep insertion order and clean tracks."""
    tracks = {
        "zeta": [0.0, 0.0, 0.01, 0.0, 0.0],
        "alpha": [0.0, 0.0, 0.7, 0.0, 0.0],
        "mu": [0.0, 0.0, -0.01, 0.0, 0.0],
    }

    result = declick_batch(tracks, {**CONFIG, "threshold": 0.3})

    assert list(result) == ["zeta", "alpha", "mu"]
    assert set(result) == set(tracks)
    assert result["zeta"] == pytest.approx(tracks["zeta"], abs=1e-9)
    assert result["mu"] == pytest.approx(tracks["mu"], abs=1e-9)
