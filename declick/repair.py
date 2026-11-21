"""Click detection and repair primitives for old audio recordings.

All audio is represented as ``list[float]`` with nominal samples in the
closed interval ``[-1.0, 1.0]``.  Functions are deterministic: they use no
randomness, no sleeping, and no third-party packages.  Unless a function says
otherwise, numeric assertions use an absolute tolerance of ``1e-6``.
"""

from __future__ import annotations

import math


class DeclickError(ValueError):
    """Raised when audio, click intervals, or configuration are invalid."""


_TOL = 1e-6
_METHODS = ("linear", "ar")
_CHANGE_TOL = 1e-12


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    """True for real integers (``bool`` is not accepted as an index)."""

    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_threshold(value: object) -> float:
    if not _is_real(value) or not math.isfinite(float(value)):
        raise DeclickError(f"threshold must be a finite number, got {value!r}")
    if float(value) <= 0.0:
        raise DeclickError(f"threshold must be positive, got {value!r}")
    return float(value)


def _check_positive_int(value: object, name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise DeclickError(f"{name} must be a positive integer, got {value!r}")
    return value


def _check_method(value: object) -> str:
    if not isinstance(value, str) or value not in _METHODS:
        raise DeclickError(
            f"method must be one of {_METHODS}, got {value!r}"
        )
    return value


def _check_max_db(value: object) -> float:
    if not _is_real(value) or not math.isfinite(float(value)):
        raise DeclickError(f"max_db must be a finite number, got {value!r}")
    if float(value) < 0.0:
        raise DeclickError(f"max_db must be non-negative, got {value!r}")
    return float(value)


def _check_samples(samples: object, name: str) -> list[float]:
    if not isinstance(samples, list):
        raise DeclickError(f"{name} must be a list of samples")
    for position, value in enumerate(samples):
        if not _is_real(value) or not math.isfinite(float(value)):
            raise DeclickError(
                f"{name}[{position}] must be a finite number, got {value!r}"
            )
        if value < -1.0 - _TOL or value > 1.0 + _TOL:
            raise DeclickError(
                f"{name}[{position}]={value!r} is outside the nominal "
                "sample range [-1.0, 1.0]"
            )
    return samples


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def second_diff(samples: list[float]) -> list[float]:
    """Return the second difference used to expose sudden sample jumps.

    For ``n = len(samples)`` the result contains ``max(0, n - 2)`` values and
    is defined by::

        out[i - 1] = samples[i + 1] - 2.0 * samples[i] + samples[i - 1]

    for ``i`` from ``1`` through ``n - 2``.  Thus an isolated upward impulse
    produces a positive left flank, a negative centre and a positive right
    flank; the sign/direction must not be reversed.  Inputs with fewer than
    three samples return ``[]``.  The input list is never modified.  Samples
    are assumed to lie in ``[-1.0, 1.0]``; comparisons use ``1e-6``.
    """
    if len(samples) < 3:
        return []
    return [
        samples[i + 1] - 2.0 * samples[i] + samples[i - 1]
        for i in range(1, len(samples) - 1)
    ]


def _detection_profile(samples: list[float]) -> list[float]:
    """Absolute second differences aligned onto sample indices.

    The two left-hand values are padding so the array has the same length as
    ``samples`` and every entry can be addressed by a sample index.
    """

    if len(samples) < 3:
        return [0.0] * len(samples)
    return [0.0, 0.0] + [abs(value) for value in second_diff(samples)]


def detect_clicks(samples: list[float], threshold: float, min_gap: int) -> list[int]:
    """Return one sample index per detected click cluster.

    A sample index ``i`` is a candidate when
    ``abs(second_diff(samples)[i - 1]) > threshold`` (strictly greater).  A
    non-positive or non-finite ``threshold`` raises :class:`DeclickError`;
    ``min_gap`` must be a positive integer or :class:`DeclickError` is raised.
    Candidates whose sample-index distance is at most ``min_gap`` belong to
    the same cluster (clusters are transitive).  Each cluster contributes
    exactly one index: the index with the largest absolute second difference,
    with ties resolved to the smallest index.  The result is sorted ascending
    and is ``[]`` when nothing exceeds the threshold.  Samples are in
    ``[-1.0, 1.0]`` and comparisons use ``1e-6``.
    """
    threshold = _check_threshold(threshold)
    min_gap = _check_positive_int(min_gap, "min_gap")

    profile = _detection_profile(samples)
    candidates = [i for i, value in enumerate(profile) if value > threshold]
    if not candidates:
        return []

    clusters: list[list[int]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        if candidate - clusters[-1][-1] <= min_gap:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    clicks: list[int] = []
    for cluster in clusters:
        best = cluster[0]
        for index in cluster[1:]:
            # Strict comparison keeps the smallest index on ties.
            if profile[index] > profile[best]:
                best = index
        clicks.append(best)
    return clicks


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def repair_point(samples: list[float], idx: int, span: int) -> list[float]:
    """Return a copy with the single sample at ``idx`` interpolated.

    The replacement is the two-sided linear interpolation, evaluated at
    ``idx``, between the untouched anchors ``samples[idx - span]`` and
    ``samples[idx + span]``.  ``span`` must be a positive integer and both
    anchors must exist; otherwise, or when ``idx`` is out of range,
    :class:`DeclickError` is raised.  Every other sample is copied unchanged
    and the input list is not mutated.  This intentionally avoids the
    previous-value fill that creates a DC step.  Samples are in ``[-1.0, 1.0]``
    and comparisons use ``1e-6``.
    """
    span = _check_positive_int(span, "span")
    if not _is_int(idx):
        raise DeclickError(f"idx must be an integer, got {idx!r}")
    if idx < 0 or idx >= len(samples):
        raise DeclickError(
            f"idx {idx} is out of range for {len(samples)} samples"
        )
    left = idx - span
    right = idx + span
    if left < 0 or right >= len(samples):
        raise DeclickError(
            f"span {span} around idx {idx} needs anchors at {left} and "
            f"{right}, which do not both exist"
        )

    left_value = samples[left]
    right_value = samples[right]
    fraction = (idx - left) / float(right - left)
    replacement = left_value + (right_value - left_value) * fraction

    result = list(samples)
    result[idx] = replacement
    return result


def repair_span(
    samples: list[float],
    start: int,
    end: int,
    method: str,
) -> list[float]:
    """Repair the inclusive interval ``[start, end]`` without a new seam.

    Both exterior anchors (``start - 1`` and ``end + 1``) must exist, so
    ``1 <= start <= end <= len(samples) - 2``; otherwise
    :class:`DeclickError` is raised.  The returned list has the same length,
    keeps both exterior anchors unchanged, and is a new list.

    Supported methods are exactly ``"linear"`` and ``"ar"``; any other value
    raises :class:`DeclickError` instead of silently falling back to linear.

    * ``"linear"`` fills the inclusive interval with evenly spaced values on
      the straight line joining the two exterior anchors.
    * ``"ar"`` uses a deterministic AR(1)-style continuation from the left
      anchor and applies the affine correction needed to meet the right
      anchor, so the exact interior curve may vary but both endpoints remain
      continuous.

    For both methods the jumps at ``start - 1 -> start`` and
    ``end -> end + 1`` are bounded by the anchor-to-anchor step, preventing
    the repaired block from introducing a fresh click at a seam.  Samples are
    in ``[-1.0, 1.0]`` and numerical comparisons use ``1e-6``.
    """
    method = _check_method(method)
    if not _is_int(start) or not _is_int(end):
        raise DeclickError(
            f"start and end must be integers, got {start!r} and {end!r}"
        )
    if start > end:
        raise DeclickError(f"start {start} must not exceed end {end}")
    if start < 1 or end > len(samples) - 2:
        raise DeclickError(
            f"interval [{start}, {end}] is outside the repairable interior "
            f"[1, {len(samples) - 2}]"
        )

    left_value = samples[start - 1]
    right_value = samples[end + 1]
    count = end - start + 1
    step = (right_value - left_value) / (count + 1)

    if method == "linear":
        replacements = [
            left_value + step * offset for offset in range(1, count + 1)
        ]
    else:
        # AR(1)-style geometric continuation from the left anchor, then an
        # affine correction that lands exactly on the right anchor.  Both
        # seam weights are then clamped to one linear step, so neither seam
        # can receive a jump larger than the anchor-to-anchor step.
        coefficient = 0.5
        span = count + 1
        denominator = 1.0 - coefficient ** span
        weights = [
            (1.0 - coefficient ** offset) / denominator
            for offset in range(1, count + 1)
        ]
        if count == 1:
            weights = [0.5]
        else:
            seam_weight = 1.0 / span
            weights[0] = min(weights[0], seam_weight)
            weights[-1] = max(weights[-1], 1.0 - seam_weight)
            # Clamping must not fold the curve back on itself.
            for position in range(1, count):
                weights[position] = max(weights[position], weights[position - 1])
            weights[-1] = max(weights[-1], 1.0 - seam_weight)
        replacements = [
            left_value + (right_value - left_value) * weight
            for weight in weights
        ]

    result = list(samples)
    for offset, value in enumerate(replacements, start=1):
        result[start - 1 + offset] = value
    return result


def envelope_guard(
    samples: list[float],
    repaired: list[float],
    max_db: float,
) -> list[float]:
    """Limit the local RMS energy added by repairs.

    ``samples`` is the original track and ``repaired`` is the candidate track;
    both must have the same length or :class:`DeclickError` is raised.
    ``max_db`` must be finite and non-negative.  A contiguous changed run is
    a maximal range where ``abs(repaired[i] - samples[i]) > 1e-12``.  For each
    such run, the reference RMS is taken from up to two original samples on
    each side of the run (with a ``1e-6`` floor for digital silence).  If the
    run's RMS exceeds ``reference_rms * 10 ** (max_db / 20)``, the entire run
    is scaled down uniformly to that limit.  Unchanged samples are copied
    verbatim and the inputs are not mutated.  The returned list is new and
    has the same length; an empty input returns ``[]``.  Samples and repaired
    values are in ``[-1.0, 1.0]`` and checks use ``1e-6``.
    """
    max_db = _check_max_db(max_db)
    if len(samples) != len(repaired):
        raise DeclickError(
            f"samples and repaired differ in length: "
            f"{len(samples)} != {len(repaired)}"
        )

    result = list(repaired)
    if not samples:
        return result

    limit_factor = 10.0 ** (max_db / 20.0)
    index = 0
    while index < len(samples):
        if abs(repaired[index] - samples[index]) <= _CHANGE_TOL:
            index += 1
            continue

        run_start = index
        while (
            index < len(samples)
            and abs(repaired[index] - samples[index]) > _CHANGE_TOL
        ):
            index += 1
        run_end = index - 1

        reference_values = samples[max(0, run_start - 2):run_start]
        reference_values += samples[run_end + 1:min(len(samples), run_end + 3)]
        if reference_values:
            reference_rms = math.sqrt(
                sum(value * value for value in reference_values)
                / len(reference_values)
            )
        else:
            reference_rms = 0.0
        reference_rms = max(reference_rms, 1e-6)

        run_values = result[run_start:run_end + 1]
        run_rms = math.sqrt(
            sum(value * value for value in run_values) / len(run_values)
        )
        allowed_rms = reference_rms * limit_factor
        if run_rms > allowed_rms + 1e-12:
            scale = allowed_rms / run_rms
            for position in range(run_start, run_end + 1):
                result[position] = result[position] * scale

    return result


def clone_measure(original: list[float], repaired: list[float]) -> float:
    """Return the normalized amount of repair, from ``0.0`` to ``1.0``.

    The value is ``sqrt(mean(((original[i] - repaired[i]) / 2.0) ** 2))``.
    Because nominal samples live in ``[-1.0, 1.0]``, an unchanged signal
    scores ``0.0`` and a full inversion scores ``1.0``.  Mismatched lengths
    raise :class:`DeclickError`; two empty lists return ``0.0``.  The inputs
    are not modified and comparisons use ``1e-6``.
    """
    if len(original) != len(repaired):
        raise DeclickError(
            f"original and repaired differ in length: "
            f"{len(original)} != {len(repaired)}"
        )
    if not original:
        return 0.0
    total = 0.0
    for before, after in zip(original, repaired):
        difference = (before - after) / 2.0
        total += difference * difference
    return math.sqrt(total / len(original))


# ---------------------------------------------------------------------------
# Plan validation and batch entry point
# ---------------------------------------------------------------------------


def _normalise_interval(click: object) -> tuple[int, int] | None:
    if isinstance(click, (tuple, list)) and len(click) == 2:
        start, end = click
        if _is_int(start) and _is_int(end):
            return start, end
    return None


def validate(samples: list[float], clicks: list[tuple[int, int]]) -> list[str]:
    """Return human-readable problems, or ``[]`` when the plan is valid.

    Each click is an inclusive ``(start, end)`` sample interval.  The checks
    are, in order:

    * out of bounds: a malformed interval, ``start < 0``, ``end >= len``, or
      ``start > end``;
    * overlap: two valid intervals share at least one sample;
    * too dense: the union of valid click intervals covers more than
      ``max(2, len(samples) // 5)`` samples (more than 20%, with a small-clip
      floor), which would flatten the track rather than repair it.

    The function never raises for bad click data; it reports the problem.
    Messages contain the stable substrings ``"out of bounds"``,
    ``"overlap"``, and ``"too dense"``.  Empty samples with no clicks return
    ``[]``.  Samples are in ``[-1.0, 1.0]``, inputs are not modified, and
    numeric checks use ``1e-6``.
    """
    length = len(samples)
    problems: list[str] = []
    valid: list[tuple[int, int]] = []

    for click in clicks:
        interval = _normalise_interval(click)
        if interval is None:
            problems.append(f"click {click!r} out of bounds: malformed interval")
            continue
        start, end = interval
        if start < 0 or end >= length or start > end:
            problems.append(
                f"click ({start}, {end}) out of bounds for {length} samples"
            )
            continue
        valid.append((start, end))

    ordered = sorted(valid)
    widest = ordered[0] if ordered else None
    for current in ordered[1:]:
        # Compare against the widest interval seen so far, not merely the
        # previous neighbour, so nested overlaps are reported too.
        if current[0] <= widest[1]:
            problems.append(
                f"click ({current[0]}, {current[1]}) overlap with "
                f"({widest[0]}, {widest[1]})"
            )
        if current[1] > widest[1]:
            widest = current

    covered = 0
    union_start: int | None = None
    union_end = -1
    for start, end in ordered:
        if union_start is None:
            union_start, union_end = start, end
        elif start <= union_end + 1:
            union_end = max(union_end, end)
        else:
            covered += union_end - union_start + 1
            union_start, union_end = start, end
    if union_start is not None:
        covered += union_end - union_start + 1

    if covered > max(2, length // 5):
        problems.append(
            f"too dense: {covered} of {length} samples would be repaired"
        )

    return problems


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _parse_config(config: object) -> tuple[float, int, int, str, float]:
    if not isinstance(config, dict):
        raise DeclickError("config must be a mapping")
    required = ("threshold", "min_gap", "span", "method", "max_db")
    missing = [key for key in required if key not in config]
    if missing:
        raise DeclickError(f"config is missing keys: {', '.join(missing)}")
    return (
        _check_threshold(config["threshold"]),
        _check_positive_int(config["min_gap"], "min_gap"),
        _check_positive_int(config["span"], "span"),
        _check_method(config["method"]),
        _check_max_db(config["max_db"]),
    )


def _repair_track(
    samples: list[float],
    threshold: float,
    min_gap: int,
    span: int,
    method: str,
    max_db: float,
) -> list[float]:
    length = len(samples)
    if length < 3:
        return list(samples)

    clicks = detect_clicks(samples, threshold, min_gap)
    intervals: list[tuple[int, int]] = []
    for click in clicks:
        start = max(1, click - span)
        end = min(length - 2, click + span)
        if start <= end:
            intervals.append((start, end))

    repaired = list(samples)
    for start, end in _merge_intervals(intervals):
        repaired = repair_span(repaired, start, end, method)

    return envelope_guard(samples, repaired, max_db)


def declick_batch(
    tracks: dict[str, list[float]],
    config: dict,
) -> dict[str, list[float]]:
    """Repair several tracks atomically and preserve their insertion order.

    ``tracks`` maps track names to sample lists.  ``config`` must contain
    positive ``threshold``, positive integer ``min_gap``, positive integer
    ``span``, ``method`` equal to ``"linear"`` or ``"ar"``, and non-negative
    ``max_db``.  All configuration and every track are validated before any
    result is produced; one invalid track raises :class:`DeclickError` and
    nothing is partially returned or mutated.

    Each track gets its own detection array (no detection state may be shared
    between tracks).  Clicks are grouped with ``min_gap``, expanded by
    ``span``, clipped to the repairable interior, repaired with
    :func:`repair_span`, and finally checked with :func:`envelope_guard`.
    Tracks shorter than three samples are copied unchanged after detection.
    Samples are in ``[-1.0, 1.0]``.  The result is a new dict with the same key
    order and new value lists; an empty ``tracks`` mapping returns ``{}``.
    Numerical checks use ``1e-6``.
    """
    if not isinstance(tracks, dict):
        raise DeclickError("tracks must be a mapping of name to samples")

    threshold, min_gap, span, method, max_db = _parse_config(config)

    # Validate everything before touching a single track, so a bad input
    # cannot leave a partially repaired batch behind.
    for name, samples in tracks.items():
        if not isinstance(name, str):
            raise DeclickError(f"track name must be a string, got {name!r}")
        _check_samples(samples, f"track {name!r}")

    result: dict[str, list[float]] = {}
    for name, samples in tracks.items():
        result[name] = _repair_track(
            samples, threshold, min_gap, span, method, max_db
        )
    return result
