"""Numerically stable statistics for adaptive benchmark timing.

Turns a stream of per-iteration timings into a trustworthy measurement. The
algorithms are ported from NVBench (Apache-2.0 WITH LLVM-exception, Copyright
NVIDIA):

- ``OnlineMeanVariance``: running mean and variance via Welford's recurrence,
  which avoids the cancellation error of the naive ``E[x^2] - E[x]^2`` formula.
- ``compute_quartiles`` / ``compute_robust_noise``: order statistics and the
  outlier-resistant relative IQR.
- ``StdRelCriterion`` / ``SampleCountCriterion`` / ``EntropyCriterion``:
  interchangeable rules that decide when enough samples have been collected.
- ``BenchmarkStatistics``: the summary bundle returned to callers.

The module is pure numpy + stdlib (no torch) so the math stays importable and
testable on its own.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Noise is not estimated below this many samples.
MIN_SAMPLES_FOR_NOISE_ESTIMATE = 5


# Descriptive statistics
def percentile_rank(percentile: int, size: int) -> int:
    """Return the nearest-rank index of ``percentile`` over ``size`` samples."""
    if size <= 0:
        raise ValueError("percentile_rank requires a non-empty sample set")
    p = min(max(percentile, 0), 100)
    q = p / 100.0
    return int(round(q * (size - 1)))


def compute_quartiles(samples: Sequence[float]) -> Tuple[float, float, float]:
    """Return ``(q1, median, q3)`` by nearest rank.

    Uses a full sort for small inputs and ``np.partition`` (O(n) selection) once
    the sample count is large.
    """
    arr = np.asarray(samples, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return (math.nan, math.nan, math.nan)

    selection_threshold = 4096
    r25 = percentile_rank(25, n)
    r50 = percentile_rank(50, n)
    r75 = percentile_rank(75, n)

    if n >= selection_threshold:
        # np.partition guarantees arr[k] is the k-th order statistic, which is
        # exactly what nearest-rank needs (and what std::nth_element provides).
        part = np.partition(arr, (r25, r50, r75))
        return (float(part[r25]), float(part[r50]), float(part[r75]))

    ordered = np.sort(arr)
    return (float(ordered[r25]), float(ordered[r50]), float(ordered[r75]))


def compute_relative_dispersion(dispersion: float, center: float) -> Optional[float]:
    """Return ``dispersion / center`` (the coefficient of variation).

    Returns ``None`` when the ratio would be meaningless: a non-positive or
    non-finite center, or a negative/NaN dispersion. A ``+inf`` ratio is kept --
    it means unbounded dispersion, not missing data.
    """
    if (
        not (center > 0.0)
        or not math.isfinite(center)
        or dispersion < 0.0
        or math.isnan(dispersion)
    ):
        return None
    return dispersion / center


def compute_relative_interquartile_range(
    first_quartile: float, median: float, third_quartile: float
) -> Optional[float]:
    """Return the IQR relative to the median, ``(q3 - q1) / median``."""
    iqr = third_quartile - first_quartile
    if not math.isfinite(iqr):
        return None
    return compute_relative_dispersion(iqr, median)


def compute_robust_noise(
    num_samples: int,
    first_quartile: float,
    median: float,
    third_quartile: float,
) -> Optional[float]:
    """Return the relative IQR as an outlier-resistant noise estimate.

    Returns ``None`` below ``MIN_SAMPLES_FOR_NOISE_ESTIMATE`` samples.
    """
    if num_samples < MIN_SAMPLES_FOR_NOISE_ESTIMATE:
        return None
    return compute_relative_interquartile_range(first_quartile, median, third_quartile)


def slope_to_degrees(slope: float) -> float:
    """Return the angle (in degrees) of a line with the given slope."""
    return math.degrees(math.atan2(slope, 1.0))


# Online mean / variance
class OnlineMeanVariance:
    """Running mean and variance via Welford's recurrence.

    Accumulates the biased (population) variance incrementally and applies
    Bessel's correction on read, which stays accurate even when the variance is
    tiny relative to the mean. ``merge`` combines two independent accumulators
    without rescanning their inputs.
    """

    __slots__ = ("_size", "_mean", "_variance")

    def __init__(self) -> None:
        self._size: int = 0
        self._mean: float = 0.0
        self._variance: float = 0.0  # biased (population) variance

    def update(self, measurement: float) -> None:
        self._size += 1
        if self._size > 2:
            f = 1.0 / self._size
            diff = measurement - self._mean
            # mu_{n} = mu_{n-1} + diff / n
            self._mean += f * diff
            diff2 = diff * diff
            # var_{n} = var_{n-1} + (((n-1)/n) * diff^2 - var_{n-1}) / n
            self._variance += f * ((diff2 - self._variance) - f * diff2)
        elif self._size == 2:
            x1 = self._mean
            x2 = measurement
            self._mean = 0.5 * (x1 + x2)
            half_diff = 0.5 * (x1 - x2)
            self._variance = half_diff * half_diff
        else:
            self._mean = measurement  # variance stays 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def sample_variance(self) -> float:
        """Biased (population) variance."""
        return self._variance

    @property
    def unbiased_variance(self) -> float:
        """Bessel-corrected variance, or NaN with fewer than two samples."""
        if self._size <= 1 or self._variance < 0.0:
            return math.nan
        f = 1.0 / self._size
        return self._variance / (1.0 - f)


# Online linear regression
class OnlineLinearRegression:
    """Incremental ordinary least squares over a sliding window.

    Used by :class:`EntropyCriterion` to track the slope of the
    cumulative-entropy curve. ``slide_window`` evicts the oldest point and
    appends a new one while keeping the window's x-values fixed at
    ``0..count-1``.
    """

    __slots__ = ("_sum_x", "_sum_y", "_sum_xy", "_sum_x2", "_sum_y2", "_count")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0
        self._count = 0

    def update(self, x: float, y: float) -> None:
        self._sum_x += x
        self._sum_y += y
        self._sum_xy += x * y
        self._sum_x2 += x * x
        self._sum_y2 += y * y
        self._count += 1

    def slide_window(self, y_out: float, y_in: float) -> None:
        """Replace the oldest sample ``y_out`` with ``y_in`` (x-values unchanged)."""
        self._sum_y -= y_out
        self._sum_y += y_in

        self._sum_y2 -= y_out * y_out
        self._sum_y2 += y_in * y_in

        # NOTE: uses the already-updated _sum_y, matching the C++ exactly.
        self._sum_xy -= self._sum_y - y_in
        self._sum_xy += (self._count - 1.0) * y_in

    @property
    def count(self) -> int:
        return self._count

    def slope(self) -> float:
        if self._count < 2:
            return math.nan
        n = float(self._count)
        mean_x = self._sum_x / n
        mean_y = self._sum_y / n
        numerator = (self._sum_xy / n) - mean_x * mean_y
        denominator = (self._sum_x2 / n) - mean_x * mean_x
        if abs(denominator) < 1e-12:
            return math.nan
        return numerator / denominator

    def intercept(self) -> float:
        if self._count < 2:
            return math.nan
        s = self.slope()
        if not math.isfinite(s):
            return math.nan
        n = float(self._count)
        return (self._sum_y / n) - s * (self._sum_x / n)

    def r_squared(self) -> float:
        if self._count < 2:
            return math.nan
        n = float(self._count)
        mean_y = self._sum_y / n
        ss_tot = (self._sum_y2 / n) - mean_y * mean_y
        if ss_tot < np.finfo(np.float64).eps:
            return 1.0
        s = self.slope()
        b = self.intercept()
        if not math.isfinite(s) or not math.isfinite(b):
            return math.nan
        mean_xy = self._sum_xy / n
        mean_xx = self._sum_x2 / n
        mean_x = self._sum_x / n
        ss_tot_m_res = (
            s * ((mean_xy - s * mean_xx) + (mean_xy - b * mean_x))
            + b * (mean_y - s * mean_x - b)
            + mean_y * (b - mean_y)
        )
        return min(max(ss_tot_m_res / ss_tot, 0.0), 1.0)


# Summary bundle
@dataclass(frozen=True)
class BenchmarkStatistics:
    """Summary statistics for one set of timing samples.

    ``noise`` is the coefficient of variation (stdev / mean); ``relative_iqr``
    is its outlier-resistant counterpart. Both are ``None`` when there are too
    few samples or the center is degenerate, so a low-confidence figure is never
    reported as if it were solid.
    """

    num_samples: int
    mean: float
    stdev: float
    noise: Optional[float]  # coefficient of variation = stdev / mean
    median: float
    q1: float
    q3: float
    iqr: float
    relative_iqr: Optional[float]  # robust noise = (q3 - q1) / median
    minimum: float
    maximum: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "BenchmarkStatistics":
        """Build a summary from raw timing samples (empty input yields all-NaN)."""
        arr = np.asarray(samples, dtype=np.float64)
        n = int(arr.shape[0])
        if n == 0:
            nan = math.nan
            return cls(0, nan, nan, None, nan, nan, nan, nan, None, nan, nan)

        mean = float(arr.mean())
        # Bessel-corrected (ddof=1) stdev == sqrt(unbiased variance); inf below
        # the noise floor, matching statistics.cuh::standard_deviation.
        if n >= MIN_SAMPLES_FOR_NOISE_ESTIMATE:
            stdev = float(arr.std(ddof=1))
            noise = compute_relative_dispersion(stdev, mean)
        else:
            stdev = float(arr.std(ddof=1)) if n >= 2 else math.inf
            noise = None

        q1, median, q3 = compute_quartiles(arr)
        relative_iqr = compute_robust_noise(n, q1, median, q3)
        return cls(
            num_samples=n,
            mean=mean,
            stdev=stdev,
            noise=noise,
            median=median,
            q1=q1,
            q3=q3,
            iqr=q3 - q1,
            relative_iqr=relative_iqr,
            minimum=float(arr.min()),
            maximum=float(arr.max()),
        )

    def summary_str(self, unit: str = "ms") -> str:
        """Return a compact one-line summary for logging."""
        noise = "n/a" if self.noise is None else f"{self.noise * 100:.2f}%"
        riqr = "n/a" if self.relative_iqr is None else f"{self.relative_iqr * 100:.2f}%"
        return (
            f"median {self.median:.4f} {unit}; noise {noise}; "
            f"relIQR {riqr}; mean {self.mean:.4f} {unit}; "
            f"std {self.stdev:.4f} {unit}; n={self.num_samples}"
        )


# Stopping criteria
class StoppingCriterion:
    """Base class for adaptive stopping rules.

    Lifecycle per run: ``reset()``, then ``add_measurement()`` once per sample,
    checking ``is_finished()`` after each. Measurements use whatever time unit
    the caller provides (FlashInfer uses milliseconds).
    """

    name = "base"

    def reset(self) -> None:
        raise NotImplementedError

    def add_measurement(self, measurement: float) -> None:
        raise NotImplementedError

    def is_finished(self) -> bool:
        raise NotImplementedError


# Stop after this many consecutive non-finite noise estimates (e.g. a kernel
# that reports zero time).
_INVALID_NOISE_ESTIMATE_LIMIT = 64


class StdRelCriterion(StoppingCriterion):
    """Stop when the relative standard deviation is small and stable.

    Finishes once the noise (relative stdev) drops below ``max_noise`` and at
    least ``min_time`` of measured time has accumulated. Two fallbacks keep an
    inherently noisy kernel from sampling forever: if the noise itself plateaus
    (its own relative stdev stays under 5% across a window of samples) the
    current value is accepted, and if the noise estimate is non-finite for too
    many samples in a row it stops anyway.

    ``min_time`` shares the unit of the measurements, so the driver passes 500.0
    for NVBench's 0.5 s lower bound in milliseconds.
    """

    name = "stdrel"

    def __init__(self, max_noise: float = 0.005, min_time: float = 0.5) -> None:
        self.max_noise = max_noise
        self.min_time = min_time
        self.reset()

    def reset(self) -> None:
        self._summary = OnlineMeanVariance()
        self._consecutive_invalid = 0
        self._noise_tracker: List[float] = []

    def add_measurement(self, measurement: float) -> None:
        self._summary.update(measurement)
        if self._summary.size < MIN_SAMPLES_FOR_NOISE_ESTIMATE:
            return
        dispersion = math.sqrt(self._summary.unbiased_variance)
        noise = compute_relative_dispersion(dispersion, self._summary.mean)
        if noise is not None and math.isfinite(noise):
            self._consecutive_invalid = 0
            self._noise_tracker.append(noise)
        else:
            self._consecutive_invalid += 1

    def is_finished(self) -> bool:
        if self._consecutive_invalid >= _INVALID_NOISE_ESTIMATE_LIMIT:
            return True

        total_measured_time = self._summary.mean * self._summary.size
        if total_measured_time <= self.min_time:
            return False
        if not self._noise_tracker:
            return False
        if self._consecutive_invalid != 0:
            return False

        if self._noise_tracker[-1] < self.max_noise:
            return True

        min_noise_stability_window = 64
        noise_stability_check_interval = 16
        do_check = (
            len(self._noise_tracker) > min_noise_stability_window
            and self._summary.size % noise_stability_check_interval == 0
        )
        if do_check:
            noise_summary = OnlineMeanVariance()
            for v in self._noise_tracker:
                noise_summary.update(v)
            if math.isfinite(noise_summary.mean) and math.isfinite(
                noise_summary.sample_variance
            ):
                noise_threshold = 0.05
                mean_scaled = noise_summary.mean * noise_threshold
                variance_threshold = mean_scaled * mean_scaled
                if noise_summary.sample_variance < variance_threshold:
                    return True
        return False


class SampleCountCriterion(StoppingCriterion):
    """Stop after a fixed number of samples.

    Trades adaptivity for run-to-run reproducibility, which is usually what a CI
    job wants.
    """

    name = "sample-count"

    def __init__(self, target_samples: int = 100) -> None:
        if target_samples <= 0:
            raise ValueError("target_samples must be greater than zero")
        self.target_samples = target_samples
        self.reset()

    def reset(self) -> None:
        self._total_samples = 0

    def add_measurement(self, measurement: float) -> None:
        self._total_samples += 1

    def is_finished(self) -> bool:
        return self._total_samples >= self.target_samples


class EntropyCriterion(StoppingCriterion):
    """Stop when the timing distribution stops revealing new information.

    Tracks the Shannon entropy of the observed timings and fits a line to the
    cumulative-entropy curve over a sliding window. When that curve flattens
    (slope below ``max_angle`` with fit quality above ``min_r2``) the sample is
    considered converged. Handles multi-modal distributions that the
    relative-stdev rule can declare converged too early.
    """

    name = "entropy"

    def __init__(
        self,
        max_angle: float = 0.048,
        min_r2: float = 0.36,
        window: int = 299,
    ) -> None:
        self.max_angle = max_angle
        self.min_r2 = min_r2
        self.window = window
        self.reset()

    def reset(self) -> None:
        self._total_samples = 0
        self._total_time = 0.0
        self._freq_keys: List[float] = []
        self._freq_counts: List[int] = []
        self._sum_count_log_counter = 0.0
        self._entropy_tracker: List[float] = []  # ring buffer of size `window`
        self._regression = OnlineLinearRegression()

    def _update_entropy_sum(self, old_count: float, new_count: float) -> None:
        if old_count > 0:
            diff = new_count - old_count
            self._sum_count_log_counter += new_count * math.log2(
                1 + diff / old_count
            ) + diff * math.log2(old_count)
        else:
            self._sum_count_log_counter += new_count * math.log2(new_count)

    def _compute_entropy(self) -> float:
        if self._total_samples == 0:
            return 0.0
        n = float(self._total_samples)
        entropy = math.log2(n) - self._sum_count_log_counter / n
        return max(0.0, entropy)

    def add_measurement(self, measurement: float) -> None:
        self._total_samples += 1
        self._total_time += measurement

        key = measurement  # bin_keys is false in NVBench
        idx = bisect.bisect_left(self._freq_keys, key)
        if idx < len(self._freq_keys) and self._freq_keys[idx] == key:
            old_count = self._freq_counts[idx]
            self._freq_counts[idx] += 1
        else:
            old_count = 0
            self._freq_keys.insert(idx, key)
            self._freq_counts.insert(idx, 1)

        self._update_entropy_sum(float(old_count), float(old_count + 1))
        entropy = self._compute_entropy()

        # x for an appended point is the current window size (0-based index).
        n = len(self._entropy_tracker)
        if n == self.window:
            old_entropy = self._entropy_tracker[0]
            self._regression.slide_window(old_entropy, entropy)
            self._entropy_tracker.pop(0)
        else:
            self._regression.update(float(n), entropy)
        self._entropy_tracker.append(entropy)

    def is_finished(self) -> bool:
        if len(self._entropy_tracker) < 2:
            return False
        if self._total_samples % 2 != 0:
            return False
        slope = self._regression.slope()
        if not math.isfinite(slope):
            return False
        if slope_to_degrees(slope) > self.max_angle:
            return False
        r2 = self._regression.r_squared()
        if not math.isfinite(r2):
            return False
        if r2 < self.min_r2:
            return False
        return True


def make_criterion(name: str, **params) -> StoppingCriterion:
    """Construct a stopping criterion by name.

    ``name`` is one of ``stdrel``, ``sample-count``, or ``entropy``. Keyword
    parameters that do not apply to the chosen criterion are ignored.
    """
    name = name.replace("_", "-").lower()
    if name == "stdrel":
        return StdRelCriterion(
            max_noise=params.get("max_noise", 0.005),
            min_time=params.get("min_time", 0.5),
        )
    if name in ("sample-count", "fixed"):
        return SampleCountCriterion(target_samples=params.get("target_samples", 100))
    if name == "entropy":
        return EntropyCriterion(
            max_angle=params.get("max_angle", 0.048),
            min_r2=params.get("min_r2", 0.36),
            window=params.get("window", 299),
        )
    raise ValueError(
        f"unknown stopping criterion {name!r}; "
        "expected one of: stdrel, sample-count, entropy"
    )
