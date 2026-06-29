"""
Numerically-stable benchmark statistics and adaptive stopping criteria.

This module is a Python port of NVBench's ``nvbench/detail/statistics.cuh`` and
its stopping criteria (``stdrel``, ``sample-count``, ``entropy``). It provides
the "statistical rigor" layer that turns a raw list of per-iteration timings
into a trustworthy measurement:

* :class:`OnlineMeanVariance` -- Welford recurrence (no catastrophic
  cancellation) with Chan parallel ``merge``.
* :func:`compute_quartiles` / :func:`compute_robust_noise` -- order statistics
  (median / IQR) and the outlier-resistant relative-IQR noise.
* :func:`compute_relative_dispersion` -- the coefficient of variation, i.e. the
  "Noise" column in an NVBench table.
* :class:`StdRelCriterion`, :class:`SampleCountCriterion`,
  :class:`EntropyCriterion` -- swappable adaptive-stopping rules.
* :class:`BenchmarkStatistics` -- the per-measurement summary bundle (classic
  mean/stdev/CV plus robust median/IQR), the analog of one NVBench cold-time row.

The C++ originals are licensed Apache-2.0 WITH LLVM-exception (Copyright NVIDIA).
The algorithms (Welford increment, Bessel correction, nearest-rank percentiles,
incremental Shannon entropy, sliding-window OLS) are reproduced faithfully so the
numbers match NVBench; see the reference comments on each piece.

Intentional departure from the repo-global "scripts use Pydantic/Typer" rule:
this is an internal numeric library module on the hot benchmarking path, so it
stays ``numpy`` + stdlib ``dataclasses`` to match ``flashinfer/testing`` idioms
and avoid importing ``torch`` (keeps the math independently importable/testable).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Below this many samples NVBench refuses to estimate noise (statistics.cuh:55).
MIN_SAMPLES_FOR_NOISE_ESTIMATE = 5


# ---------------------------------------------------------------------------
# Descriptive statistics (statistics.cuh free functions)
# ---------------------------------------------------------------------------
def percentile_rank(percentile: int, size: int) -> int:
    """Nearest-rank index: ``round(p / 100 * (size - 1))`` (statistics.cuh:209)."""
    if size <= 0:
        raise ValueError("percentile_rank requires a non-empty sample set")
    p = min(max(percentile, 0), 100)
    q = p / 100.0
    return int(round(q * (size - 1)))


def compute_percentiles(
    samples: Sequence[float], percentiles: Sequence[int]
) -> List[float]:
    """Nearest-rank percentiles by sorting (statistics.cuh:219)."""
    if len(samples) == 0:
        return [math.nan] * len(percentiles)
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    n = ordered.shape[0]
    return [float(ordered[percentile_rank(p, n)]) for p in percentiles]


def compute_quartiles(samples: Sequence[float]) -> Tuple[float, float, float]:
    """Return ``(q1, median, q3)`` via nearest-rank ranks (statistics.cuh:316).

    Mirrors NVBench's complexity-aware path: full sort for small inputs, partial
    selection (``np.partition`` ~ ``std::nth_element``) once the sample count
    crosses ``selection_threshold`` so the cost stays O(n) rather than O(n log n).
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
    """``dispersion / center`` (coefficient of variation), guarded.

    Returns ``None`` for a non-positive / non-finite center or a negative /
    NaN dispersion (statistics.cuh:334). ``+inf`` is intentionally allowed --
    it means unbounded relative dispersion, not missing data.
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
    """Relative IQR ``(q3 - q1) / median`` (statistics.cuh:346)."""
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
    """Outlier-resistant noise = relative IQR; ``None`` below the sample floor."""
    if num_samples < MIN_SAMPLES_FOR_NOISE_ESTIMATE:
        return None
    return compute_relative_interquartile_range(first_quartile, median, third_quartile)


def slope_to_degrees(slope: float) -> float:
    """``atan2(slope, 1)`` in degrees (statistics.cuh:474)."""
    return math.degrees(math.atan2(slope, 1.0))


# ---------------------------------------------------------------------------
# Welford online mean / variance (statistics.cuh:105)
# ---------------------------------------------------------------------------
class OnlineMeanVariance:
    """Numerically-stable running mean and variance.

    Keeps the *biased* (MLE) population variance via the Welford recurrence and
    applies Bessel's correction on read, exactly as NVBench does. ``merge``
    implements Chan's parallel combination so two partial summaries fuse without
    re-scanning. This avoids the catastrophic cancellation of the naive
    ``E[x^2] - E[x]^2`` formula -- which matters precisely in the stable-kernel
    regime where the variance is tiny relative to the mean.
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

    def merge(self, other: "OnlineMeanVariance") -> None:
        """Chan parallel combine (statistics.cuh:161)."""
        if other._size == 0:
            return
        if self._size == 0:
            self._size = other._size
            self._mean = other._mean
            self._variance = other._variance
            return

        self._size += other._size
        f = other._size / self._size
        diff = other._mean - self._mean
        self._mean += f * diff
        diff2 = diff * diff
        # var arg is (self.var - other.var), captured before mutation.
        var_arg = self._variance - other._variance
        self._variance += f * ((diff2 - var_arg) - f * diff2)

    @property
    def size(self) -> int:
        return self._size

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def sample_variance(self) -> float:
        """Biased (MLE) variance, as accumulated."""
        return self._variance

    @property
    def unbiased_variance(self) -> float:
        """Bessel-corrected variance ``var / (1 - 1/n)``; NaN if undefined."""
        if self._size <= 1 or self._variance < 0.0:
            return math.nan
        f = 1.0 / self._size
        return self._variance / (1.0 - f)


# ---------------------------------------------------------------------------
# Sliding-window online linear regression (online_linear_regression.cuh)
# ---------------------------------------------------------------------------
class OnlineLinearRegression:
    """Incremental OLS supporting append and ring-buffer slide.

    Used by :class:`EntropyCriterion` to fit the cumulative-entropy curve over a
    sliding window. ``slide_window`` keeps the running cross-product correct when
    the oldest point is evicted and a new one appended, with the window's
    x-values held fixed at ``0..(window-1)`` -- the same trick as NVBench's
    ``online_linear_regression::slide_window``.
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
        """Evict oldest ``y_out``, append ``y_in``; x-values stay 0..count-1."""
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


# ---------------------------------------------------------------------------
# Measurement summary bundle
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkStatistics:
    """One measurement's worth of summary stats -- an NVBench cold-time row.

    ``noise`` is the coefficient of variation (classic, mean-based).
    ``relative_iqr`` is the robust, outlier-resistant analog. Both are ``None``
    when there are too few samples (< ``MIN_SAMPLES_FOR_NOISE_ESTIMATE``) or the
    center is degenerate, so a low-confidence number is never silently reported.
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
        noise = "n/a" if self.noise is None else f"{self.noise * 100:.2f}%"
        riqr = "n/a" if self.relative_iqr is None else f"{self.relative_iqr * 100:.2f}%"
        return (
            f"median {self.median:.4f} {unit}; noise {noise}; "
            f"relIQR {riqr}; mean {self.mean:.4f} {unit}; "
            f"std {self.stdev:.4f} {unit}; n={self.num_samples}"
        )

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "mean": self.mean,
            "stdev": self.stdev,
            "noise": self.noise,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "relative_iqr": self.relative_iqr,
            "min": self.minimum,
            "max": self.maximum,
        }


# ---------------------------------------------------------------------------
# Stopping criteria (stopping_criterion.cuh interface)
# ---------------------------------------------------------------------------
class StoppingCriterion:
    """Interface: ``reset`` -> ``add_measurement`` (per sample) -> ``is_finished``.

    Measurements and any time parameters share the same unit (FlashInfer feeds
    milliseconds).
    """

    name = "base"

    def reset(self) -> None:
        raise NotImplementedError

    def add_measurement(self, measurement: float) -> None:
        raise NotImplementedError

    def is_finished(self) -> bool:
        raise NotImplementedError


# Tolerate transient invalid noise estimates but terminate after this many
# consecutive ones (stdrel_criterion.cxx:32).
_INVALID_NOISE_ESTIMATE_LIMIT = 64


class StdRelCriterion(StoppingCriterion):
    """Converge the relative standard deviation (NVBench default ``stdrel``).

    Stops once the relative stdev (noise) has dropped below ``max_noise`` *and*
    at least ``min_time`` of accumulated measured time has elapsed. Two escape
    hatches keep inherently-noisy or degenerate kernels from sampling forever:

    * **Noise-stability fallback** -- after > 64 noise values, every 16 samples,
      if the relative stdev of the noise series itself is < 5%, declare the noise
      plateaued and stop (a convergence test on the second moment of the second
      moment).
    * **Invalid-estimate termination** -- 64 consecutive non-finite noise
      estimates (zero-time / degenerate kernels) -> stop.

    Defaults mirror ``stdrel_criterion.cxx``: ``max_noise=0.005`` (0.5%) and
    ``min_time=0.5`` (NVBench seconds). FlashInfer measures in milliseconds, so
    its driver passes ``min_time=500.0`` to preserve the 0.5 s semantics.
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

    @property
    def current_noise(self) -> Optional[float]:
        return self._noise_tracker[-1] if self._noise_tracker else None


class SampleCountCriterion(StoppingCriterion):
    """Deterministic N samples (sample_count_criterion.cxx).

    Trades statistical adaptivity for run-to-run reproducibility -- the right
    choice when a CI job needs a fixed sample count.
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
    """Information-theoretic convergence (entropy_criterion.cxx).

    Instead of "is the variance small?" it asks "is the sample still telling me
    anything new?": maintain a frequency histogram, track Shannon entropy
    ``H = log2(n) - (1/n) * sum c_i log2 c_i`` (updated incrementally), fit an
    online OLS over a sliding window of the cumulative-entropy curve, and stop
    when that curve has gone flat -- slope angle < ``max_angle`` (0.048 deg) and
    fit quality ``R^2`` > ``min_r2`` (0.36). Catches multi-modal distributions
    that ``stdrel`` declares converged too early, and stops earlier on clean
    unimodal kernels.
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
    """Build a stopping criterion by name (``stdrel`` / ``sample-count`` /
    ``entropy``). Unknown ``params`` for a criterion are ignored, matching
    NVBench's ``criterion_params::set_from``."""
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
