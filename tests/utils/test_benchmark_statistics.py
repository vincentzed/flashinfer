"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for the NVBench-inspired statistics port (flashinfer.testing.statistics)
and the adaptive bench_gpu_time_with_statistics driver. The math tests are
CPU-only (validated against numpy / brute-force); the timing test is
GPU-guarded.
"""

import math

import numpy as np
import pytest
import torch

from flashinfer.testing.statistics import (
    MIN_SAMPLES_FOR_NOISE_ESTIMATE,
    BenchmarkStatistics,
    EntropyCriterion,
    OnlineLinearRegression,
    OnlineMeanVariance,
    SampleCountCriterion,
    StdRelCriterion,
    compute_quartiles,
    compute_relative_dispersion,
    compute_robust_noise,
    make_criterion,
    percentile_rank,
)


# --------------------------------------------------------------------------- #
# Welford online mean / variance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e6])
def test_welford_matches_numpy(scale):
    rng = np.random.default_rng(0)
    x = rng.normal(100.0, scale, size=5000)
    omv = OnlineMeanVariance()
    for v in x:
        omv.update(float(v))
    assert omv.size == x.size
    assert math.isclose(omv.mean, float(x.mean()), rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(omv.unbiased_variance, float(x.var(ddof=1)), rel_tol=1e-7)


def test_welford_tiny_variance_no_cancellation():
    # Variance tiny relative to mean: where naive E[x^2]-E[x]^2 cancels.
    rng = np.random.default_rng(1)
    x = 1e8 + rng.normal(0.0, 1e-3, size=2000)
    omv = OnlineMeanVariance()
    for v in x:
        omv.update(float(v))
    assert math.isclose(omv.unbiased_variance, float(x.var(ddof=1)), rel_tol=1e-4)


def test_welford_merge_equals_single_pass():
    rng = np.random.default_rng(2)
    x = rng.normal(5.0, 2.0, size=1234)
    full = OnlineMeanVariance()
    for v in x:
        full.update(float(v))
    a, b = OnlineMeanVariance(), OnlineMeanVariance()
    for v in x[:500]:
        a.update(float(v))
    for v in x[500:]:
        b.update(float(v))
    a.merge(b)
    assert a.size == full.size
    assert math.isclose(a.mean, full.mean, rel_tol=1e-9)
    assert math.isclose(a.unbiased_variance, full.unbiased_variance, rel_tol=1e-7)


def test_welford_degenerate_sizes():
    omv = OnlineMeanVariance()
    assert math.isnan(omv.unbiased_variance)
    omv.update(3.0)
    assert omv.mean == 3.0
    assert math.isnan(omv.unbiased_variance)  # n == 1
    omv.update(5.0)
    assert math.isclose(omv.mean, 4.0)
    assert math.isclose(omv.unbiased_variance, 2.0)  # var of {3,5} ddof=1


# --------------------------------------------------------------------------- #
# Order statistics: quartiles / percentiles / robust noise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [10, 100, 4096, 10000])
def test_quartiles_match_nearest_rank(n):
    rng = np.random.default_rng(n)
    x = rng.normal(0.0, 1.0, size=n)
    q1, med, q3 = compute_quartiles(x)  # selection path when n >= 4096
    ordered = np.sort(x)
    expect = tuple(float(ordered[percentile_rank(p, n)]) for p in (25, 50, 75))
    assert np.allclose((q1, med, q3), expect)
    assert q1 <= med <= q3


def test_relative_dispersion_guards():
    assert compute_relative_dispersion(1.0, 0.0) is None  # non-positive center
    assert compute_relative_dispersion(1.0, -2.0) is None
    assert compute_relative_dispersion(-1.0, 2.0) is None  # negative dispersion
    assert compute_relative_dispersion(math.nan, 2.0) is None
    assert math.isclose(compute_relative_dispersion(2.0, 8.0), 0.25)


def test_robust_noise_sample_floor():
    # Below the floor -> None, regardless of values.
    assert compute_robust_noise(MIN_SAMPLES_FOR_NOISE_ESTIMATE - 1, 1, 2, 3) is None
    val = compute_robust_noise(MIN_SAMPLES_FOR_NOISE_ESTIMATE, 1.0, 2.0, 3.0)
    assert val is not None and math.isclose(val, (3.0 - 1.0) / 2.0)


# --------------------------------------------------------------------------- #
# BenchmarkStatistics bundle
# --------------------------------------------------------------------------- #
def test_benchmark_statistics_fields():
    rng = np.random.default_rng(3)
    x = rng.normal(100.0, 1.0, size=200)
    bs = BenchmarkStatistics.from_samples(x)
    assert bs.num_samples == 200
    assert math.isclose(bs.mean, float(x.mean()), rel_tol=1e-9)
    assert math.isclose(bs.stdev, float(x.std(ddof=1)), rel_tol=1e-9)
    assert math.isclose(bs.noise, bs.stdev / bs.mean, rel_tol=1e-9)
    assert math.isclose(bs.minimum, float(x.min()))
    assert math.isclose(bs.maximum, float(x.max()))
    assert bs.q1 <= bs.median <= bs.q3
    assert math.isclose(bs.iqr, bs.q3 - bs.q1)
    assert "median" in bs.summary_str()


def test_benchmark_statistics_below_noise_floor():
    bs = BenchmarkStatistics.from_samples([1.0, 2.0, 3.0])
    assert bs.num_samples == 3
    assert bs.noise is None  # < MIN_SAMPLES_FOR_NOISE_ESTIMATE
    assert bs.relative_iqr is None


def test_benchmark_statistics_empty():
    bs = BenchmarkStatistics.from_samples([])
    assert bs.num_samples == 0
    assert bs.noise is None
    assert math.isnan(bs.median)


# --------------------------------------------------------------------------- #
# Online linear regression (entropy helper) vs brute-force OLS
# --------------------------------------------------------------------------- #
def test_online_regression_append_matches_polyfit():
    rng = np.random.default_rng(4)
    ys = rng.normal(0.0, 1.0, size=40).cumsum()  # trending series
    reg = OnlineLinearRegression()
    for i, y in enumerate(ys):
        reg.update(float(i), float(y))
    xs = np.arange(len(ys), dtype=np.float64)
    slope_bf, intercept_bf = np.polyfit(xs, ys, 1)
    assert math.isclose(reg.slope(), slope_bf, rel_tol=1e-7, abs_tol=1e-9)
    assert math.isclose(reg.intercept(), intercept_bf, rel_tol=1e-7, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Stopping criteria
# --------------------------------------------------------------------------- #
def test_sample_count_criterion_stops_at_target():
    sc = SampleCountCriterion(target_samples=37)
    stopped = None
    for i in range(1, 100):
        sc.add_measurement(1.0)
        if sc.is_finished():
            stopped = i
            break
    assert stopped == 37


def test_sample_count_criterion_rejects_nonpositive():
    with pytest.raises(ValueError):
        SampleCountCriterion(target_samples=0)


def test_stdrel_converges_on_low_noise_stream():
    rng = np.random.default_rng(5)
    crit = StdRelCriterion(max_noise=0.005, min_time=0.0)
    stopped = None
    for i in range(1, 3000):
        crit.add_measurement(100.0 + float(rng.normal(0.0, 0.05)))  # ~0.05%
        if i >= MIN_SAMPLES_FOR_NOISE_ESTIMATE and crit.is_finished():
            stopped = i
            break
    assert stopped is not None
    assert crit.current_noise is not None and crit.current_noise < 0.005


def test_stdrel_respects_min_time():
    # Zero-noise stream converges instantly on noise, but min_time gates it:
    # accumulated time = mean * size must exceed min_time first.
    crit = StdRelCriterion(max_noise=0.005, min_time=1e9)
    for _ in range(50):
        crit.add_measurement(1.0)
    assert not crit.is_finished()  # 50 * 1.0 = 50 < 1e9


def test_stdrel_noise_plateau_fallback_terminates():
    # Inherently noisy stream that never reaches 0.5%: must still terminate via
    # the noise-stability fallback rather than spinning forever.
    rng = np.random.default_rng(6)
    crit = StdRelCriterion(max_noise=0.005, min_time=0.0)
    stopped = None
    for i in range(1, 5000):
        crit.add_measurement(100.0 + float(rng.normal(0.0, 5.0)))  # ~5% noise
        if i >= MIN_SAMPLES_FOR_NOISE_ESTIMATE and crit.is_finished():
            stopped = i
            break
    assert stopped is not None  # plateau fallback fired
    assert crit.current_noise is not None and crit.current_noise > 0.005


def test_entropy_criterion_terminates_on_discretized_stream():
    rng = np.random.default_rng(7)
    ent = EntropyCriterion(window=299)
    stopped = None
    for i in range(1, 8000):
        v = float(round(100.0 + rng.normal(0.0, 1.5)))  # timer-quantized
        ent.add_measurement(v)
        if i % 2 == 0 and ent.is_finished():
            stopped = i
            break
    assert stopped is not None


def test_entropy_sliding_window_matches_brute_force():
    rng = np.random.default_rng(8)
    ent = EntropyCriterion(window=50)
    for v in rng.normal(10.0, 0.5, size=400):
        ent.add_measurement(float(v))
        w = ent._entropy_tracker
        if len(w) >= 2:
            xs = np.arange(len(w), dtype=np.float64)
            ys = np.asarray(w, dtype=np.float64)
            slope_bf = np.polyfit(xs, ys, 1)[0]
            s = ent._regression.slope()
            if math.isfinite(s):
                assert math.isclose(s, slope_bf, rel_tol=1e-6, abs_tol=1e-9)


def test_make_criterion_factory():
    assert isinstance(make_criterion("stdrel"), StdRelCriterion)
    assert isinstance(make_criterion("sample-count"), SampleCountCriterion)
    assert isinstance(make_criterion("sample_count"), SampleCountCriterion)
    assert isinstance(make_criterion("entropy"), EntropyCriterion)
    with pytest.raises(ValueError):
        make_criterion("does-not-exist")


# --------------------------------------------------------------------------- #
# Adaptive GPU timing (requires CUDA + CUPTI >= 13)
# --------------------------------------------------------------------------- #
def _cupti_available() -> bool:
    try:
        from importlib.metadata import version as _v

        import cupti  # noqa: F401

        return int(_v("cupti-python").split(".")[0]) >= 13
    except Exception:
        return False


requires_cupti = pytest.mark.skipif(
    not (torch.cuda.is_available() and _cupti_available()),
    reason="requires CUDA + cupti-python >= 13",
)


def test_bench_gpu_time_with_statistics_requires_cupti():
    # Without CUPTI the adaptive path must hard-error (no CUDA-event fallback).
    if _cupti_available():
        pytest.skip("CUPTI present; cannot exercise the missing-CUPTI path")
    from flashinfer.testing import bench_gpu_time_with_statistics

    with pytest.raises(RuntimeError, match="CUPTI"):
        bench_gpu_time_with_statistics(lambda: None)


@requires_cupti
def test_bench_gpu_time_with_statistics_stdrel():
    from flashinfer.testing import bench_gpu_time_with_statistics, compute_statistics

    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")

    def run(x, y):
        torch.mm(x, y)

    samples, stats = bench_gpu_time_with_statistics(
        run,
        input_args=(a, b),
        stopping_criterion="stdrel",
        min_samples=10,
        max_time_ms=4000.0,
    )
    assert len(samples) >= 10
    assert stats.num_samples == len(samples)
    assert stats.median > 0.0
    assert stats.mean > 0.0
    # noise should be defined once we are above the sample floor
    assert stats.noise is not None and stats.noise >= 0.0
    # compute_statistics on the raw list reproduces the bundle
    again = compute_statistics(samples)
    assert again.num_samples == stats.num_samples
    assert math.isclose(again.median, stats.median)


@requires_cupti
def test_bench_gpu_time_with_statistics_sample_count_is_deterministic():
    from flashinfer.testing import bench_gpu_time_with_statistics

    x = torch.randn(1024, 1024, device="cuda")

    def run(t):
        t.relu()

    stats = bench_gpu_time_with_statistics(
        run,
        input_args=(x,),
        stopping_criterion="sample-count",
        target_samples=64,
        min_samples=1,
        return_samples=False,
    )
    # sample-count stops as soon as the floor (min_samples) and target both pass.
    assert stats.num_samples == 64
