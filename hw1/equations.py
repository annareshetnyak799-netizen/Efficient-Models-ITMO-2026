"""Analytical forward-pass costs in FLOPs, bytes, seconds and joules."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


PARAMETER_BYTES = 4_161_296

# FLOPs / BS², traffic / BS², parameter bytes, ReLU traffic / BS².
CONV_COSTS = (
    (2352, 44, 18_816, 64),
    (6400, 24, 204_800, 32),
    (2304, 24, 294_912, 16),
    (1024, 24, 131_072, 32),
    (4608, 20, 2_359_296, 8),
    (1024, 12, 524_288, 16),
)


def _inputs(
    image_size: ArrayLike, batch: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    size, batch = np.broadcast_arrays(
        np.asarray(image_size, dtype=np.float64),
        np.asarray(batch, dtype=np.float64),
    )
    if np.any(~np.isfinite(size) | (size <= 0) | (size % 16 != 0)):
        raise ValueError("image_size must contain positive multiples of 16")
    if np.any(~np.isfinite(batch) | (batch <= 0) | (batch % 1 != 0)):
        raise ValueError("batch must contain positive integers")
    return batch * size**2, batch


def _result(value: ArrayLike) -> float | NDArray[np.float64]:
    value = np.asarray(value, dtype=np.float64)
    return value.item() if value.ndim == 0 else value


def flops(
    image_size: ArrayLike, batch: ArrayLike,
) -> float | NDArray[np.float64]:
    """Arithmetic FLOPs; comparisons are excluded."""
    q, batch = _inputs(image_size, batch)
    return _result(17_714 * q + 313_700 * batch)


def memory(
    image_size: ArrayLike, batch: ArrayLike,
) -> float | NDArray[np.float64]:
    """Parameter and activation sum in bytes, excluding workspaces."""
    q, batch = _inputs(image_size, batch)
    return _result(PARAMETER_BYTES + 104 * q + 3472 * batch)


def bytes_moved(
    image_size: ArrayLike, batch: ArrayLike,
) -> float | NDArray[np.float64]:
    """Bytes assuming one input/parameter read and one output write."""
    q, batch = _inputs(image_size, batch)
    return _result(364 * q + 8592 * batch + PARAMETER_BYTES)


def latency(
    image_size: ArrayLike,
    batch: ArrayLike,
    theta: Mapping[str, float],
) -> float | NDArray[np.float64]:
    """Seconds; theta keys: t0 (s), p_eff (FLOP/s), bw_eff (byte/s)."""
    q, batch = _inputs(image_size, batch)
    t0, compute, bandwidth = (
        float(theta[key]) for key in ("t0", "p_eff", "bw_eff")
    )
    if not all(np.isfinite(v) for v in (t0, compute, bandwidth)):
        raise ValueError("latency parameters must be finite")
    if t0 < 0 or compute <= 0 or bandwidth <= 0:
        raise ValueError("require t0 >= 0, p_eff > 0 and bw_eff > 0")

    time = t0 + (40 * q + 2048 * batch) / bandwidth  # Pool and head ReLU.
    for flop_coeff, byte_coeff, weights, relu_coeff in CONV_COSTS:
        time = time + np.maximum(
            flop_coeff * q / compute,
            (byte_coeff * q + weights) / bandwidth,
        )
        time = time + relu_coeff * q / bandwidth

    time = time + np.maximum(
        2 * q / compute, (8 * q + 2048 * batch) / bandwidth,
    )
    time = time + np.maximum(
        262_400 * batch / compute,
        (3072 * batch + 525_312) / bandwidth,
    )
    time = time + np.maximum(
        51_300 * batch / compute,
        (1424 * batch + 102_800) / bandwidth,
    )
    return _result(time)


def energy(
    image_size: ArrayLike,
    batch: ArrayLike,
    theta_energy: Mapping[str, float],
) -> float | NDArray[np.float64]:
    """Joules; latency parameters plus power (whole-GPU mean watts)."""
    power = float(theta_energy["power"])
    if not np.isfinite(power) or power < 0:
        raise ValueError("power must be finite and nonnegative")
    return _result(power * latency(image_size, batch, theta_energy))
