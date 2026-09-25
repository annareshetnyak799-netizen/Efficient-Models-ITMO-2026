"""Fit performance-model parameters using calibration rows only."""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    from .equations import latency
except ImportError:
    from equations import latency


def calibrate(measurements: pd.DataFrame) -> dict:
    """Return latency and optional energy parameters in SI units."""
    validation = measurements["is_validation"].astype(str).str.lower()
    if not validation.isin(["true", "false"]).all():
        raise ValueError("is_validation must contain booleans")
    train = measurements.loc[
        validation.eq("false") & measurements["status"].eq("OK")
    ].copy()
    train = train.loc[
        np.isfinite(train["latency"]) & train["latency"].gt(0)
    ]
    if len(train) < 4:
        raise ValueError("At least four valid calibration rows are needed")

    sizes = train["S"].to_numpy()
    batches = train["B"].to_numpy()
    observed = train["latency"].to_numpy()
    keys = ("t0", "p_eff", "bw_eff")

    def parameters(log_values):
        return dict(zip(keys, np.exp(log_values).tolist()))

    def residuals(log_values):
        predicted = latency(sizes, batches, parameters(log_values))
        return np.log(predicted / observed)

    fits = []
    for compute, bandwidth in [(1e12, 1e11), (1e13, 1e11), (1e12, 1e12)]:
        initial = [max(observed.min() / 2, 1e-8), compute, bandwidth]
        fit = least_squares(
            residuals,
            np.log(initial),
            bounds=(np.log([1e-12, 1e6, 1e6]),
                    np.log([1e3, 1e18, 1e18])),
            max_nfev=2000,
        )
        if fit.success:
            fits.append(fit)
    if not fits:
        raise RuntimeError("Latency calibration did not converge")
    best = min(fits, key=lambda fit: fit.cost)
    theta = parameters(best.x)

    energy_rows = train.loc[
        np.isfinite(train["energy"]) & train["energy"].gt(0)
    ]
    theta_energy = None
    if not energy_rows.empty:
        predicted_time = latency(
            energy_rows["S"].to_numpy(),
            energy_rows["B"].to_numpy(),
            theta,
        )
        ratios = energy_rows["energy"].to_numpy() / predicted_time
        power = float(np.exp(np.mean(np.log(ratios))))
        theta_energy = {**theta, "power": power}

    return {
        "latency": theta,
        "energy": theta_energy,
        "calibration_rows": len(train),
        "energy_calibration_rows": len(energy_rows),
        "latency_log_rmse": float(np.sqrt(np.mean(best.fun**2))),
    }
