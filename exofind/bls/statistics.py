import numpy as np


def transit_statistics(flux, inside):
    """
    Compute statistics for a candidate transit box.
    """

    outside = ~inside

    n_inside = np.sum(inside)
    n_outside = np.sum(outside)

    if n_inside < 5 or n_outside < 5:
        return None

    inside_mean = np.mean(flux[inside])
    outside_mean = np.mean(flux[outside])

    depth = outside_mean - inside_mean

    if depth <= 0:
        return None

    # ---------- Build box model ----------
    model = np.full_like(flux, outside_mean)
    model[inside] = inside_mean

    # ---------- Residuals ----------
    residuals = flux - model

    # ---------- Sum of Squared Residuals ----------
    ssr = np.sum(residuals ** 2)

    variance = np.var(flux)

    if variance == 0:
        return None

    scatter = np.sqrt(variance)

    snr = depth / scatter

    power = (depth ** 2) * n_inside / variance
    bls_score = -ssr

    return {
        "depth": depth,
        "snr": snr,
        "power": power,
        "ssr": ssr,
        "bls_score": bls_score,
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "n_inside": n_inside,
        "n_outside": n_outside,
    }