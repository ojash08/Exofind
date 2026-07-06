import numpy as np


def interval_sum(prefix, start, end):
    """
    Sum over interval [start, end)
    using prefix sums.
    """
    return prefix[end] - prefix[start]


def bls_statistic(
        prefix_flux,
        prefix_flux2,
        prefix_counts,
        start,
        end,
):
    """
    Compute the Box Least Squares
    statistic for one transit window.
    """

    total_flux = prefix_flux[-1]
    total_count = prefix_counts[-1]

    inside_flux = interval_sum(
        prefix_flux,
        start,
        end
    )

    inside_count = interval_sum(
        prefix_counts,
        start,
        end
    )

    outside_flux = total_flux - inside_flux
    outside_count = total_count - inside_count

    if inside_count < 5:
        return None

    if outside_count < 5:
        return None

    inside_mean = inside_flux / inside_count
    outside_mean = outside_flux / outside_count

    depth = outside_mean - inside_mean

    if depth <= 0:
        return None

    # Fraction of observations in transit
    r = inside_count / total_count

    # Transit signal
    s = inside_count * depth

    # Signal Residue (Kovacs et al.)
    sr = (s ** 2) / (r * (1 - r))

    return {
        "depth": depth,
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "inside_count": inside_count,
        "outside_count": outside_count,
        "sr": sr
    }