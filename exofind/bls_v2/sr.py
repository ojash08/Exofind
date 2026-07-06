"""
sr.py — Signal Residue

Equations [E1]–[E4] from Kovács, Zucker & Mazeh (2002)

The core statistic of the BLS algorithm.

[E1] Signal Residue:
    SR = s^2 / (r * (1 - r))

    where:
        s = sum(w_tilde_i * x_tilde_i)  for in-transit points
        r = sum(w_tilde_i)              for in-transit points

[E2] In-transit weighted mean:
    L = s / r

[E3] Out-of-transit weighted mean:
    H = -s / (1 - r)

    (follows from sum(w_tilde * x_tilde) = 0)

[E4] Transit depth:
    delta = H - L = -s / (r * (1 - r))

    For a transit (dip), s < 0, so delta > 0.
"""


def signal_residue(s, r):
    """
    Compute the Signal Residue for a given
    in-transit weighted sum (s) and weight sum (r).

    Parameters
    ----------
    s : float
        Weighted flux sum of in-transit points:
        s = sum(w_tilde_i * x_tilde_i)

    r : float
        Weight sum of in-transit points:
        r = sum(w_tilde_i)

    Returns
    -------
    sr : float
        Signal Residue = s^2 / (r * (1 - r)).
        Returns 0.0 if r is degenerate (0 or 1).
    """

    # Guard against degenerate cases
    if r <= 0.0 or r >= 1.0:
        return 0.0

    return (s * s) / (r * (1.0 - r))


def transit_parameters(s, r):
    """
    Compute the physical transit parameters from
    the BLS weighted sums.

    Parameters
    ----------
    s : float
        Weighted flux sum of in-transit points.

    r : float
        Weight sum of in-transit points.

    Returns
    -------
    params : dict
        L       : in-transit weighted mean [E2]
        H       : out-of-transit weighted mean [E3]
        depth   : transit depth = H - L [E4]
        s       : in-transit weighted flux sum
        r       : in-transit weight sum
        sr      : Signal Residue [E1]

    Returns None if r is degenerate.
    """

    if r <= 0.0 or r >= 1.0:
        return None

    sr = (s * s) / (r * (1.0 - r))

    L = s / r
    H = -s / (1.0 - r)
    depth = H - L

    return {
        "L": L,
        "H": H,
        "depth": depth,
        "s": s,
        "r": r,
        "sr": sr
    }
