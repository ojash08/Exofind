"""
ExoFind BLS v2 — Scientific Box Least Squares

Implementation of the BLS algorithm from:
    Kovács, Zucker & Mazeh (2002)
    "A box-fitting algorithm in the search for periodic transits"
    A&A 391, 369–377 (arXiv:astro-ph/0206099)

Every module maps to one equation group from the paper:

    weights.py      [A1, A2]    Data normalization
    fold.py         [B1, B2]    Phase folding
    bins.py         [C1, C2]    Weighted phase binning
    prefix_sum.py   [D1, D2]    Weighted prefix sums
    sr.py           [E1–E4]     Signal Residue
    window.py       [F1–F3]     Sliding transit window
    periodogram.py  [G1–G3]     Period grid generation
    search.py       [H]         Period search orchestrator
    sde.py          [I1, I2]    Signal Detection Efficiency
    peaks.py        [J1–J3]     Peak detection
    confidence.py   [K1, K2]    False alarm probability
    validation.py   [—]         Astropy cross-validation
"""
