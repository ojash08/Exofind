import numpy as np


def box_score(phase, flux, width=0.05):
    """
    Slide a transit box across phase and
    find the deepest transit.
    """

    best_score = -np.inf
    best_center = None

    centers = np.linspace(0, np.max(phase), 200)

    for center in centers:

        inside = np.abs(phase - center) < width / 2

        if np.sum(inside) < 5:
            continue

        outside = ~inside

        inside_mean = np.mean(flux[inside])
        outside_mean = np.mean(flux[outside])

        depth = outside_mean - inside_mean

        scatter = np.std(flux[outside])

        if scatter == 0:
            continue

        score = depth / scatter

        if score > best_score:
            best_score = score
            best_center = center

    return best_score, best_center