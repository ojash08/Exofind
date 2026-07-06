import numpy as np

from photometry.aperture import build_aperture
from photometry.photometer import build_lightcurve
from photometry.normalize import normalize_flux
from photometry.quality import quality_score, rms
from photometry.snr import signal, noise, photometric_snr


def optimize_aperture(
        tpf,
        average_image,
        background,
        star_position,
):
    """
    Try multiple aperture thresholds and
    return the best one.
    """

    # Higher sigmas create tighter, smaller apertures to aggressively
    # exclude local background noise while keeping the bright star center.
    sigmas = [5.0, 6.0, 7.0, 8.0, 9.0]

    best_score = -np.inf
    best_mask = None
    best_flux = None
    best_time = None
    best_growth_order = None

    results = []

    for sigma in sigmas:

        threshold = background + sigma * np.std(average_image)

        mask, growth_order = build_aperture(
            average_image,
            star_position,
            threshold
        )

        time, flux = build_lightcurve(
            tpf,
            mask,
            background
        )

        flux = normalize_flux(flux)

        score = quality_score(
            flux,
            mask
        )

        results.append(
            {
                "sigma": sigma,
                "pixels": np.sum(mask),
                "signal": signal(flux),
                "noise": noise(flux),
                "rms": rms(flux),
                "snr": photometric_snr(flux),
                "score": score,
            }
        )

        print(f"Sigma={sigma:.1f}  Score={score:.2f}")

        if score > best_score:
            best_score = score
            best_mask = mask
            best_flux = flux
            best_time = time
            best_growth_order = growth_order

    return (
        best_mask,
        best_growth_order,
        best_time,
        best_flux,
        best_score,
        results,
    )