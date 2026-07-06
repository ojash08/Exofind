import numpy as np


def extract_flux(frame, mask, background):
    """
    Calculates the total brightness of the star
    in one image.
    """

    corrected_frame = frame - background

    total_flux = np.sum(corrected_frame[mask])

    return total_flux


def build_lightcurve(tpf, mask, background):
    """
    Convert every frame into one brightness value.
    """

    flux = []

    for frame in tpf.flux.value:

        brightness = extract_flux(
            frame,
            mask,
            background
        )

        flux.append(brightness)

    return (
        tpf.time.value,
        np.array(flux)
    )