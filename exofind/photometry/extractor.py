import numpy as np


def extract_lightcurve(tpf, mask, background):

    flux = []

    for frame in tpf.flux.value:

        corrected = frame - background

        total_flux = corrected[mask].sum()

        flux.append(total_flux)

    return np.array(flux)