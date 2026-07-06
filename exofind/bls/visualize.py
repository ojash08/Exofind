import matplotlib.pyplot as plt
import numpy as np

from bls.transit_box import make_transit_box
from bls.binning import bin_lightcurve


def plot_candidate(
        phase,
        flux,
        center,
        duration
):
    """
    Plot the folded light curve, detected transit,
    and binned light curve.
    """

    inside = make_transit_box(
        phase,
        center,
        duration
    )

    # Compute binned light curve
    bin_phase, bin_flux = bin_lightcurve(
        phase,
        flux,
        bins=100
    )

    plt.figure(figsize=(12, 5))

    # Raw data
    plt.scatter(
        phase,
        flux,
        s=3,
        alpha=0.35,
        color="tab:blue",
        label="Raw Data"
    )

    # Highlight detected transit points
    plt.scatter(
        phase[inside],
        flux[inside],
        s=10,
        color="red",
        label="Transit Points"
    )

    # Binned light curve
    plt.plot(
        bin_phase,
        bin_flux,
        color="black",
        linewidth=2,
        label="Binned Flux"
    )

    # Transit window
    plt.axvspan(
        center - duration / 2,
        center + duration / 2,
        color="red",
        alpha=0.15,
        label="Transit Window"
    )

    plt.xlabel("Phase")
    plt.ylabel("Normalized Flux")
    plt.title("ExoFind - Best Transit Candidate")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()