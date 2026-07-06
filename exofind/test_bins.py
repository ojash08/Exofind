import matplotlib.pyplot as plt

from photometry.loader import load_tpf
from photometry.star_locator import average_image
from photometry.star_locator import brightest_pixel
from photometry.background import estimate_local_background
from photometry.optimizer import optimize_aperture
from photometry.quality_flags import filter_quality
from photometry.detrend import detrend
from photometry.outliers import remove_outliers

from bls_v2.fold import fold_lightcurve
from bls_v2.bins import bin_lightcurve

FILE = "data/target_pixel/tess2018206045859-s0001-0000000261136679-0120-s_tp.fits"

tpf = load_tpf(FILE)

avg = average_image(tpf)

row, col = brightest_pixel(avg)

background = estimate_local_background(
    avg,
    (row, col)
)

mask, growth, time, flux, score, results = optimize_aperture(
    tpf,
    avg,
    background,
    (row, col)
)

time, flux, quality = filter_quality(
    time,
    flux,
    tpf.quality
)

flat_flux, trend = detrend(flux)

time_clean, flux_clean = remove_outliers(
    time,
    flat_flux
)

period = 6.26848

phase, folded_flux = fold_lightcurve(
    time_clean,
    flux_clean,
    period
)

centers, mean_flux, counts, scatter = bin_lightcurve(
    phase,
    folded_flux,
    bins=400
)

plt.figure(figsize=(12,5))

plt.scatter(
    phase,
    folded_flux,
    s=2,
    alpha=0.2,
    label="Raw"
)

plt.plot(
    centers,
    mean_flux,
    color="red",
    linewidth=2,
    label="Binned"
)

plt.xlabel("Phase")
plt.ylabel("Normalized Flux")
plt.legend()
plt.show()