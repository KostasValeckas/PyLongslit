from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
import scipy.integrate as spi

normrange = 100.0

plt.close("all")

data = np.loadtxt("flux_obj.ms_1d.dat", skiprows=0)

lam = data[:, 0]
flux = data[:, 1] * 1.0e20
err_flux = data[:, 4] * 1.0e20
lstep = lam[1] - lam[0]

df_frame = {"wave": lam, "flux": flux, "err_flux": err_flux}
df = pd.DataFrame(df_frame, dtype="float64")

# Plot the spectrum
plt.figure()
plt.title("Click on line")
plt.xlim(3300, 4700)
plt.ylim(-0.5, 2)
plt.step(df.wave, df.flux, color="tab:red", zorder=1)
plt.step(df.wave, df.err_flux, color="tab:red", zorder=2, linestyle="dotted")
plt.xlabel("lambda i Å")
plt.ylabel("Flux")

# Select the line
slitregions = list(range(1))
nregions = 0
while nregions <= 0:
    tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    pix_ref, _ = tpoints[0]
    slitregions[nregions] = int(pix_ref)
    plt.axvline(slitregions[nregions], ymin=-1.0, ymax=2.0, color="b", lw=0.5)
    plt.draw()
    nregions = nregions + 1
lamline = slitregions[0]
plt.show()

# Plot region around the line
plt.figure()
plt.title("Mark regions on each side of the line for normalisation")
plt.xlim(lamline - normrange, lamline + normrange)
plt.ylim(-0.5, 3)
plt.step(df.wave, df.flux, color="tab:red", zorder=1)
plt.step(df.wave, df.err_flux, color="tab:red", zorder=2, linestyle="dotted")
plt.xlabel("lambda i Å")
plt.ylabel("Flux")

# Select the normalisation regions
slitregions = list(range(4))
nregions = 0
while nregions <= 3:
    tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    pix_ref, _ = tpoints[0]
    slitregions[nregions] = int(pix_ref)
    plt.axvline(slitregions[nregions], ymin=-1.0, ymax=2.0, color="b", lw=0.5)
    plt.draw()
    nregions = nregions + 1
laml1 = slitregions[0]
laml2 = slitregions[1]
lamr1 = slitregions[2]
lamr2 = slitregions[3]
plt.show()


# Normalise
def func(x, a, b):
    return a * x + b


df_fit = df[
    (df.wave > laml1) & (df.wave < laml2) | (df.wave > lamr1) & (df.wave < lamr2)
]
Parameter_guesses = [0.0, 1.0]
val, cov = curve_fit(func, df_fit.wave, df_fit.flux, p0=Parameter_guesses)
a = val[0]
b = val[1]
norm = df.flux / (a * df.wave + b)
normerr = df.err_flux / (a * df.wave + b)

# Now measure the line
plt.figure()
plt.title("Click on each side of the line to measure")
plt.xlim(lamline - normrange, lamline + normrange)
plt.ylim(-0.5, 1.5)
plt.step(df.wave, norm, color="tab:red", zorder=1)
plt.step(df.wave, normerr, color="tab:red", zorder=2, linestyle="dotted")
plt.xlabel("lambda i Å")
plt.ylabel("Normalised Flux")

# Select the normalisation regions
slitregions = list(range(2))
nregions = 0
while nregions <= 1:
    tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    pix_ref, _ = tpoints[0]
    slitregions[nregions] = int(pix_ref)
    plt.axvline(slitregions[nregions], ymin=-1.0, ymax=2.0, color="b", lw=0.5)
    plt.draw()
    nregions = nregions + 1
laml = slitregions[0]
lamr = slitregions[1]
plt.show()

# First moment
df_moment = df[(df.wave > laml) & (df.wave < lamr)]
profile = 1.0 - (df_moment.flux / (a * df_moment.wave + b))
M1 = np.sum(df_moment.wave * profile / df_moment.err_flux**2) / np.sum(
    profile / df_moment.err_flux**2
)
print("Wavelength of the line:", f"{M1:.2f}", "AA")

# No calculate the Equivalength width
EW = lstep * np.sum(profile)
EWerr = lstep * np.sqrt(np.sum(df_moment.err_flux**2))
print("EW: ", f"{EW:.2f}", "AA +/- ", f"{EWerr:.2f}", "AA")
