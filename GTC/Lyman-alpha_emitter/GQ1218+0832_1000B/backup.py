import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
import scipy.special
from astropy.io import fits
from scipy.optimize import curve_fit

plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)
plt.rcParams.update({"font.size": 15})


def gauss_function(x, I, mu, sigma):
    return I / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


plt.close("all")

ystart = 92
yend = 115
xstart = 1290
xend = 1320

image = fits.open("subtract2D.fits")
print(image.info())
hdr = image[0].header

cdelt = hdr["CDELT1"]
crval = hdr["CRVAL1"]
crpix = hdr["CRPIX1"]
wave = (np.arange(hdr["NAXIS1"]) - (crpix - 1)) * cdelt + crval

data = np.array(image[0].data)

profile = np.zeros(shape=(yend - ystart))

# First spatial profile
nsum = 0
for i in range(xstart, xend):
    profile[:] = profile[:] + data[ystart:yend, i]
    nsum = nsum + 1
profile = profile / nsum
plt.figure()
plt.plot(profile)
plt.show()

xstart = 1250
xend = 1350
spectrum = np.zeros(shape=(xend - xstart))

# Now extract spectrum
for i in range(xstart, xend):
    weight = 0.0
    for j in range(ystart, yend):
        spectrum[i - xstart] = spectrum[i - xstart] + data[j, i] * profile[j - ystart]
        weight = weight + profile[j - ystart]
    spectrum[i - xstart] = spectrum[i - xstart] / weight

Parameter_guesses = [np.max(spectrum), 3860.0, 100.0]
val, cov = curve_fit(gauss_function, wave[xstart:xend], spectrum, p0=Parameter_guesses)


plt.figure()
plt.step(wave[xstart:xend], spectrum, zorder=1, color="black", linewidth=0.5)
plt.plot(
    np.linspace(3843.0, 3880.0),
    gauss_function(np.linspace(3843.0, 3880.0), *val),
    "-",
    c="#fc5a50",
    label="Gaussian fit",
)
plt.xlim(3830, 3890)
plt.xlabel("Observed wavelength [Å]")
plt.ylabel("Flux [arbitrary units]")
plt.savefig("SDSSJ1012+0358_redshift.pdf", bbox_inches="tight")
plt.show()


print("Peak wavelength " + "{:.6}".format(val[1]) + " Å")
print("Sigma " + "{:.6}".format(val[2]) + " Å")
print("Error " + "{:.2}".format(np.sqrt(cov[1][1])) + " Å")
