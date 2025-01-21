import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
import numpy as np
import pylab as pl
import matplotlib.gridspec as gridspec  ## Tool to make arbitrary subgrids
from scipy.optimize import curve_fit
from PyAstronomy import pyasl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset

# Some parameters
pl.rcParams["font.size"] = 18  # increase fontsize to an appropriate level
pl.rcParams["mathtext.default"] = (
    "regular"  # display latex text with the same font as the the other text
)

gs = gridspec.GridSpec(
    2, 1, height_ratios=[0.1, 0.99]
)  ## grid with two rows and one column

# Read the 2d fits file
imageframe = fits.open("obj.trans.skysub.fits")

# Now make np array with actual data
image = imageframe[0].data

# Create the figure with its axes
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(
    gs[0]
)  # gs is a list containing one entry for each subplot, idnexing from the upper left to the lower right, starting with 0
ax2 = fig.add_subplot(gs[1])
fig.subplots_adjust(hspace=0.000)

# Show the image
ax1.imshow(
    np.fliplr(image[6:194, 1169:1999]),
    cmap=cm.gray_r,
    vmin=-16,
    vmax=24,
    aspect="auto",
    origin="lower",
)

### <- apect = 'auto': change the aspect ratio and fill your subplot;
### <- apec = 'None': keeps the aspect ratio and fills the box w.r.t to the ratio
### <- To have the image shown, as e.g. ds9 would do it, you need to set origin = 'lower'


ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)

# Now the 1d spectrum
data = np.loadtxt("flux_obj.ms_1d.dat")
lam = data[:, 0]
flux = data[:, 1] * 1.0e17
err_flux = data[:, 3] * 1.0e17
lamvac = pyasl.airtovac2(lam)
lam = lamvac

# Make the plot
ax2.step(lam, flux, zorder=1, color="black", linewidth=0.5)
ax2.set_xlim(3700, 5400)
ax2.set_ylim(0, 2.5)
ax2.set_xlabel("Observed wavelength [Å]")
ax2.set_ylabel(
    "Flux [10"
    + r"$^{-17}$"
    + " erg s"
    + r"$^{-1}$"
    + "cm"
    + r"$^{-2}$"
    + r"$\AA^{-1}$"
    + "]"
)

# Make inset
ax3 = plt.axes([0, 0, 1, 1])
ip = InsetPosition(ax2, [0.25, 0.50, 0.30, 0.45])
ax3.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax2 and draw lines
# in grey linking the two axes.
mark_inset(ax2, ax3, loc1=2, loc2=4, fc="none", ec="0.5")
ax3.step(lam, flux, color="black", linewidth=0.5)
ax3.set_xlim(3.2261 * 1215.67 - 70, 3.2261 * 1215.67 + 70)
ax3.set_ylim(0, 2.0)
ax3.axvline(3.2261 * 1215.67, color="black", ymin=-10, ymax=100, linestyle="dotted")

fig.canvas.draw()
plt.savefig("LymanEmitter.pdf", bbox_inches="tight")
plt.show()

# raw_input('Continue with enter')
