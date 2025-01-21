import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits

plt.close("all")

image = fits.open("obj.fits")
print(image.info())

data = np.array(image[0].data)
flux1 = np.zeros(shape=(200, 1))
flux2 = np.zeros(shape=(200, 1))

for i in range(1196, 1215):  # Sum flux
    flux1[:, 0] = flux1[:, 0] + data[:, i]

for i in range(1166, 1185):  # Sum flux
    flux2[:, 0] = flux2[:, 0] + data[:, i]

plt.figure()
plt.xlim(80, 120)
plt.plot(flux1 * 1.15, linestyle="dashed")
plt.plot(flux2)
plt.show()
