import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits

plt.close('all')

ys1=1
ye1=62
ys2=84
ye2=287
ys3=317
ye3=367

image = fits.open('obj.trans.fits') 
print(image.info())
hdr = image[0].header

data = np.array(image[0].data)

sky = data[0,:]*0.
ns = 0
for ny in range(ys1,ye1):
    ns += 1
    sky[:] = sky[:]+data[ny,:]
for ny in range(ys2,ye2):
    ns += 1
    sky[:] = sky[:]+data[ny,:]
for ny in range(ys3,ye3):
    ns += 1
    sky[:] = sky[:]+data[ny,:]
sky = sky/float(ns)

for i in range(1,400): 
    data[i,:] = data[i,:]-sky[:]

fits.writeto('subtract2D.fits',data,hdr,overwrite=True)
