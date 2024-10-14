import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits as pf
import numpy as np
import pylab as pl

import matplotlib.gridspec as gridspec ## Tool to make arbitrary subgrids

# Some parameters, that 

pl.rcParams['font.size'] = 14   # increase fontsize to an appropriate level 
pl.rcParams['mathtext.default'] = 'regular' # display latex text with the same font as the the other text


gs = gridspec.GridSpec(2,1,height_ratios=[0.2,0.7])   ## grid with two rows and one column


#Read the 2d fits file
imageframe = pf.open('050730BSw.fits')
print 'Info on 2d spectrum:'
imageframe.info()
# Now make np array with actual data
image = imageframe[0].data

# Create the figure with its axes
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(gs[0])  # gs is a list containing one entry for each subplot, idnexing from the upper left to the lower right, starting with 0
ax2 = fig.add_subplot(gs[1])
fig.subplots_adjust(hspace = .000)

## Show the image
ax1.imshow(image[16:216,186:1849], cmap=cm.gray_r, vmin=-30, vmax=300, aspect = 'auto', origin = 'lower')

   ### <- apect = 'auto': change the aspect ratio and fill your subplot; 
   ### <- apec = 'None': keeps the aspect ratio and fills the box w.r.t to the ratio
   ### <- To have the image shown, as e.g. ds9 would do it, you need to set origin = 'lower'



#Now the 1d spectrum
specframe = pf.open('comb050730.fits')
print 'Info on 1d spectrum:'
specframe.info()
spec = specframe[0].data
header = specframe[0].header


### Determine the wavelength grid from the FITS header 
crval = header['CRVAL1']
cdelt = header['CDELT1']
crpix1 = header['CRPIX1']
wave = np.arange(1,len(spec)+1)
wave = (wave-crpix1)*cdelt   +  crval

ax2.set_xlabel(r'$\lambda[\AA]$')
ax2.yaxis.set_visible(False)


#Make the plot
ax2.plot(wave[186:1849],spec[186:1849])
ax2.set_xlim(wave[186],wave[1849])
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
fig.canvas.draw()

raw_input('Continue with enter')
