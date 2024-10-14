# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from matplotlib import axes
import numpy as np
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.io.fits import getheader
from scipy import optimize
from astroscrappy import detect_cosmics
from astropy.stats import SigmaClip
from photutils.background import ModeEstimatorBackground
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.aperture import aperture_photometry
from PyAstronomy import pyasl
from scipy.optimize import curve_fit


#Input images and parameters

vmin = -3.e-18
vmax = 4.e-17

aperture_radius = 4.0
r_in = 10
r_out = 15

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

image_file = get_pkg_data_filename('obj.trans.skysub.fits')
fits.info(image_file)
image_data = fits.getdata(image_file, ext=0)
hdr = fits.getheader(image_file, ext=0)
image_data = image_data[:,11099-100:11099+100]
image_noise = image_data*0.+np.std(image_data[58:97,:])

plt.figure()
plt.imshow(image_data, cmap='gray_r', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
plt.colorbar()

fws = 20

print('Click on star with the cursor. End with q')
tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
xstar = int(tpoints[0][0])
ystar = int(tpoints[0][1])
plt.show()

#Now fit a 2d-gaussian to a region with size fws x fws around this position
subimage = image_data[ystar-fws:ystar+fws,xstar-fws:xstar+fws]
plt.matshow(subimage, cmap=plt.cm.gist_earth_r, origin='lower')
params = fitgaussian(subimage)
fit = gaussian(*params)
plt.contour(fit(*np.indices(subimage.shape)), cmap=plt.cm.copper)
ax = plt.gca()
(height, x, y, width_x, width_y) = params

plt.text(0.95, 0.05, """
x : %.1f
y : %.1f
width_x : %.1f
width_y : %.1f""" %(y+xstar-fws, x+ystar-fws, width_x, width_y),
     fontsize=16, horizontalalignment='right',
     verticalalignment='bottom', transform=ax.transAxes)
plt.show()
wl = 10.*((y+xstar-fws+11099-100.)*hdr['CD1_1']+hdr['CRVAL1'])
wlAir = pyasl.vactoair2(wl)
print('----------------------')
print('Wavelength of the line (Å) : ',wlAir)
#See https://astronomy.stackexchange.com/questions/44999/vacuum-and-air-wavelengths-in-spectroscopy

aperture = CircularAperture((y+xstar-fws,x+ystar-fws), r=aperture_radius)
annulus = CircularAnnulus((y+xstar-fws,x+ystar-fws), r_in=r_in, r_out=r_out)
apers = [aperture, annulus]
phot_table_local_bkg = aperture_photometry(image_data, apers, error=image_noise)
bkg_mean = phot_table_local_bkg['aperture_sum_1'] / annulus.area
bkg_sum = bkg_mean * aperture.area
final_sum = phot_table_local_bkg['aperture_sum_0'] - bkg_sum
final_err = phot_table_local_bkg['aperture_sum_err_1']
phot_table_local_bkg['residual_aperture_sum'] = final_sum
print('Flux in erg/s/cm2:',final_sum*2.046) # 2.046 is dispersion in Å/pix
print('Flux error in erg/s/cm2:',final_err*2.046) # 2.046 is dispersion in Å/pix


plt.figure()
plt.imshow(image_data, cmap='gray_r', vmin=vmin, vmax=vmax, origin='lower')
plt.ylim(x+ystar-fws-100,x+ystar-fws+100)
plt.xlim(y+xstar-fws-100,y+xstar-fws+100)
ap_patches = aperture.plot(color='white', lw=2,
                        label='Photometry aperture')
ann_patches = annulus.plot(color='red', lw=2,
                                 label='Background annulus')
handles = (ap_patches[0], ann_patches[0])
plt.legend(loc=(0.17, 0.05), facecolor='#458989', labelcolor='white',
        handles=handles, prop={'weight': 'bold', 'size': 11})

plt.colorbar()
plt.show()

#Position of the trace
#Left: 10983 to 11045
#Right 11134 to 11164

scale = 0.16 # #arcsec/pixel
#z=0.151 kpc/arcsec: 2.7032701610846903

# Define the Gaussian function
print
def Gauss(x, A, B, C):
    y = A*np.exp(-1*B*(x-C)**2)
    return y

left = np.mean(image_data[:,0:11045-11099+100], axis=1)*1.e19
col = np.arange(len(left))
Parameter_guesses = [np.max(left),1, 50]
parameters, covariance = curve_fit(Gauss, col, left, p0=Parameter_guesses)
fit_A = parameters[0]
fit_B = parameters[1]
fit_C = parameters[2]
fit = Gauss(col, fit_A, fit_B, fit_C)
plt.figure()
plt.plot(col,left)
plt.plot(col,fit)
plt.show()
print('Left offset: ',scale*(x+ystar-fws-fit_C), ' arcsec')
print('Left offset: ',scale*(x+ystar-fws-fit_C)*2.70, ' kpc')

right = np.mean(image_data[:,11134-11099+100:200], axis=1)*1.e19
col = np.arange(len(right))
Parameter_guesses = [np.max(right),1, 50]
parameters, covariance = curve_fit(Gauss, col, right, p0=Parameter_guesses)
fit_A = parameters[0]
fit_B = parameters[1]
fit_C = parameters[2]
fit = Gauss(col, fit_A, fit_B, fit_C)
plt.figure()
plt.plot(col,right)
plt.plot(col,fit)
plt.show()
print('Right offset: ',scale*(x+ystar-fws-fit_C), ' arcsec')
print('Right offset: ',scale*(x+ystar-fws-fit_C)*2.70, ' kpc')

