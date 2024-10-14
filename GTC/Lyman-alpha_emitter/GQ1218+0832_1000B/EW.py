from astropy.io import fits
from astropy.io.fits import getheader
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
import scipy.integrate as spi
from PyAstronomy import pyasl

normrange = 100.

plt.close('all')

#Read 1d-file
data = np.loadtxt('flux_obj.ms_1d.dat',skiprows=1)
lam = data[:,0]*1.
lamvac = pyasl.airtovac2(lam)
flux = data[:,1]*1.e18
err_flux = data[:,4]*1.e18
lstep = lam[0]-lam[1]
df_frame = {'wave':lamvac, 'flux':flux, 'err_flux':err_flux}
df = pd.DataFrame(df_frame,dtype='float64')

#Read the 2d-file
imageframe = fits.open('obj.trans.skysub.fits')
#Get the wavelength imformation
hdr = getheader('obj.trans.skysub.fits')
wl2d = (hdr['CRVAL1']+hdr['CDELT1']*np.arange(hdr['NAXIS1']))
NAXIS1 = hdr['NAXIS1']

#Read sky transmission file
data = np.loadtxt('skycalc_abs.dat',skiprows=0)
lamt = data[:,0]*10.
trans = data[:,1]

#Plot the 1d-spectrum
plt.figure()
plt.title('Click on line')
plt.xlim(7300,10000)
plt.ylim(-0.1,100)
plt.step(df.wave,df.flux,color='tab:red',zorder=1)
plt.step(df.wave,df.err_flux,color='tab:red',zorder=2,linestyle='dotted')
plt.xlabel('lambda i Å')
plt.ylabel('Flux')

#Select the line
slitregions = list(range(1))
nregions = 0
while nregions <= 0:
    tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    pix_ref, _ = tpoints[0]
    slitregions[nregions] = int(pix_ref)
    plt.axvline(slitregions[nregions], ymin=-1., ymax=2., color='b', lw=0.5)
    plt.draw()
    nregions = nregions+1
lamline = slitregions[0]
plt.show()

diff = abs(wl2d-lamline)
mindiff = min(diff)
cen = np.where(diff == mindiff)
cen2d= int(cen[0])


#Plot region around the line both in 2d and 1d
gs = gridspec.GridSpec(2,1,height_ratios=[0.2,0.7])
image = np.flip(imageframe[0].data,1)
# Create the figure with its axes
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(gs[0])  # gs is a list containing one entry for each subplot, idnexing from the upper left to the lower right, starting with 0
ax2 = fig.add_subplot(gs[1])
fig.subplots_adjust(hspace = .000)
# Show the image
sigmabk = 50.
medianbk = 0.
ax1.imshow(image[0:hdr['NAXIS2'],NAXIS1-cen2d-int(normrange/abs(hdr['CDELT1'])):NAXIS1-cen2d+int(normrange/abs(hdr['CDELT1']))], cmap=cm.gray_r, vmin=medianbk-3*sigmabk, vmax=medianbk+10*sigmabk, aspect = 'auto', origin = 'lower')
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
#Now the 1d spectrum
df_use = df[(df.wave > lamline-normrange) &  (df.wave < lamline+normrange)]
sig1d = np.median(df_use.err_flux)
mean1d = np.mean(df_use.flux)
ax2.set_xlabel(r'$\lambda[\AA]$')
ax2.set_xlabel('Observed wavelength [Å]')
ax2.yaxis.set_visible(True)
ax2.set_title('Mark regions on each side of the line in the 1d plot for normalisation')
ax2.set_xlim(lamline-normrange,lamline+normrange)
ax2.set_ylim(-2.*sig1d,1.1*mean1d+9.*sig1d)
ax2.step(df.wave,df.flux,color='tab:red',zorder=1)
ax2.step(df.wave,df.err_flux,color='tab:red',zorder=2,linestyle='dotted')
ax2.set_xlabel('lambda i Å')
ax2.set_ylabel('Flux')

#Select the normalisation regions
slitregions = list(range(4))
nregions = 0
while nregions <= 3:
    tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    pix_ref, _ = tpoints[0]
    slitregions[nregions] = int(pix_ref)
    plt.axvline(slitregions[nregions], ymin=-1., ymax=2., color='b', lw=0.5)
    plt.draw()
    nregions = nregions+1
laml1 = slitregions[0]
laml2 = slitregions[1]
lamr1 = slitregions[2]
lamr2 = slitregions[3]
fig.canvas.draw()
plt.show()

#Normalise
def func(x, a, b):
    return a * x + b
df_fit = df[(df.wave > laml1) & (df.wave < laml2) | (df.wave > lamr1) & (df.wave < lamr2)]
Parameter_guesses = [0.,1.]
val, cov = curve_fit(func, df_fit.wave, df_fit.flux, p0=Parameter_guesses)
a = val[0]
b = val[1]
norm = df.flux/(a*df.wave+b)
normerr = df.err_flux/(a*df.wave+b)

#Now measure the line. Again show both 1d and 2d
gs = gridspec.GridSpec(2,1,height_ratios=[0.2,0.7])
# Create the figure with its axes
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(gs[0])  # gs is a list containing one entry for each subplot, idnexing from the upper left to the lower right, starting with 0
ax2 = fig.add_subplot(gs[1])
fig.subplots_adjust(hspace = .000)
# Show the image
ax1.imshow(image[0:hdr['NAXIS2'],NAXIS1-cen2d-int(normrange/abs(hdr['CDELT1'])):NAXIS1-cen2d+int(normrange/abs(hdr['CDELT1']))], cmap=cm.gray_r, vmin=medianbk-3*sigmabk, vmax=medianbk+10*sigmabk, aspect = 'auto', origin = 'lower')
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
#Now the 1d spectrum
ax2.set_xlabel(r'$\lambda[\AA]$')
ax2.set_xlabel('Observed wavelength [Å]')
ax2.yaxis.set_visible(True)
ax2.set_title('Click on each side of the line in the 1d plot to measure')
ax2.set_xlim(lamline-normrange,lamline+normrange)
ax2.set_ylim(-0.5,1.5)
ax2.step(df.wave,norm,color='tab:red',zorder=1)
ax2.step(lamt,trans,color='tab:blue',zorder=1)
ax2.step(df.wave,normerr,color='tab:red',zorder=2,linestyle='dotted')
ax2.set_xlabel('lambda i Å')
ax2.set_ylabel('Normalised Flux')

#Select the measurement region
slitregions = list(range(2))
nregions = 0
while nregions <= 1:
    tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    pix_ref, _ = tpoints[0]
    slitregions[nregions] = int(pix_ref)
    plt.axvline(slitregions[nregions], ymin=-1., ymax=2., color='b', lw=0.5)
    plt.draw()
    nregions = nregions+1
laml = slitregions[0]
lamr = slitregions[1]
fig.canvas.draw()
plt.show()

#First moment
df_moment = df[(df.wave > laml) & (df.wave < lamr)]
profile = 1.-(df_moment.flux/(a*df_moment.wave+b))
profile_err = (df_moment.err_flux/(a*df_moment.wave+b))
M1 = np.sum(df_moment.wave*profile/df_moment.err_flux**2)/np.sum(profile/df_moment.err_flux**2)
print('Wavelength of the line:',f'{M1:.2f}','AA')

#No calculate the Equivalength width
EW = lstep*np.sum(profile)
EWerr = lstep*np.sqrt(np.sum(profile_err**2))
print('EW: ',f'{EW:.2f}','AA +/- ',f'{EWerr:.2f}','AA')
