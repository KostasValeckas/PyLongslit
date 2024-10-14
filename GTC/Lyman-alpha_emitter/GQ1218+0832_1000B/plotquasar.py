import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits

plt.close('all')

#Parameters
name = 'PSSJ0052+2405'
z = 2.500
AB = 0.90 

umag = 24.62
gmag = 20.50
rmag = 18.68
imag = 18.84
zmag = 17.67
jmag = 15.45
hmag = 14.534
kmag = 13.879
w1mag = 13.228
w2mag = 12.222

mags = [umag, gmag, rmag, imag, zmag, jmag, hmag, kmag]
mags = mags + np.array([0., 0., 0., 0., 0., 0.91, 1.39, 1.85])
lambphot = [3540., 4750., 6220., 7630., 9050., 12480., 16310., 22010.]
lambphot = np.array(lambphot)
fluxphot = 10.**(-0.4*(48.60+mags))
fluxphot = fluxphot*3.e18/(lambphot**2)*2.3e19

#Load spectrum
data = np.loadtxt('flux_obj.ms_1d.dat')
lam = data[:,0]
flux = data[:,1]*1.e19
err_flux = data[:,3]*1.e19

data = np.loadtxt('../PSSJ0052+2405_EMIR/flux_obj.ms_1d.dat')
lamir = data[:,0]
fluxir = data[:,1]*5.e16
errir_flux = data[:,3]*5.e16

#Also the SDSS spectrum
sdss_file = fits.open('spec-7675-57327-0278.fits')
hdr = sdss_file[0].header
specdata = sdss_file[1].data
sdsslam = 10**specdata['loglam'] #* u.AA 
sdssflux = specdata['flux']*100.

#Read template
temp = np.loadtxt('compoM.data')
lamtemp = temp[:,0]
fluxtemp = temp[:,1]

# Smooth spectra
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


#SMC extinction parameters
ai = np.array([185.,27.,0.005,0.010,0.012,0.030])
wli = np.array([0.042,0.08,0.22,9.7,18.,25.])
bi = np.array([90.,5.50,-1.95,-1.95,-1.80,0.0])
ni = np.array([2.0,4.0,2.0,2.0,2.0,2.0])
Ki = np.array([2.89,0.91,0.02,1.55,1.72,1.89])

#Redden the template
Alambda = fluxtemp*0.
wlr = lamtemp/1.e4
for e in range(len(ai)):
    Alambda=Alambda+ai[e]/((wlr/wli[e])**ni[e]+(wli[e]/wlr)**ni[e]+bi[e])

Alambda = Alambda*AB
model = 10**(-0.4*Alambda)*fluxtemp

#Normalise
specfilt = np.nonzero((lam > (4000)) & (lam < (4400)))
normspec = np.mean(flux[specfilt])
modelfilt = np.nonzero((lamtemp*(1+z) > (5700)) & (lamtemp*(1+z) < (6600)))
normmodel = np.mean(model[modelfilt])
factor = normspec/normmodel

#Make plot
#Yrange
rangefilt = np.nonzero((lamtemp*(1+z) > (3500)) & (lamtemp*(1+z) < (9500)))
yrange = np.max(model[rangefilt]*factor)*70
plt.figure()
#plt.plot(lamtemp*(1+z),smooth(fluxtemp*factor,3),color='tab:red',linestyle='dashed',zorder=2)
plt.plot(lamtemp*(1+z),smooth(model*factor,3)*30.,color='tab:red',zorder=3)
#plt.errorbar(lam,flux,yerr=err_flux,fmt='.',zorder=1)
plt.plot(sdsslam,sdssflux,zorder=1)
plt.plot(lam,flux,zorder=2)
plt.plot(lamir,fluxir,zorder=1)
plt.plot(lambphot,fluxphot, zorder=4, linestyle='', marker='o',color='blue')
plt.xlim(3600,25000)
plt.ylim(0,yrange)
plt.xlabel('Observed wavelength [Å]')
plt.ylabel('Flux [arbitrary units]')
plt.title(name)
plt.annotate('z='+str(z)+', A$_B$='+str(AB), xy = (6900,0.9*yrange))
plt.axvline(x = (1+2.50)*6564., color = 'g', label = 'Halpha')
plt.savefig(name+".pdf", bbox_inches='tight', figsize=(8,5))

plt.show()
