import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import integrate
plt.rcParams.update({'font.size': 16})
from scipy.optimize import curve_fit
from PyAstronomy import pyasl

plt.close('all')

#Load filtercurve
data = np.loadtxt('134.data')
lamfilt = data[:,0]
transmis = data[:,1]

#Load spectrum
file = fits.open('flux_obj.ms_1d.fits')
hdr = file[0].header
flux = file[0].data[0,0,:]
err_flux = file[0].data[3,0,:]
wave = np.arange(len(flux))*hdr['CD1_1']+hdr['CRVAL1']
wavevac = pyasl.airtovac2(wave)
wave = wavevac

#Make plot
fig, ax1 = plt.subplots(figsize=(10, 8))
#ax2 = ax1.twinx()
ax1.step(wave,flux,zorder=1,color='black', linewidth=0.5)
ax1.step(wave,err_flux,zorder=1,color='black', linewidth=0.5)
#ax2.step(lamfilt,transmis*100.,color='black', linewidth=0.5, linestyle='--')
ax1.set_xlim(3700,5400)
ax1.set_ylim(-0.5e-17,4.0e-17)
#ax2.set_ylim(0,100)
ax1.set_xlabel('Observed wavelength [Å]')
ax1.set_ylabel('Flux [erg s'+r'$^{-1}$'+'cm'+r'$^{-2}$'+r'$\AA^{-1}$'+']')
#ax2.set_ylabel('Filter transmission [%]')
#ax1.axvline(1215.67*3.238,-1,1,color='black',linestyle='dotted')
#ax1.axvline(3925,-1,1,color='black',linestyle='dotted')
#ax1.axvline(3955,-1,1,color='black',linestyle='dotted')
ax1.axhline(0,0,1,color='black',linestyle='dotted')
plt.savefig('Emitter.pdf', bbox_inches='tight')
plt.show()

#Flux:
fitrange = np.where( (wave > 3925) & (wave < 3955))
#integral = integrate.cumtrapz(wave[fitrange], flux[fitrange], initial=0)
integral = sum(flux[fitrange])*2.046
print(integral)

#Wavelength
def gauss_function(x,I,mu,sigma):
    return I/(np.sqrt(2.0*np.pi)*sigma)*np.exp(-0.5*((x-mu)/sigma)**2)

Parameter_guesses = [2.e-17,3934.,3.]
val, cov = curve_fit(gauss_function, wave[fitrange], flux[fitrange], p0=Parameter_guesses)
print('Peak wavelength '+"{:.6}".format(val[1])+' Å')
print('Sigma '+"{:.6}".format(val[2])+' Å')
print('Error '+"{:.2}".format(np.sqrt(cov[1][1]))+' Å')
