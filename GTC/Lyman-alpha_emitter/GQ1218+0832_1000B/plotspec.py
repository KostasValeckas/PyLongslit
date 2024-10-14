import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

name = 'spectrum'

#Load spectrum
data = np.loadtxt('flux_obj.ms_1d.dat')
lam = data[:,0]
flux = data[:,2]
err_flux = data[:,4]

# Smooth spectra
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


#Make plot
#Yrange
plt.figure()
plt.plot(lam,flux,zorder=1)
plt.plot(lam,err_flux,zorder=1)
plt.xlim(5000,9900)
plt.xlabel('lambda i Å')
plt.ylabel('Flux')
plt.title(name)
plt.xlabel('Observed wavelength [Å]')
plt.ylabel('Flux [arbitrary units]')

plt.savefig(name+'.pdf', bbox_inches='tight', figsize=(8,5))


plt.show()
