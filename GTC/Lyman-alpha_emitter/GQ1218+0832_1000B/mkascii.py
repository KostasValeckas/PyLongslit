
from astropy.io import fits
import pandas as pd
import numpy as np

# Load file
file_name = 'flux_GQ125248+263805'
sn_file = fits.open(file_name+'.fits')

hdr = sn_file[0].header
flux = sn_file[0].data[0,0,:]
err_flux = sn_file[0].data[3,0,:]
wave = np.arange(len(flux))*hdr['CD1_1']+hdr['CRVAL1']

df_frame = {'wave':wave, 'flux':flux, 'err_flux':err_flux}
df = pd.DataFrame(df_frame,dtype='float64')

# Cut and only use data at lambda > 3800 AA
df_use = df[(df.wave > 3600) &  (df.wave < 8000) & (np.abs(df.wave-7650.) > 70) & (np.abs(df.wave-5888.) > 20)]

# Output file name
output_file = file_name
df_use.to_csv(output_file+'.tab', header=None, index=None, sep=' ')
