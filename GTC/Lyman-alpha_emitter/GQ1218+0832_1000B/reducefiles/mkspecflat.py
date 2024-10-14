# This is a python program to make a specflat frame
print('Script running')
import astropy
import numpy as np
from astropy.io import fits
import glob
import numpy as np

#Read information in info.txt
file = open('info.txt', 'r')
OB =  file.readline()
ysize = file.readline()
xsize = file.readline()
OB = OB.strip("\n")
xsize = int(xsize)
ysize = int(ysize)


#Read in the raw flat frames and subtact mean of overscan region 
list = glob.glob("../../GTCMULTIPLE2J-23A/"+OB+"/flat/*fits")
nframes = len(list)
print(nframes)
bigflat = np.zeros((nframes,ysize,xsize),float)
BIASframe = fits.open('BIAS.fits')
BIAS = np.array(BIASframe[0].data)
i = 0
for frame in list:
   print('Image:', frame)
   rawflat = fits.open(str(frame))
   print('Info on file:')
   print(rawflat.info())
   data = np.array(rawflat[0].data)
   data = data - BIAS
   print('Subtracted the BIAS')
   bigflat[i-1,0:ysize-1,0:xsize-1] = data[0:ysize-1,0:xsize-1]
   norm = np.median(bigflat[i-1,100:2000,100:2000])
   print('Normalised with the median of the frame :',norm)
   bigflat[i-1,:,:] = bigflat[i-1,:,:]/norm
   i+=1

#Calculate flat is median at each pixel
medianflat = np.median(bigflat,axis=0)

#Normalise the flat field
lampspec = np.mean(medianflat,axis=0)
norm = medianflat*0.
for i in range(0,ysize-1):
    medianflat[i,:] = medianflat[i,:] / lampspec[:]
#for i in range(0,xsize-1):
#   medianflat[:,i] = medianflat[:,i] / lampspec[:]

#Write out result to fitsfile
hdr = rawflat[0].header
fits.writeto('FLAT.fits',medianflat,hdr,overwrite=True)
