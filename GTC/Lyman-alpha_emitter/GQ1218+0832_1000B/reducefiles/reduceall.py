# This is a python program to reduce the science frames
print('Script running')

import numpy as np
from astropy.io import fits
import os as os
import glob

import mkspecbias
import mkspecflat 
import crremoval
import reducestd

#Read information in info.txt
file = open('info.txt', 'r')
OB = file.readline().strip("\n")
ysize = int(file.readline())
xsize = int(file.readline())

mkspecbias
mkspecflat

BIASframe = fits.open('BIAS.fits')
BIAS = np.array(BIASframe[0].data)
FLATframe = fits.open('FLAT.fits')
FLAT = np.array(FLATframe[0].data)

crremoval
reducestd

rawimages = sorted(glob.glob("crr*.fits"))
#rawimages = sorted(glob.glob("00*.fits"))
nframes = len(rawimages)
outnames = ['sub1.fits','sub2.fits']
centers = [528,516]

#os.remove(outnames[0])
#os.remove(outnames[1])

#Read the raw file, subtract overscan, bias and divide by the flat
for n in range(0,nframes):
     spec = fits.open(str(rawimages[n]))
     print('Info on file:')
     print(spec.info())
     specdata = np.array(spec[0].data)
     specdata = (specdata-BIAS)/FLAT
     hdr = spec[0].header
     specdata1 = specdata[centers[n]-100:centers[n]+100,25:2072] 
     print(outnames[n])
     fits.writeto(outnames[n],specdata1,hdr,overwrite=True)

#Add and rotate
sub1 = fits.open(outnames[0])
sub2 = fits.open(outnames[1])
sum = sub1[0].data+sub2[0].data
hduout = fits.PrimaryHDU(sum)
hduout.header.extend(hdr, strip=True, update=True,
        update_first=False, useblanks=True, bottom=False)
hduout.header['DISPAXIS'] = 1
hduout.header['NEXP'] = len(rawimages)
hduout.header['CRVAL1'] = 1
hduout.header['CRVAL2'] = 1
hduout.header['CRPIX1'] = 1
hduout.header['CRPIX2'] = 1
hduout.header['CRVAL1'] = 1
hduout.header['CRVAL1'] = 1
hduout.header['CDELT1'] = 1
hduout.header['CDELT2'] = 1
hduout.writeto("../obj.fits", overwrite=True)
os.remove(outnames[0])
os.remove(outnames[1])


#Arcframe
arclist = glob.glob("../../GTCMULTIPLE2J-23A/"+OB+"/arc/*.fits")
print(arclist)
weights = [1.,1.]
specdata = np.zeros((ysize,xsize),float)
nweight = 0
for frames in arclist:
    spec = fits.open(str(frames))
    specdata += np.array(spec[0].data)*weights[nweight]
    nweight = nweight+1
specdata = (specdata-BIAS)/FLAT
hdr = spec[0].header
center = int((centers[0]+centers[1])/2.)
specdata1 = specdata[center-100:center+100,25:2072] 
hduout = fits.PrimaryHDU(specdata1)
hduout.header.extend(hdr, strip=True, update=True,
        update_first=False, useblanks=True, bottom=False)
hduout.header['DISPAXIS'] = 1
hduout.header['CRVAL1'] = 1
hduout.header['CRVAL2'] = 1
hduout.header['CRPIX1'] = 1
hduout.header['CRPIX2'] = 1
hduout.header['CRVAL1'] = 1
hduout.header['CRVAL1'] = 1
hduout.header['CDELT1'] = 1
hduout.header['CDELT2'] = 1
hduout.writeto("../arcsub.fits", overwrite=True)


