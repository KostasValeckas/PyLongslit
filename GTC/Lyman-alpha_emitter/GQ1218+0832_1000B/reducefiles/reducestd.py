import numpy as np
from astropy.io import fits
import os as os
import glob

# Run the setup script
# exec(open("setup.py").read())

# This is a python program to reduce the science frames
print("Script running")

# Read information in info.txt
file = open("info.txt", "r")
OB = file.readline().strip("\n")
ysize = int(file.readline())
xsize = int(file.readline())

BIASframe = fits.open("BIAS.fits")
BIAS = np.array(BIASframe[0].data)
FLATframe = fits.open("FLAT.fits")
FLAT = np.array(FLATframe[0].data)

rawimages = sorted(glob.glob("../../GTCMULTIPLE2J-23A/" + OB + "/stds/*.fits"))
print(rawimages)
outnames = ["sub1.fits"]
centers = [768]

# Read the raw file, subtract overscan, bias and divide by the flat
spec = fits.open(rawimages[2])
print("Info on file:")
print(spec.info())
specdata = np.array(spec[0].data)
specdata = (specdata - BIAS) / FLAT
hdr = spec[0].header
specdata1 = specdata[centers[0] - 100 : centers[0] + 100, 25:2072]
print(outnames[0])
fits.writeto(outnames[0], specdata1, hdr, overwrite=True)

# Add
sub1 = fits.open(outnames[0])
sum = sub1[0].data
fits.writeto("../std.fits", sum, hdr, overwrite=True)
os.remove(outnames[0])

# Arcframe
arclist = glob.glob("../../GTCMULTIPLE2J-23A/" + OB + "/arc/*.fits")
print(arclist)
specdata = np.zeros((ysize, xsize), float)
for frames in arclist:
    spec = fits.open(str(frames))
    specdata += np.array(spec[0].data)
specdata = (specdata - BIAS) / FLAT
hdr = spec[0].header
center = int(centers[0])
specdata1 = specdata[center - 100 : center + 100, 25:2072]
hduout = fits.PrimaryHDU(specdata1)
hduout.header.extend(
    hdr, strip=True, update=True, update_first=False, useblanks=True, bottom=False
)
hduout.header["DISPAXIS"] = 1
hduout.header["CRVAL1"] = 1
hduout.header["CRVAL2"] = 1
hduout.header["CRPIX1"] = 1
hduout.header["CRPIX2"] = 1
hduout.header["CRVAL1"] = 1
hduout.header["CRVAL1"] = 1
hduout.header["CDELT1"] = 1
hduout.header["CDELT2"] = 1
hduout.writeto("../arcsub_std.fits", overwrite=True)
