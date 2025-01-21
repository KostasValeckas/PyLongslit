# This is a python program to make a BIAS frame
print("Script running")
import astropy
import numpy as np
from astropy.io import fits
import glob

# Read information in info.txt
file = open("info.txt", "r")
OB = file.readline()
ysize = file.readline()
xsize = file.readline()
OB = OB.strip("\n")
xsize = int(xsize)
ysize = int(ysize)

# Read in the raw bias frames and subtact mean of overscan region
list = glob.glob("../../GTCMULTIPLE2J-23A/" + OB + "/bias/*fits")
nframes = len(list)
print("../../GTCMULTIPLE2J-23A/" + OB + "/bias/*fits")
bigbias = np.zeros((nframes, ysize, xsize), float)
i = 0
for frame in list:
    print("Image:", frame)
    rawbias = fits.open(str(frame))
    print("Info on file:")
    print(rawbias.info())
    data = np.array(rawbias[0].data)
    bigbias[i - 1, 0 : ysize - 1, 0 : xsize - 1] = data[0 : ysize - 1, 0 : xsize - 1]
    i += 1

##Calculate bias is median at each pixel
medianbias = np.median(bigbias, axis=0)

# Write out result to fitsfile
hdr = rawbias[0].header
fits.writeto("BIAS.fits", medianbias, hdr, overwrite=True)
