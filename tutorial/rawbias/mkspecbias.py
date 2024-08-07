
import numpy
from astropy.io import fits

"""
This is a python program to make a BIAS frame
"""

# THESE HAVE TO BE SET MANUALLY
ysize = 2102
xsize = 500

print('Script running')
print('\n ---Using the following parameters:---\n')
print(f'ysize = {ysize}')
print(f'xsize = {xsize}')
print('----------------------------------------\n')

#Read in the raw bias frames and subtact mean of overscan region 

list = open('specbias.list')

nframes = len(list.readlines())
list.seek(0)

bigbias = numpy.zeros((nframes,ysize,xsize),float)
#bigbias = numpy.zeros((nframes,3,3))
for i in range(0,nframes):
   print('Image number:', i)
   rawbias = fits.open(str.rstrip(list.readline()))
   print('Info on file:')
   print(rawbias.info())
   data = numpy.array(rawbias[1].data)
   mean = numpy.mean(data[2066:ysize-5,0:xsize-1])
   data = data - mean
   print('Subtracted the median value of the overscan :',mean)
   bigbias[i-1,0:ysize-1,0:xsize-1] = data[0:ysize-1,0:xsize-1]
list.close()

##Calculate bias is median at each pixel
medianbias = numpy.median(bigbias,axis=0)

#Write out result to fitsfile
hdr = rawbias[0].header
fits.writeto('BIAS.fits',medianbias,hdr,overwrite=True)
