import numpy
from astropy.io import fits
from logger import logger
from parser import detector_params, bias_params
from utils import FileList
from utils import check_dimensions

"""
Module for creating a master bias frame from raw bias frames.
"""

def run_bias():

    # Extract the detector parameters

    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]


    logger.info("Bias procedure running...")
    logger.info("Using the following parameters:")
    logger.info(f"xsize = {xsize}")
    logger.info(f"ysize = {ysize}")
    #TODO: specify what direction is the spectral direction

    # read the names of the bias files from the list
    file_list = FileList(bias_params["bias_dir"])

    logger.info(f"Found {file_list.num_files} bias frames.")
    logger.info(f"Files used for bias processing")

    print("\n")
    for file in file_list: print(file)
    print("\n")

    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # Read in the raw bias frames and subtact mean of overscan region
    bigbias = numpy.zeros((file_list.num_files, ysize, xsize), float)
    # bigbias = numpy.zeros((nframes,3,3))
    for i,file in enumerate(file_list):

        # user might have forgotten to add a slash at the end of the path
        try:
            rawbias = fits.open(bias_params["bias_dir"] + file)
        except FileNotFoundError:
            rawbias = fits.open(bias_params["bias_dir"] + "/" + file)
        
        logger.info(f"Processing file: {file}")
        

        data = numpy.array(rawbias[1].data)

        mean = numpy.mean(data[2066 : ysize - 5, 0 : xsize - 1])


        data = data - mean
        print(f"Subtracted the median value of the overscan : {mean}\n")

        bigbias[i, 0 : ysize - 1, 0 : xsize - 1] = data[0 : ysize - 1, 0 : xsize - 1]

        #close the file handler
        rawbias.close()



    ##Calculate bias is median at each pixel
    medianbias = numpy.median(bigbias, axis=0)

    # Write out result to fitsfile
    hdr = rawbias[0].header
    fits.writeto("BIAS.fits", medianbias, hdr, overwrite=True)

if __name__ == "__main__":
    run_bias()
