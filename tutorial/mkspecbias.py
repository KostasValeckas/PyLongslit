import numpy
from astropy.io import fits
from logger import logger
from parser import detector_params, bias_params
from astropy.visualization import ImageNormalize, LogStretch
import matplotlib.pyplot as plt
from utils import FileList, check_dimensions, create_grid_pattern
import numpy as np

"""
Module for creating a master bias frame from raw bias frames.
"""

def run_bias():

    # Extract the detector parameters

    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    overscan_x_start = detector_params["overscan_x_start"]
    overscan_x_end = detector_params["overscan_x_end"]
    overscan_y_start = detector_params["overscan_y_start"]
    overscan_y_end = detector_params["overscan_y_end"]

    #TODO: specify what direction is the spectral direction
    logger.info("Bias procedure running...")
    logger.info("Using the following parameters:")
    logger.info(f"xsize = {xsize}")
    logger.info(f"ysize = {ysize}")
    
    # read the names of the bias files from the directory
    file_list = FileList(bias_params["bias_dir"])

    if overscan_x_end != 0 and overscan_y_end != 0:
        logger.info("Non - zero overscan region is defined.")
        logger.info(
            f"Overscan region: x: {overscan_x_start}:{overscan_x_end}," 
            f" y: {overscan_y_start}:{overscan_y_end}"
        )

        # show the selected overscan region for the user to inspect  

        #dummy pattern to illustrate the detector:
        tile = create_grid_pattern(xsize, ysize)
        
        # Display the tile
        plt.imshow(tile, cmap='gray')
        plt.gca().invert_yaxis()
        plt.title('Uniform Vertically Lined Tile')
        plt.xlabel('Pixels in x direction')
        plt.ylabel('Pixels in y direction')
        plt.show()


    logger.info(f"Found {file_list.num_files} bias frames.")
    logger.info(f"Files used for bias processing:")

    print("------------------------------------")
    for file in file_list: print(file)
    print("------------------------------------")

    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # initialize a big array to hold all the bias frames for stacking
    bigbias = numpy.zeros((file_list.num_files, ysize, xsize), float)

    # loop over all the bias files, subtract the median value of the overscan
    # and stack them in the bigbias array
    for i,file in enumerate(file_list):

        try:
            rawbias = fits.open(bias_params["bias_dir"] + file)
        # user might have forgotten to add a slash at the end of the path
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
