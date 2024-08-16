import numpy
from astropy.io import fits
from logger import logger
from parser import detector_params, bias_params, output_dir
from utils import FileList, check_dimensions, show_overscan, open_fits, write_to_fits

"""
Module for creating a master bias frame from raw bias frames.
"""

def run_bias():

    """
    Driver for the bias procedure. 

    The function reads the raw bias frames from the directory specified in the
    'bias_dir' parameter in the 'config.json' file. It then subtracts the
    overscan region, stacks the frames and calculates the median value at each
    pixel. The final master bias frame is written to disc in the output directory.
    """

    # Extract the detector parameters
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    overscan_x_start = detector_params["overscan_x_start"]
    overscan_x_end = detector_params["overscan_x_end"]
    overscan_y_start = detector_params["overscan_y_start"]
    overscan_y_end = detector_params["overscan_y_end"]

    # TODO: specify what direction is the spectral direction
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

        # Show the overscan region on a flat fram for Quality Assesment

        show_overscan()

    logger.info(f"Found {file_list.num_files} bias frames.")
    logger.info(f"Files used for bias processing:")

    print("------------------------------------")
    for file in file_list:
        print(file)
    print("------------------------------------")

    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # initialize a big array to hold all the bias frames for stacking
    bigbias = numpy.zeros((file_list.num_files, ysize, xsize), float)

    # loop over all the bias files, subtract the median value of the overscan
    # and stack them in the bigbias array
    for i, file in enumerate(file_list):

        rawbias = open_fits(bias_params["bias_dir"], file)

        logger.info(f"Processing file: {file}")

        data = numpy.array(rawbias[1].data)

        # TODO: if this is needed more - move it to utils
        if overscan_x_end != 0 and overscan_y_end != 0:
            overscan_mean = numpy.mean(
                data[overscan_y_start:overscan_y_end, overscan_x_start:overscan_y_end]
            )

            data = data - overscan_mean
            logger.info(
                f"Subtracted the median value of the overscan : {overscan_mean}\n"
            )

        bigbias[i, 0 : ysize - 1, 0 : xsize - 1] = data[0 : ysize - 1, 0 : xsize - 1]

        # close the file handler
        rawbias.close()

    # Calculate bias as median at each pixel
    medianbias = numpy.median(bigbias, axis=0)

    logger.info("Bias frames processed.")
    logger.info("Attaching header and writing to disc...")

    # Write out result to fitsfile
    hdr = rawbias[0].header

    write_to_fits(medianbias, hdr, "master_bias.fits", output_dir)

    logger.info(
        f"Master bias frame written to disc at in {output_dir}, filename master_bias.fits"
    )


if __name__ == "__main__":
    run_bias()
