import numpy
from logger import logger
from parser import detector_params, bias_params, output_dir, data_params
from utils import FileList, check_dimensions, open_fits, write_to_fits
from utils import list_files
from overscan import check_overscan, subtract_overscan_from_frame
"""
Module for creating a master bias frame from raw bias frames.
"""

def run_bias():

    """
    Driver for the bias procedure. 

    The function reads the raw bias frames from the directory specified in the
    'bias_dir' parameter in the 'config.json' file. It then stacks the frames and calculates the median value at each
    pixel. The final master bias frame is written to disc in the output directory.
    """

    # Extract the detector parameters
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    # TODO: specify what direction is the spectral direction
    logger.info("Bias procedure running...")
    logger.info("Using the following parameters:")
    logger.info(f"xsize = {xsize}")
    logger.info(f"ysize = {ysize}")

    # read the names of the bias files from the directory
    file_list = FileList(bias_params["bias_dir"])

    logger.info(f"Found {file_list.num_files} bias frames.")
    logger.info(f"Files used for bias processing:")

    list_files(file_list)

    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # initialize a big array to hold all the bias frames for stacking
    bigbias = numpy.zeros((file_list.num_files, ysize, xsize), float)

    use_overscan = check_overscan()
    # loop over all the bias files and stack them in the bigbias array
    for i, file in enumerate(file_list):

        rawbias = open_fits(bias_params["bias_dir"], file)

        logger.info(f"Processing file: {file}")

        data = numpy.array(rawbias[data_params["raw_data_hdu_index"]].data)

        if use_overscan:
            data = subtract_overscan_from_frame(data)

        bigbias[i, 0 : ysize - 1, 0 : xsize - 1] = \
            data[0 : ysize - 1, 0 : xsize - 1]

        # close the file handler
        rawbias.close()

        logger.info(f"File {file} processed.\n")

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
