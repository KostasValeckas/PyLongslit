import numpy
from astropy.io import fits
from logger import logger
from parser import detector_params, flat_params, output_dir
from utils import FileList, check_dimensions, open_fits, write_to_fits
from utils import show_overscan, show_flat_norm_region
import matplotlib.pyplot as plt

"""
Module for creating a master flat from from raw flat frames.
"""


def run_flats():

    # Extract the detector parameters
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    # overscan area (if any)
    overscan_x_start = detector_params["overscan_x_start"]
    overscan_x_end = detector_params["overscan_x_end"]
    overscan_y_start = detector_params["overscan_y_start"]
    overscan_y_end = detector_params["overscan_y_end"]

    # used defined area used for normalization
    norm_start_x = flat_params["norm_area_start_x"]
    norm_end_x = flat_params["norm_area_end_x"]
    norm_start_y = flat_params["norm_area_start_y"]
    norm_end_y = flat_params["norm_area_end_y"]

    # TODO: specify what direction is the spectral direction
    logger.info("Flat-field procedure running...")
    logger.info("Using the following parameters:")
    logger.info(f"xsize = {xsize}")
    logger.info(f"ysize = {ysize}")

    # read the names of the flat files from the directory
    file_list = FileList(flat_params["flat_dir"])

    if overscan_x_end != 0 and overscan_y_end != 0:
        logger.info("Non - zero overscan region is defined.")
        logger.info(
            f"Overscan region: x: {overscan_x_start}:{overscan_x_end},"
            f" y: {overscan_y_start}:{overscan_y_end}"
        )
        # Show the overscan region on a flat fram for Quality Assesment
        show_overscan()

    if norm_end_x != 0 and norm_end_y != 0:
        logger.info("Normalisation region is defined.")
        logger.info(
            f"Normalisation region: x: {norm_start_x}:{norm_end_x},"
            f" y: {norm_start_y}:{norm_end_y}"
        )
        # Show the normalisation region on a flat frame for Quality Assesment
        show_flat_norm_region()

    logger.info(f"Found {file_list.num_files} flat frames.")
    logger.info(f"Files used for flat-fielding:")
    print("------------------------------------")
    for file in file_list:
        print(file)
    print("------------------------------------")

    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # initialize a big array to hold all the flat frames for stacking
    bigflat = numpy.zeros((file_list.num_files, ysize, xsize), float)

    logger.info("Fetching the master bias frame...")

    try:
        BIASframe = fits.open(output_dir + "/master_bias.fits")
    except FileNotFoundError:

        try:
            BIASframe = fits.open(output_dir + "master_bias.fits")

        except FileNotFoundError:

            logger.critical("Master bias frame not found.")
            logger.error(
                "Make sure a master bias frame exists before proceeding with flats."
            )
            logger.error("Run the mkspecbias.py script first.")
            exit()

    BIAS = numpy.array(BIASframe[0].data)
    logger.info("Master bias frame found and loaded.")

    # loop over all the falt files, subtract the median value of the overscan,
    # subtract bias and stack them in the bigflat array
    for i, file in enumerate(file_list):

        rawflat = open_fits(flat_params["flat_dir"], file)

        logger.info(f"Processing file: {file}")

        data = numpy.array(rawflat[1].data)

        # TODO: if this is needed more - move it to utils
        if overscan_x_end != 0 and overscan_y_end != 0:
            overscan_mean = numpy.mean(
                data[overscan_y_start:overscan_y_end, overscan_x_start:overscan_y_end]
            )
            data = data - overscan_mean
            logger.info(
                f"Subtracted the median value of the overscan : {overscan_mean}"
            )

        data = data - BIAS
        logger.info("Subtracted the bias.")

        bigflat[i, 0 : ysize - 1, 0 : xsize - 1] = data[0 : ysize - 1, 0 : xsize - 1]

        # Normalise the frame

        #if normalization region is provided:
        if norm_end_x != 0 and norm_end_y != 0:
            norm = numpy.median(
                bigflat[i, norm_start_y:norm_end_y, norm_start_x:norm_end_x]
            )
        # if not , use the whole frame:
        else:
            norm = numpy.median(bigflat[i, :, :])

        logger.info(f"Normalising frame with the median of the frame :{norm}\n")
        bigflat[i, :, :] = bigflat[i, :, :] / norm

        # close the file handler
        rawflat.close()

    logger.info("Normalizing the final master flat-field....")

    # Calculate flat is median at each pixel
    medianflat = numpy.median(bigflat, axis=0)

    # Find a mean in spectral direction for each row
    lampspec = numpy.mean(medianflat, axis=1)

    for i in range(0, xsize - 1):
        medianflat[:, i] = medianflat[:, i] / lampspec[:]

    logger.info("Flat frames processed.")


    logger.info(
        "Mean pixel value of the final master flat-field: "
        f"{numpy.nanmean(medianflat)}"
    )

    # check if the median is 1 to within 5 decimal places
    if round(numpy.nanmean(medianflat),5) != 1.0:
        logger.warning("The mean pixel value of the final master flat-field is not 1.0.")
        logger.warning("This may indicate a problem with the normalisation.")
        logger.warning("Check the normalisation region in the flat-field frames.")


    logger.info("Attaching header and writing to disc...")

    # Write out result to fitsfile
    hdr = rawflat[0].header

    write_to_fits(medianflat, hdr, "master_flat.fits", output_dir)

    logger.info(
        f"Master flat frame written to disc in {output_dir}, filename master_flat.fits"
    )


if __name__ == "__main__":
    run_flats()
