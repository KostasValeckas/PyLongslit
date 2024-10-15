import numpy
from astropy.io import fits
from logger import logger
from parser import detector_params, flat_params, output_dir, data_params
from utils import FileList, check_dimensions, open_fits, write_to_fits
from utils import show_flat, list_files
from overscan import subtract_overscan_from_frame
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""
Module for creating a master flat from from raw flat frames.
"""

def show_flat_norm_region():
    """
    Show the user defined flat normalization region.

    Fetches a raw flat frame from the user defined directory
    and displays the normalization region overlayed on it.
    """

    logger.info("Showing the normalization region on a raw flat frame for user inspection...")

    show_flat()

    # Add rectangular box to show the overscan region
    width = flat_params["norm_area_end_x"] \
                - flat_params["norm_area_start_x"]
    height = flat_params["norm_area_end_y"] \
                - flat_params["norm_area_start_y"]

    rect = Rectangle(
        (flat_params["norm_area_start_x"],
         flat_params["norm_area_start_y"]),
        width,
        height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
        linestyle="--",
        label="Region used for estimation of normalization factor",
    )
    plt.gca().add_patch(rect)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("Pixels in x-direction")
    plt.ylabel("Pixels in y-direction")
    plt.title(
        "Region used for estimation of normalization factor overlayed on a raw flat frame.\n"
        "The region should somewhat brightly illuminated with no abnormalities or artifacts.\n"
        "If it is not, check the normalization region definition in the config file."
    )
    plt.show()



def run_flats():

    """
    Driver for the flat-fielding procedure.

    The function reads the raw flat frames from the directory specified in the
    'flat_dir' parameter in the 'config.json' file. It then subtracts the
    overscan region, subtracts the master bias frame and normalizes the frames
    by the median value of the frame. The final master flat-field is written to
    disc in the output directory.
    """

    # Extract the detector parameters
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    use_overscan = detector_params["overscan"]["use_overscan"]

    if flat_params["user_custom_norm_area"]:
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

    logger.info(f"Found {file_list.num_files} flat frames.")
    logger.info(f"Files used for flat-fielding:")
    list_files(file_list)

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

        data = numpy.array(rawflat[data_params["raw_data_hdu_index"]].data)

        # TODO: if this is needed more - move it to utils
        if use_overscan:
            
            data = subtract_overscan_from_frame(data)

        data = data - BIAS
        logger.info("Subtracted the bias.")

        bigflat[i, 0 : ysize - 1, 0 : xsize - 1] = data[0 : ysize - 1, 0 : xsize - 1]

        # Normalise the frame

        #if normalization region is provided:
        if flat_params["user_custom_norm_area"]:
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

        logger.info(f"File {file} processed.\n")

    logger.info("Normalizing the final master flat-field....")

    # Calculate flat is median at each pixel
    medianflat = numpy.median(bigflat, axis=0)

    #TODO: right now we flip and rotate to universal frame configuration
    # when master frames are reduced, so for illumination flats we need to
    # account for that here. Consider flipping the raw frames instead
    # for more readable code.
    if detector_params["dispersion"]["spectral_dir"] == "x":

        # Find a mean in spectral direction for each row
        lampspec = numpy.mean(medianflat, axis=0)

        for i in range(0, ysize - 1):
            medianflat[i, :] = medianflat[i, :] / lampspec[:]

    else:
            
        # Find a mean in spectral direction for each column
        lampspec = numpy.mean(medianflat, axis=1)
    
        for i in range(0, xsize - 1):
            medianflat[:, i] = medianflat[:, i] / lampspec[:]

    logger.info("Flat frames processed.")


    logger.info(
        "Mean pixel value of the final master flat-field: "
        f"{round(numpy.nanmean(medianflat),5)} (should be 1.0)."
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
