import numpy as np
from astropy.io import fits
from logger import logger
from parser import detector_params, flat_params, output_dir, data_params
from utils import FileList, check_dimensions, open_fits, write_to_fits
from utils import show_flat, list_files, load_bias, wavelength_sol
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from overscan import subtract_overscan_from_frame, detect_overscan_direction
from wavecalib import get_tilt_fit_from_disc, get_wavelen_fit_from_disc
from scipy.interpolate import make_lsq_spline, BSpline
from wavecalib import construct_wavelen_map
from utils import check_rotation, flip_and_rotate


"""
Module for creating a master flat from from raw flat frames.
"""

def fit_spectral_response(medianflat):

    y_size = detector_params["ysize"]
    x_size = detector_params["xsize"]

    #TODO: right now we flip and rotate to universal frame configuration
    # when master frames are reduced, so for flats we need to
    # account for that here. Consider flipping the raw frames instead
    # for more readable code.

    # extract the spectrum of the central 5 rows of the frame

    if detector_params["dispersion"]["spectral_dir"] == "x":

        # extract the spectrum of the central 5 rows of the frame
        middle_row = y_size // 2
        spectrum = np.mean(medianflat[middle_row-2: middle_row+2, :], axis=0)
        spectral_array = np.arange(x_size)
        spatial_array = np.arange(y_size)

    else:
            
        # extract the spectrum of the central 5 rows of the frame
        middle_row = x_size // 2
        spectrum = np.mean(medianflat[:, middle_row-2: middle_row+2], axis=1)
        spectral_array = np.arange(y_size)
        spatial_array = np.arange(x_size)


    wavelen_fit = get_wavelen_fit_from_disc()
    tilt_fit = get_tilt_fit_from_disc()

    middlew_row_array = np.full(len(spectral_array), middle_row)     

    wavelength = wavelength_sol(spectral_array, middlew_row_array, wavelen_fit, tilt_fit)


    print(wavelength)
    print(spectrum) 

    wavelength = wavelength
    spectrum = spectrum

    # Plot the original spectrum and the bspline fit
    plt.plot(wavelength, spectrum, label='Original Spectrum')
    plt.show()


    # Check for NaN or infinite values
    if np.any(np.isnan(wavelength)) or np.any(np.isnan(spectrum)):
        raise ValueError("NaN values found in wavelength or spectrum.")
    if np.any(np.isinf(wavelength)) or np.any(np.isinf(spectrum)):
        raise ValueError("Infinite values found in wavelength or spectrum.")

    # Ensure wavelength is sorted
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("Wavelength values are not sorted in ascending order.")

    num_interior_knots = (len(wavelength) - 2)//2
    
    # Create the knots array
    t = np.concatenate((
        np.repeat(wavelength[0], 4),  # k+1 knots at the beginning
        np.linspace(wavelength[1], wavelength[-2], num_interior_knots),  # interior knots
        np.repeat(wavelength[-1], 4)  # k+1 knots at the end
    )) 

    spl = make_lsq_spline(wavelength, spectrum, t=t, k=3)
    bspline = BSpline(spl.t, spl.c, spl.k)

    # Plot the bspline fit
    plt.plot(wavelength, spectrum, "+", label='BSpline Fit')
    plt.plot(wavelength, bspline(wavelength), label='BSpline Fit')
    plt.show()

    wave_map = construct_wavelen_map(wavelen_fit, tilt_fit)

    transpose, flip = check_rotation()

    print(wave_map)

    spectral_response_model = bspline(wave_map)

    plt.imshow(spectral_response_model)

    plt.show()

    spectral_response_model = flip_and_rotate(spectral_response_model, transpose, flip, inverse = True)

    plt.imshow(spectral_response_model)

    plt.show()

    normalized_flat = medianflat / spectral_response_model

    plt.imshow(normalized_flat)

    plt.show()

    return normalized_flat







def show_flat_norm_region():
    """
    Show the user defined flat normalization region.

    Fetches a raw flat frame from the user defined directory
    and displays the normalization region overlayed on it.
    """

    logger.info("Showing the normalization region on a raw flat frame for user inspection...")

    show_flat()

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
    'flat_dir' parameter in the 'config.json' file. It then subtracts the bias and normalizes the frames
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
    bigflat = np.zeros((file_list.num_files, ysize, xsize), float)

    if use_overscan:
        logger.warning("Using overscan subtraction instead of master bias.")
        logger.warning("If this is not intended, check the config file.")

        # get the overscan direction
        overscan_dir = detect_overscan_direction()

    else:

        logger.info("Fetching the master bias frame...")

        BIASframe = load_bias()

        BIAS = np.array(BIASframe[0].data)
        logger.info("Master bias frame found and loaded.")

    print("\n------------------------------------------------------------\n")

    # loop over all the falt files, subtract bias and stack them in the bigflat array
    for i, file in enumerate(file_list):

        rawflat = open_fits(flat_params["flat_dir"], file)

        logger.info(f"Processing file: {file}")

        data = np.array(rawflat[data_params["raw_data_hdu_index"]].data, dtype=np.float64)

        # Subtract the bias
        if use_overscan:
            data = subtract_overscan_from_frame(data, overscan_dir)
        else:
            data = data - BIAS
            logger.info("Subtracted the bias.")

        bigflat[i, 0 : ysize - 1, 0 : xsize - 1] = data[0 : ysize - 1, 0 : xsize - 1]

        # Normalise the frame

        #if normalization region is provided:
        if flat_params["user_custom_norm_area"]:
            norm = np.median(
                bigflat[i, norm_start_y:norm_end_y, norm_start_x:norm_end_x]
            )
        # if not , use the whole frame:
        else:
            norm = np.median(bigflat[i, :, :])

        logger.info(f"Normalising frame with the median of the frame :{norm}\n")
        bigflat[i, :, :] = bigflat[i, :, :] / norm

        # close the file handler
        rawflat.close()

        logger.info(f"File {file} processed.\n")

    logger.info("Normalizing the final master flat-field....")

    # Calculate flat is median at each pixel
    medianflat = np.median(bigflat, axis=0)

    medianflat = fit_spectral_response(medianflat)


    logger.info("Flat frames processed.")


    logger.info(
        "Mean pixel value of the final master flat-field: "
        f"{round(np.nanmean(medianflat),5)} (should be 1.0)."
    )

    # check if the median is 1 to within 5 decimal places
    if round(np.nanmean(medianflat),5) != 1.0:
        logger.warning("The mean pixel value of the final master flat-field is not 1.0.")
        logger.warning("This may indicate a problem with the normalisation.")
        logger.warning("Check the normalisation region in the flat-field frames.")

    if use_overscan:
        # if overscan was used, set the overscan region to one to avoid
        # explosive values in the final flat

        # Extract the overscan region
        overscan_x_start = detector_params["overscan"]["overscan_x_start"]
        overscan_x_end = detector_params["overscan"]["overscan_x_end"]
        overscan_y_start = detector_params["overscan"]["overscan_y_start"]
        overscan_y_end = detector_params["overscan"]["overscan_y_end"]

        medianflat[overscan_y_start:overscan_y_end, overscan_x_start:overscan_x_end] = 1.0


    logger.info("Attaching header and writing to disc...")

    # Write out result to fitsfile
    hdr = rawflat[0].header

    write_to_fits(medianflat, hdr, "master_flat.fits", output_dir)

    logger.info(
        f"Master flat frame written to disc in {output_dir}, filename master_flat.fits"
    )


if __name__ == "__main__":
    run_flats()
