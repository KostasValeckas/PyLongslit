import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import make_lsq_spline, BSpline

"""
PyLongslit Module for creating a master flat from raw flat frames.
"""


def estimate_spectral_response(medianflat):

    """
    Estimates the flat-field lamp spectral response (later used in normalization)
    from a median flat-field frame. This is done by fitting a B-spline to the
    spectrum of the flat-field lamp.

    Parameters
    ----------
    medianflat : numpy.ndarray
        The median flat-field frame.

    Returns
    -------
    spectral_response_model : numpy.ndarray
        The 1D spectral response model.
    
    bpm : numpy.ndarray
        The bad pixel mask for the spectral response model.

    RMS : float
        The root mean square of the residuals of the fit.
    """

    from pylongslit.parser import detector_params, wavecalib_params, flat_params
    from pylongslit.wavecalib import get_tilt_fit_from_disc, get_wavelen_fit_from_disc, construct_wavelen_map
    from pylongslit.utils import wavelength_sol, show_1d_fit_QA, check_rotation
    from pylongslit.utils import flip_and_rotate, interactively_crop_spec
    from pylongslit.logger import logger

    y_size = detector_params["ysize"]
    x_size = detector_params["xsize"]

    # for taking a sample of the flat lamp,
    # offset is needed for cases where the detector middle is a bad place to
    # take a cut. 
    middle_offset = wavecalib_params["offset_middle_cut"]
    

    # the user can set a extraction width for the sample of the flat lamp
    pixel_cut_extension = wavecalib_params["pixel_cut_extension"]


    # TODO: right now we flip and rotate to universal frame configuration
    # when master frames are reduced, so for flats we need to
    # account for that here. Consider flipping the raw frames instead
    # for more readable code.

    # extract the spectrum of the flat lamp and allign with the orientation

    if detector_params["dispersion"]["spectral_dir"] == "x":

        middle_row = (y_size // 2) + middle_offset
        spectrum = np.mean(medianflat[middle_row - pixel_cut_extension : middle_row + pixel_cut_extension + 1, :], axis=0)
        spectral_array = np.arange(x_size)

        # flip the spectrum to have it in the right order if needed
        if not detector_params["dispersion"]["wavelength_grows_with_pixel"]:
            spectrum = spectrum[::-1]

    else:

        middle_row = (x_size // 2) + middle_offset
        spectrum = np.mean(medianflat[:, middle_row - pixel_cut_extension : middle_row + pixel_cut_extension + 1], axis=1)
        spectral_array = np.arange(y_size)

        # for y spectra we need to flip if it does grow with pixel, due to 
        # the way the way numpy indexes arrays
        if detector_params["dispersion"]["wavelength_grows_with_pixel"]:
            spectrum = spectrum[::-1]

    # read the wavecalib data
    wavelen_fit = get_wavelen_fit_from_disc()
    tilt_fit = get_tilt_fit_from_disc()

    # this is the y-coordinate of the middle row of the detector for the wavelength solution
    middlew_row_array = np.full(len(spectral_array), middle_row + middle_offset)

    # create the wavelength array
    wavelength = wavelength_sol(
        spectral_array, middlew_row_array, wavelen_fit, tilt_fit
    )

    # Mask NaN or infinite values in the spectrum and corresponding wavelength values
    mask = np.isnan(spectrum) | np.isinf(spectrum)
    spectrum = spectrum[~mask]
    wavelength = wavelength[~mask]

    # Ensure wavelength is sorted
    if not np.all(np.diff(wavelength) > 0):
        logger.error("Wavelength values are not sorted in ascending order.")
        logger.error("Please check the wavelength solution.")
        logger.error("Contact the developers if the wavelength solution is correct.")   

    # crop the spec for noisy end-pieces
    min_wave, max_wave = interactively_crop_spec(
        wavelength, spectrum,
        x_label="Wavelength (Å)",
        y_label="Counts (ADU)",
        label="Flat-field lamp spectrum",
        title=
        "Use sliders to crop out any noisy parts on detector edges.\n"
        "Press \"Q\" or close the window when done."
    )

    # Get the final selected range
    min_wavelength = min_wave
    max_wavelength = max_wave
    valid_indices = (wavelength >= min_wavelength) & (wavelength <= max_wavelength)
    wavelength_cut = wavelength[valid_indices]
    spectrum_cut = spectrum[valid_indices]

    # setup the B-spline fit
    num_interior_knots = flat_params["knots_spectral_bspline"]

    # check that the number of knots is reasonable
    if num_interior_knots > len(wavelength_cut) // 2:
        logger.warning(
            "The number of interior knots is larger than half the number of data points."
        )
        logger.warning("This may lead to overfitting.")
        logger.warning("Consider reducing the number of knots in the configuration file.")

    if num_interior_knots >= len(wavelength_cut):
        logger.error(
            "The number of interior knots is larger than the number of data points."
        )
        logger.error("This will lead to overfitting.")
        logger.error("Please reduce the number of knots in the configuration file.")
        exit()

    if num_interior_knots < 4:
        logger.error(
            "The number of interior knots is less than 4."
        )
        logger.error("This will lead to underfitting.")
        logger.error("Please increase the number of knots in the configuration file.")
        exit()

    # Create the knots array
    t = np.concatenate(
        (
            np.repeat(wavelength_cut[0], 4),  # k+1 knots at the beginning
            np.linspace(
                wavelength_cut[1], wavelength_cut[-2], num_interior_knots
            ),  # interior knots
            np.repeat(wavelength_cut[-1], 4),  # k+1 knots at the end
        )
    )
    
    # this part does the actual fitting
    spl = make_lsq_spline(wavelength_cut, spectrum_cut, t=t, k=3)
    bspline = BSpline(spl.t, spl.c, spl.k)

    residuals = spectrum_cut - bspline(wavelength_cut)
    RMS = np.sqrt(np.mean(residuals ** 2))

    show_1d_fit_QA(
        wavelength_cut,
        spectrum_cut,
        x_fit_values=wavelength_cut,
        y_fit_values=bspline(wavelength_cut),
        residuals=residuals,
        x_label="Wavelength (Å)",
        y_label="Counts (ADU)",
        legend_label="Extracted flat-field lamp spectrum",
        title=
            f"Spectral response B-spline fit with {num_interior_knots} interior knots.\n"
            "You should see very little to no large-scale structure in the residuals, \n"
            "with the lowest amount of knots possible (this is set in the configuration file).",
    )

    # construct the model - map the spectral response to every pixel
    wave_map = construct_wavelen_map(wavelen_fit, tilt_fit)

    transpose, flip = check_rotation()

    spectral_response_model = bspline(wave_map)

    # mark the pixels corresponding to the cropped wavelengths
    bpm = np.zeros_like(spectral_response_model, dtype=bool)

    bpm[wave_map < min_wavelength] = True
    bpm[wave_map > max_wavelength] = True

    spectral_response_model[bpm] = 1.0

    # flip back to original position of the raw data
    spectral_response_model = flip_and_rotate(
        spectral_response_model, transpose, flip, inverse=True
    )
    bpm = flip_and_rotate(bpm, transpose, flip, inverse=True)

    return spectral_response_model, bpm, RMS


def estimate_spacial_response(medianflat):
 
    from pylongslit.parser import detector_params, flat_params
    from pylongslit.utils import interactively_crop_spec

    y_size = detector_params["ysize"]
    x_size = detector_params["xsize"]

    # TODO: right now we flip and rotate to universal frame configuration
    # when master frames are reduced, so for flats we need to
    # account for that here. Consider flipping the raw frames instead
    # for more readable code.

    # estimate orientaition
    spectral_axis = 0 if detector_params["dispersion"]["spectral_dir"] == "x" else 1
    spacial_axis = 1 if detector_params["dispersion"]["spectral_dir"] == "x" else 0

    middle_pixel = x_size // 2 if spectral_axis == 0 else y_size // 2

    # first, do some initial cropping to get rid of the noisy edges

    test_slice = (
            medianflat[:, middle_pixel].copy()
            if spacial_axis == 1
            else medianflat[middle_pixel, :].copy()
        )
    
    spectral_slice = np.arange(len(test_slice))
    
    min_spat, max_spat = interactively_crop_spec(
        spectral_slice, test_slice,
        x_label="Spacial pixel",
        y_label="Counts (ADU)",
        label=f"Flat-field frame cut at spacial puxel {middle_pixel}",
        title=
        "Use sliders to crop out any noisy parts on detector edges.\n"
        "Press \"Q\" or close the window when done."
    )

   
    # 5 rows will be plotted for QA
    fig, ax = plt.subplots(5, 2, figsize=(18, 32))

    indices_to_plot = np.linspace(
        5, x_size if spectral_axis == 0 else y_size, 5, endpoint=False, dtype=int
    )

    plot_num = 0 # a bit hacked solution for plotting

    spacial_model = np.zeros((y_size, x_size))

    residuals = np.zeros((y_size, x_size))

    # loop over spacial columns and fit for every column

    #used when fitting with cropped edges
    spectral_array_cropped = np.arange(min_spat, max_spat)

    # used when constructing the model
    spectral_array = np.arange(y_size) if spectral_axis == 0 else np.arange(x_size)

    residuals = np.array([])

    for spacial_row_index in range(x_size) if spectral_axis == 0 else range(y_size):

        spacial_slice = (
            medianflat[:, spacial_row_index].copy()
            if spacial_axis == 1
            else medianflat[spacial_row_index, :].copy()
        )

        # crop the spacial slice for user defined region
        spacial_slice_cropped = spacial_slice[min_spat:max_spat]

        # remove outliers
        mean = np.nanmean(spacial_slice_cropped)
        two_std = 2 * np.nanstd(spacial_slice_cropped)
        mask = np.isnan(spacial_slice_cropped) | np.isinf(spacial_slice_cropped) | (spacial_slice_cropped > (mean + two_std)) | (spacial_slice_cropped < (mean - two_std))
        

        spacial_slice_masked = spacial_slice_cropped[~mask]
        spectral_array_masked = spectral_array_cropped[~mask]

        num_interior_knots = flat_params["knots_spacial_bspline"]

        # Create the knots array
        t = np.concatenate(
            (
                np.repeat(spectral_array_masked[0], 4),  # k+1 knots at the beginning
                np.linspace(
                    spectral_array_masked[1], spectral_array_masked[-2], num_interior_knots
                ),  # interior knots
                np.repeat(spectral_array_masked[-1], 4),  # k+1 knots at the end
            )
        )

        # do the fit
        spl = make_lsq_spline(spectral_array_masked, spacial_slice_masked, t=t, k=3)
        bspline = BSpline(spl.t, spl.c, spl.k)

        residuals_temp = spacial_slice_cropped - bspline(spectral_array_cropped)

        residuals = np.append(residuals, residuals_temp)

        # evaluate the fit at the column and calculate residuals
        if spacial_axis == 1:
            spacial_model[:, spacial_row_index] = bspline(spectral_array)

        else:
            spacial_model[spacial_row_index, :] = bspline(spectral_array)

        if spacial_row_index in indices_to_plot:
            if plot_num <= 4:
                ax[plot_num, 0].plot(
                    spectral_array_masked, spacial_slice_masked, ".", label=f"Data at spacial pixel: {spacial_row_index}"
                )
                ax[plot_num, 0].plot(
                    spectral_array_masked, bspline(spectral_array_masked), label="Fit"
                )
                ax[plot_num, 0].plot(
                    spectral_array_cropped[mask], spacial_slice_cropped[mask], 'o', color='red', label="Masked outliers"
                )

                ax[plot_num, 0].legend()

  

                ax[plot_num, 1].plot(
                    spectral_array_cropped, residuals_temp, 'o', color='black', label=f"Residuals at spacial pixel: {spacial_row_index}"
                )
                ax[plot_num, 1].axhline(0, color='red', linestyle='--')

                ax[plot_num, 1].legend()

                plot_num += 1

    fig.suptitle(
        "Slit illumination B-spline fits at different spectral pixels.\n"
        f"Number of interior knots: {num_interior_knots} (this is set in the configuration file).\n"
        "You should see very little to no large-scale structure in the residuals, with the lowest amount of knots possible.",
        fontsize=16,
        y=1.02
    )
    fig.text(0.5, 0.04, "Spacial pixel", ha="center", fontsize=16)
    fig.text(
        0.04,
        0.5,
        "Normalized Counts (ADU)",
        va="center",
        rotation="vertical",
        fontsize=16,
    )
    plt.show()

    RMS = np.sqrt(np.nanmean(residuals ** 2))

    return spacial_model, RMS


def show_flat_norm_region():
    """
    NOT USED

    Show the user defined flat normalization region.

    Fetches a raw flat frame from the user defined directory
    and displays the normalization region overlayed on it.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import flat_params
    from pylongslit.utils import show_flat

    logger.info(
        "Showing the normalization region on a raw flat frame for user inspection..."
    )

    show_flat()

    width = flat_params["norm_area_end_x"] - flat_params["norm_area_start_x"]
    height = flat_params["norm_area_end_y"] - flat_params["norm_area_start_y"]

    rect = Rectangle(
        (flat_params["norm_area_start_x"], flat_params["norm_area_start_y"]),
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
    'flat_dir' parameter in the 'config.json' file. It subtracts the bias from
    the raw frames, constructs a median master flat, and then normalizes it 
    by the spectral and (optionally) the spacial response of the flat-field lamp.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import detector_params, flat_params, data_params
    from pylongslit.utils import FileList, check_dimensions, open_fits, PyLongslit_frame
    from pylongslit.utils import load_bias
    from pylongslit.overscan import estimate_frame_overscan_bias
    from pylongslit.stats import bootstrap_median_errors_framestack

    # Extract the detector parameters
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    use_overscan = detector_params["overscan"]["use_overscan"]

    logger.info("Flat-field procedure running...")
    logger.info("Using the following parameters:")
    logger.info(f"xsize = {xsize}")
    logger.info(f"ysize = {ysize}")

    # read the names of the flat files from the directory
    file_list = FileList(flat_params["flat_dir"])

    logger.info(f"Found {file_list.num_files} flat frames.")
    logger.info(f"Files used for flat-fielding:")
    file_list.print_files()

    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # initialize a big array to hold all the flat frames for stacking
    bigflat = np.zeros((file_list.num_files, ysize, xsize), float)

    logger.info("Fetching the master bias frame...")

    BIASframe = PyLongslit_frame.read_from_disc("master_bias.fits")

    BIAS = np.array(BIASframe.data)
    logger.info("Master bias frame found and loaded.")

    print("\n------------------------------------------------------------\n")

    # loop over all the falt files, subtract bias and stack them in the bigflat array
    for i, file in enumerate(file_list):

        rawflat = open_fits(flat_params["flat_dir"], file)

        logger.info(f"Processing file: {file}")

        data = np.array(
            rawflat[data_params["raw_data_hdu_index"]].data, dtype=np.float64
        )

        # Subtract the bias
        if use_overscan:
            overscan = estimate_frame_overscan_bias(data, plot=False)
            data = data - overscan.data

        data = data - BIAS
        logger.info("Subtracted the bias.")

        
        bigflat[i] = data

        # close the file handler
        rawflat.close()

        logger.info(f"File {file} processed.\n")

    
    # Calculate flat is median at each pixel
    medianflat = np.median(bigflat, axis=0)

    # Error estimation depending on user-chosen method

    if file_list.num_files < 30 and (not flat_params["bootstrap_errors"]):
        logger.warning(
            f"Number of flat frames ({file_list.num_files}) is less than 30. Error estimation might not be accurate."
        )
        logger.warning("Please consider taking more flat frames or activating error bootstrapping in the config file.")
   
    if  not flat_params["bootstrap_errors"]:
        medianflat_error =  1.2533*np.std(bigflat, axis=0)/np.sqrt(file_list.num_files)

    else:
        medianflat_error = bootstrap_median_errors_framestack(bigflat)

    logger.info("Estimating the spectral response and normalizing...")    

    spectral_response_model, _, RMS_spectral = estimate_spectral_response(medianflat)

    spectral_normalized = medianflat / spectral_response_model


    medianflat_error = spectral_normalized * np.sqrt(
        ((medianflat_error / medianflat)) ** 2 + ((RMS_spectral/spectral_response_model) ** 2)
    )

    # correct any outliers (these are ususally the non-illuminated parts of the flat)
    spectral_normalized[spectral_normalized < 0.5] = 1
    spectral_normalized[spectral_normalized > 1.5] = 1

    # if requested, do the spacial response normalization
    if not flat_params["skip_spacial"]:
        logger.info("Estimating the spacial response and normalizing...")    
        spacial_response_model, RMS_spacial = estimate_spacial_response(spectral_normalized)
        master_flat = spectral_normalized / spacial_response_model
        medianflat_error = master_flat * np.sqrt(
            ((medianflat_error / medianflat)) ** 2 + ((RMS_spacial/spacial_response_model) ** 2)
        )   

    else:

        master_flat = spectral_normalized

    # the below code sets outliers to 1 - these are usually the non-illuminated parts of the flat

    if use_overscan:
        # if overscan was used, set the overscan region to one to avoid
        # explosive values in the final flat

        # Extract the overscan region
        overscan_x_start = detector_params["overscan"]["overscan_x_start"]
        overscan_x_end = detector_params["overscan"]["overscan_x_end"]
        overscan_y_start = detector_params["overscan"]["overscan_y_start"]
        overscan_y_end = detector_params["overscan"]["overscan_y_end"]

        master_flat[overscan_y_start:overscan_y_end, overscan_x_start:overscan_x_end] = (
            1.0
        )

        medianflat_error[overscan_y_start:overscan_y_end, overscan_x_start:overscan_x_end] = (
            np.nanmean(medianflat_error)
        )

    
    master_flat[master_flat < 0.5] = 1.0
    master_flat[master_flat > 1.5] = 1.0

    master_flat[np.isnan(master_flat)] = 1.0
    medianflat_error[np.isnan(medianflat_error)] = np.nanmean(medianflat_error)

    medianflat_error[master_flat < 0.5] = np.nanmean(medianflat_error)
    medianflat_error[master_flat > 1.5] = np.nanmean(medianflat_error)




    fig, ax = plt.subplots(5 if not flat_params["skip_spacial"] else 3, 2, figsize=(18, 32))

    # only show positive values to avoid outliers that disorts the color map
    ax[0][0].imshow(np.clip(medianflat.T, 0, None), cmap="gray", origin="lower")
    ax[0][0].set_title("Master flat prior to normalization")
    ax[0][0].axis("off")

    ax[1][0].imshow(spectral_response_model.T, cmap="gray", origin="lower")
    ax[1][0].set_title("2D spectral response model")
    ax[1][0].axis("off")

    ax[2][0].imshow(spectral_normalized.T, cmap="gray", origin="lower")
    ax[2][0].set_title("Master flat normalized by spectral response model")
    ax[2][0].axis("off")

    if not flat_params["skip_spacial"]:
        ax[3][0].imshow(spacial_response_model.T, cmap="gray", origin="lower")
        ax[3][0].set_title("2D slit illumination model")
        ax[3][0].axis("off")

        ax[4][0].imshow(master_flat.T, cmap="gray", origin="lower")
        ax[4][0].set_title(
            "Final master flat - normalized by slit illumination and spectral response models"
        )
        ax[4][0].axis("off")

    N_bins = int(np.sqrt(len(medianflat.flatten())))

    ax[0][1].hist(
        medianflat.flatten(),
        bins=N_bins,
        range=(0, np.max(medianflat)),
        color="black",
    )
    ax[1][1].hist(spectral_response_model.flatten(), bins=N_bins, color="black")
    ax[2][1].hist(spectral_normalized.flatten(), bins=N_bins, color="black")
    if not flat_params["skip_spacial"]:
        ax[3][1].hist(spacial_response_model.flatten(), bins=N_bins, color="black")
        ax[4][1].hist(master_flat.flatten(), bins=N_bins, color="black")

    for a in ax[:, 1]:
        a.set_ylabel("N pixels")
    for a in ax[-1, :]:
        a.set_xlabel("Counts (ADU)")
  
    fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
    fig.align_ylabels(ax[:, 1])

    fig.suptitle(
        "Make sure the models match the data, and that final flat field pixel "
        "sensitivity distribution is somewhat Gaussian.",
        fontsize=16, y = 1
    )

    plt.show()

    logger.info(
        "Mean pixel value of the final master flat-field: "
        f"{round(np.nanmean(master_flat),5)} +/- {np.std(master_flat)/np.sqrt(len(master_flat))} (should be 1.0)."
    )

    # check if the median is 1 to within 5 decimal places
    if round(np.nanmean(master_flat), 1) != 1.0:
        logger.warning(
            "The mean pixel value of the final master flat-field is not 1.0."
        )
        logger.warning("This may indicate a problem with the normalisation.")
        logger.warning("Check the normalisation region in the flat-field frames.")

    logger.info("Attaching header and writing to disc...")

    # Write out result to fitsfile
    hdr = rawflat[0].header

    master_flat_frame = PyLongslit_frame(master_flat, medianflat_error, hdr, "master_flat")

    master_flat_frame.show_frame(save=True)
    master_flat_frame.write_to_disc()

    

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit flatfield procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_flats()


if __name__ == "__main__":
    main()
