import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import make_lsq_spline, BSpline
from sklearn.metrics import r2_score
from scipy.interpolate import griddata

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
    from pylongslit.wavecalib import (
        get_tilt_fit_from_disc,
        get_wavelen_fit_from_disc,
        construct_wavelen_map,
    )
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
        spectrum = np.mean(
            medianflat[
                middle_row - pixel_cut_extension : middle_row + pixel_cut_extension + 1,
                :,
            ],
            axis=0,
        )
        spectral_array = np.arange(x_size)

        # flip the spectrum to have it in the right order if needed
        if not detector_params["dispersion"]["wavelength_grows_with_pixel"]:
            spectrum = spectrum[::-1]

    else:

        middle_row = (x_size // 2) + middle_offset
        spectrum = np.mean(
            medianflat[
                :,
                middle_row - pixel_cut_extension : middle_row + pixel_cut_extension + 1,
            ],
            axis=1,
        )
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
        wavelength,
        spectrum,
        x_label="Wavelength (Å)",
        y_label="Counts (ADU)",
        label="Flat-field lamp spectrum",
        title="Use sliders to crop out any noisy parts on detector edges.\n"
        'Press "Q" or close the window when done.',
    )

    # Get the final selected range
    min_wavelength = min_wave
    max_wavelength = max_wave
    valid_indices = (wavelength >= min_wavelength) & (wavelength <= max_wavelength)
    wavelength_cut = wavelength[valid_indices]
    spectrum_cut = spectrum[valid_indices]

    # setup the B-spline fit
    num_interior_knots = flat_params["knots_spectral_bspline"]
    degree = flat_params["degree_spectral_bspline"]

    # check that the number of knots is reasonable
    if num_interior_knots > len(wavelength_cut) // 2:
        logger.warning(
            "The number of interior knots is larger than half the number of data points."
        )
        logger.warning("This may lead to overfitting.")
        logger.warning(
            "Consider reducing the number of knots in the configuration file."
        )

    if num_interior_knots >= len(wavelength_cut):
        logger.error(
            "The number of interior knots is larger than the number of data points."
        )
        logger.error("This will lead to overfitting.")
        logger.error("Please reduce the number of knots in the configuration file.")
        exit()

    # Create the knots array
    t = np.concatenate(
        (
            np.repeat(wavelength_cut[0], degree + 1),  # k+1 knots at the beginning
            np.linspace(
                wavelength_cut[1], wavelength_cut[-2], num_interior_knots
            ),  # interior knots
            np.repeat(wavelength_cut[-1], degree + 1),  # k+1 knots at the end
        )
    )

    # this part does the actual fitting
    spl = make_lsq_spline(wavelength_cut, spectrum_cut, t=t, k=degree)
    bspline = BSpline(spl.t, spl.c, spl.k)

    residuals = spectrum_cut - bspline(wavelength_cut)
    RMS = np.sqrt(np.mean(residuals**2))

    show_1d_fit_QA(
        wavelength_cut,
        spectrum_cut,
        x_fit_values=wavelength_cut,
        y_fit_values=bspline(wavelength_cut),
        residuals=residuals,
        x_label="Wavelength (Å)",
        y_label="Counts (ADU)",
        legend_label="Extracted flat-field lamp spectrum",
        title=f"Spectral response B-spline fit with {num_interior_knots} interior knots, degree {degree} (this is set in the configuration file).\n"
        "You should aim for very little to no large-scale structure in the residuals, "
        "with the lowest amount of knots possible.",
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
    logger.info("Rotating spectral response model...")
    spectral_response_model = flip_and_rotate(
        spectral_response_model, transpose, flip, inverse=True
    )
    logger.info("Rotating the bad pixel mask...")
    bpm = flip_and_rotate(bpm, transpose, flip, inverse=True)

    return spectral_response_model, bpm, RMS


def estimate_spacial_response(medianflat):

    from pylongslit.parser import detector_params, flat_params, developer_params
    from pylongslit.utils import interactively_crop_spec
    from pylongslit.logger import logger
    from pylongslit.stats import safe_mean

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

    # these are the spacial coordinates (x values)
    spacial_slice = np.arange(len(test_slice))

    min_spat, max_spat = interactively_crop_spec(
        spacial_slice,
        test_slice,
        x_label="Spacial pixel",
        y_label="Counts (ADU)",
        label=f"Flat-field frame cut at spectral pixel {middle_pixel}",
        title="Use sliders to crop out any noisy parts on detector edges.\n"
        'Press "Q" or close the window when done.',
    )

    # 5 rows will be plotted for QA
    fig, ax = plt.subplots(5, 2, figsize=(10, 6))

    indices_to_plot = np.linspace(
        5, x_size if spectral_axis == 0 else y_size, 5, endpoint=False, dtype=int
    )

    plot_num = 0  # a bit hacked solution for plotting

    # these will be the actual models for the spacial response
    spacial_model = np.ones((y_size, x_size))
    error = np.zeros_like(medianflat)

    # x values - used when fitting with cropped edges
    spacial_array_cropped = np.arange(min_spat, max_spat)

    # loading the user - defined parameters
    num_interior_knots = flat_params["knots_spacial_bspline"]
    degree = flat_params["degree_spacial_bspline"]

    # check that the number of knots is reasonable
    if num_interior_knots > len(spacial_array_cropped) // 2:
        logger.warning(
            "The number of interior knots is larger than half the number of data points."
        )
        logger.warning("This may lead to overfitting.")
        logger.warning(
            "Consider reducing the number of knots in the configuration file."
        )

    if num_interior_knots >= len(spacial_array_cropped):
        logger.error(
            "The number of interior knots is larger than the number of data points."
        )
        logger.error("This will lead to overfitting.")
        logger.error("Please reduce the number of knots in the configuration file.")
        exit()

    # used for logging how many fits were successful
    num_fits = x_size if spectral_axis == 0 else y_size
    bad_spectral_indices = np.array([])

    # Used for rejecting bad fits
    R2_thresh = flat_params["R2_spacial_bspline"]

    # this keeps tracks of bad fits
    mask_int = np.ones_like(spacial_model, dtype=bool)

    for spectral_pixel in range(num_fits):

        spacial_slice = (
            medianflat[:, spectral_pixel].copy()
            if spacial_axis == 1
            else medianflat[spectral_pixel, :].copy()
        )

        # crop the spacial slice for user defined region
        spacial_slice_cropped = spacial_slice[min_spat:max_spat]

        # remove outliers
        mean = np.nanmean(spacial_slice_cropped)
        two_std = 2 * np.nanstd(spacial_slice_cropped)
        mask = (
            np.isnan(spacial_slice_cropped)
            | np.isinf(spacial_slice_cropped)
            | (spacial_slice_cropped > (mean + two_std))
            | (spacial_slice_cropped < (mean - two_std))
        )

        spacial_slice_masked = spacial_slice_cropped[~mask]
        spacial_array_masked = spacial_array_cropped[~mask]

        # Create the knots array
        t = np.concatenate(
            (
                np.repeat(
                    spacial_array_masked[0], degree + 1
                ),  # k+1 knots at the beginning
                np.linspace(
                    spacial_array_masked[1],
                    spacial_array_masked[-2],
                    num_interior_knots,
                ),  # interior knots
                np.repeat(spacial_array_masked[-1], degree + 1),  # k+1 knots at the end
            )
        )

        # do the fit
        spl = make_lsq_spline(spacial_array_masked, spacial_slice_masked, t=t, k=degree)
        bspline = BSpline(spl.t, spl.c, spl.k)

        # we trim the edges by fit degree + 1 as these are usually diverging
        min_spacial_masked = np.min(spacial_array_masked) + degree + 1
        max_spacial_masked = np.max(spacial_array_masked) - (degree + 1)

        good_spectral_interval = np.arange(min_spacial_masked, max_spacial_masked)

        good_model = bspline(good_spectral_interval)

        R2 = r2_score(spacial_slice[min_spacial_masked:max_spacial_masked], good_model)

        residuals_column = spacial_slice_masked - bspline(spacial_array_masked)

        RMS = np.sqrt(np.nanmean(residuals_column**2))

        # take a sample of 10 points at both ends, and fit a line.
        # This is used for extrapolating the edges, as these are often
        # diverging in a B-spline fit.

        sample_start = good_model[:10]
        spacial_start = good_spectral_interval[:10]
        fit_start = np.polyfit(spacial_start, sample_start, 1)

        sample_end = good_model[-10:]
        spacial_end = good_spectral_interval[-10:]
        fit_end = np.polyfit(spacial_end, sample_end, 1)

        # evaluate the fit at the column and calculate residuals
        if spacial_axis == 1:
            # bad fits
            if R2 < R2_thresh:
                bad_spectral_indices = np.append(bad_spectral_indices, spectral_pixel)
            # good fits - keep the data
            else:
                spacial_model[good_spectral_interval, spectral_pixel] = good_model
                spacial_model[:min_spacial_masked, spectral_pixel] = np.polyval(
                    fit_start, np.arange(min_spacial_masked)
                )
                spacial_model[max_spacial_masked:, spectral_pixel] = np.polyval(
                    fit_end, np.arange(max_spacial_masked, y_size)
                )

                if developer_params["debug_plots"]:
                    plt.plot(spacial_model[:, spectral_pixel])
                    plt.title(
                        f"Estimated slit illumination at spectral pixel: {spectral_pixel}"
                    )
                    plt.show()

                error[:, spectral_pixel] = RMS

        else:
            # bad fits
            if R2 < R2_thresh:
                bad_spectral_indices = np.append(bad_spectral_indices, spectral_pixel)
            # good fits - keep the data
            else:
                spacial_model[spectral_pixel, good_spectral_interval] = good_model
                spacial_model[spectral_pixel, :min_spacial_masked] = np.polyval(
                    fit_start, np.arange(min_spacial_masked)
                )
                spacial_model[spectral_pixel, max_spacial_masked:] = np.polyval(
                    fit_end, np.arange(max_spacial_masked, x_size)
                )

                if developer_params["debug_plots"]:
                    plt.plot(spacial_model[spectral_pixel, :])
                    plt.title(
                        f"Estimated slit illumination at spectral pixel: {spectral_pixel}"
                    )
                    plt.show()

                error[spectral_pixel, :] = RMS

        if spectral_pixel in indices_to_plot:
            if plot_num <= 4:
                ax[plot_num, 0].plot(
                    spacial_array_masked,
                    spacial_slice_masked,
                    ".",
                    label=f"Data at spectral pixel: {spectral_pixel}",
                )
                ax[plot_num, 0].plot(
                    spacial_array_masked,
                    bspline(spacial_array_masked),
                    label="Fit with R2: {:.3f}".format(R2),
                    c="red" if R2 < R2_thresh else "black",
                )
                error[spectral_pixel, :] = RMS

        if spectral_pixel in indices_to_plot:
            if plot_num <= 4:
                ax[plot_num, 0].plot(
                    spacial_array_cropped[mask],
                    spacial_slice_cropped[mask],
                    "o",
                    color="red",
                    label="Masked outliers",
                    markersize=2,  # Make datapoints smaller
                )

                ax[plot_num, 0].legend(fontsize=8)  # Make legend font smaller

                ax[plot_num, 1].plot(
                    spacial_array_masked,
                    residuals_column,
                    "o",
                    color="black",
                    label=f"Residuals at spectral pixel: {spectral_pixel}",
                    markersize=2,  # Make datapoints smaller
                )
                ax[plot_num, 1].axhline(0, color="red", linestyle="--", linewidth=0.5)  # Thinner line

                ax[plot_num, 1].legend(fontsize=8)  # Make legend font smaller

                plot_num += 1

    fig.suptitle(
        f"Slit illumination B-spline fits at different spectral pixels. Rejection set in the configuration file is R2 < {R2_thresh}. \n"
        f"Number of interior knots: {num_interior_knots}, fit degree {degree} (this is set in the configuration file).\n"
        "You should aim for very little to no large-scale structure in the residuals, with the lowest amount of knots possible.",
        fontsize=12,
    )
    fig.text(0.5, 0.04, "Spacial pixel", ha="center", fontsize=12)
    fig.text(
        0.04,
        0.5,
        "Normalized Counts (ADU)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    plt.show()

    # throw a warning if more than a fith of the fits fails
    if len(bad_spectral_indices) > num_fits * 20:
        logger.warning(
            f"{(len(bad_spectral_indices)/num_fits) * 100} %% has been not fitted succesfully."
        )
        logger.warning(
            "This might be expected if you detector has a lot of not-illuminated pixels - but the majority of illuminated pixel should be fitted succesfully."
        )
        logger.warning(
            "Pay attention to the upcoming quality assesment plots, and revise the spacial flat parameters in the configuration file if needed."
        )
    else:
        logger.info(
            f"Number of fits with R2 < {R2_thresh}: {len(bad_spectral_indices)} out of total {num_fits} fits. A handfull is expected."
        )

    logger.info(f"Bad spectral indices: {bad_spectral_indices.astype(int)}")
    logger.info(f"These will be interpolated.")

    if spacial_axis == 1:
        mask_int[:, bad_spectral_indices.astype(int)] = False
    else:
        mask_int[bad_spectral_indices.astype(int), :] = False

    plt.close("all")
    plt.figure(figsize=(10, 6))
    plt.imshow(~mask_int, cmap="grey")
    plt.title(
        "The marked lines (value 1) have not beed fitted succesfully, and will be interpolated.\n"
        "A handfull is expected, but if a substantial amount of fits are bad,\n revise the spacial flat parameters in the configuration file."
    )
    plt.colorbar()
    plt.show()

    spacial_rows = y_size if spectral_axis == 0 else x_size

    spectral_array = np.arange(y_size) if spectral_axis == 1 else np.arange(x_size)

    for spacial_row in range(spacial_rows):
        mask_cut = (
            mask_int[:, spacial_row] if spectral_axis == 1 else mask_int[spacial_row, :]
        )
        if sum(mask_cut) < len(spectral_array) * 0.5:
            if developer_params["verbose_print"]:
                print(f" Skipping spacial row {spacial_row} due to too many bad values")
            continue

        else:
            model_cut = (
                spacial_model[:, spacial_row]
                if spectral_axis == 1
                else spacial_model[spacial_row, :]
            )

            if developer_params["debug_plots"]:
                plt.plot(
                    spectral_array[mask_cut],
                    model_cut[mask_cut],
                    label=f"Good values at spacial pixel: {spacial_row}",
                )
                plt.plot(
                    spectral_array[~mask_cut],
                    model_cut[~mask_cut],
                    "o",
                    color="red",
                    label=f"Bad values at spacial pixel: {spacial_row}",
                )
                plt.legend()
                plt.show()

            # interpolate the bad values using a simple 1d nearest neibhour interpolation
            interpolated_values = griddata(
                spectral_array[mask_cut],
                model_cut[mask_cut],
                spectral_array[~mask_cut],
                method="nearest",
            )

            if spectral_axis == 1:
                spacial_model[~mask_cut, spacial_row] = interpolated_values
            else:
                spacial_model[spacial_row, ~mask_cut] = interpolated_values

            if developer_params["debug_plots"]:
                plt.plot(
                    spectral_array[mask_cut],
                    model_cut[mask_cut],
                    label=f"Good values at spacial pixel: {spacial_row}",
                )
                plt.plot(
                    spectral_array[~mask_cut],
                    model_cut[~mask_cut],
                    "o",
                    color="red",
                    label=f"Bad values at spacial pixel: {spacial_row}",
                )
                plt.plot(
                    spectral_array[~mask_cut],
                    interpolated_values,
                    "x",
                    color="green",
                    label=f"Interpolated values at spacial pixel: {spacial_row}",
                )
                plt.legend()
                plt.show()

    error[~mask_int] = safe_mean(error[mask_int])

    if developer_params["debug_plots"]:
        plt.imshow(error)
        plt.title("Error map for the spacial response model")
        plt.show()

    return spacial_model, error


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
    from pylongslit.parser import (
        detector_params,
        flat_params,
        data_params,
        developer_params,
    )
    from pylongslit.utils import FileList, check_dimensions, open_fits, PyLongslit_frame
    from pylongslit.overscan import estimate_frame_overscan_bias
    from pylongslit.stats import bootstrap_median_errors_framestack, safe_mean
    from pylongslit.dark import check_dark_directory, estimate_dark

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

    # check if dark frames are provided, skip dark subtraction if not

    if check_dark_directory(flat_params["flat_dir"]):
        logger.info("Dark frames found. Estimating dark current for flats...")
        DARKframe = estimate_dark(flat_params["flat_dir"], "flats")
        logger.info("Dark current estimated.")
    else:
        logger.info(
            "No dark frames found. Dark current subtraction will be skipped for flats."
        )
        DARKframe = None

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

        # Subtract the dark if provided
        if DARKframe is not None:
            data = data - DARKframe.data
            logger.info("Subtracted the dark.")

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
        logger.warning(
            "Please consider taking more flat frames or activating error bootstrapping in the config file."
        )

    if not flat_params["bootstrap_errors"]:
        medianflat_error = (
            1.2533 * np.std(bigflat, axis=0) / np.sqrt(file_list.num_files)
        )

    else:
        medianflat_error = bootstrap_median_errors_framestack(bigflat)

    if developer_params["debug_plots"]:
        plt.imshow(medianflat_error, cmap="gray")
        plt.title("Errors in the median flat")
        plt.show()

    print("\n-------------------------\n")
    logger.info("Estimating the spectral response and normalizing...")

    spectral_response_model, bpm, RMS_spectral = estimate_spectral_response(medianflat)

    if developer_params["debug_plots"]:
        plt.imshow(spectral_response_model, cmap="gray")
        plt.title("2D spectral response model")
        plt.show()

        plt.imshow(bpm, cmap="gray")
        plt.title("Bad pixel mask for spectral response model")
        plt.show()

    spectral_normalized = medianflat / spectral_response_model

    medianflat_error = spectral_normalized * np.sqrt(
        ((medianflat_error / medianflat)) ** 2
        + ((RMS_spectral / spectral_response_model) ** 2)
    )

    logger.info("Spectral response normalization done.")

    # correct any outliers (these are ususally the non-illuminated parts of the flat)
    medianflat_error[bpm] = safe_mean(medianflat_error[~bpm])
    medianflat_error[np.isinf(medianflat_error)] = safe_mean(medianflat_error)
    medianflat_error[np.isnan(medianflat_error)] = safe_mean(medianflat_error)
    medianflat_error[np.isnan(spectral_normalized)] = safe_mean(medianflat_error)
    medianflat_error[np.isinf(spectral_normalized)] = safe_mean(medianflat_error)
    medianflat_error[spectral_normalized < 0.5] = safe_mean(medianflat_error)
    medianflat_error[spectral_normalized > 1.5] = safe_mean(medianflat_error)

    spectral_normalized[spectral_normalized < 0.5] = 1
    spectral_normalized[spectral_normalized > 1.5] = 1
    spectral_normalized[np.isnan(spectral_normalized)] = 1
    spectral_normalized[np.isinf(spectral_normalized)] = 1

    if developer_params["debug_plots"]:
        plt.imshow(medianflat_error, cmap="gray")
        plt.title("Just after spectral normalization")
        plt.show()

    # if requested, do the spacial response normalization
    if not flat_params["skip_spacial"]:
        print("\n-------------------------\n")
        logger.info("Estimating the spacial response and normalizing...")
        spacial_response_model, RMS_spacial = estimate_spacial_response(
            spectral_normalized
        )

        if developer_params["debug_plots"]:
            plt.imshow(spacial_response_model, cmap="gray")
            plt.title("2D slit illumination model")
            plt.show()

        master_flat = spectral_normalized / spacial_response_model
        medianflat_error = master_flat * np.sqrt(
            ((medianflat_error / medianflat)) ** 2
            + ((RMS_spacial / spacial_response_model) ** 2)
        )

        logger.info("Spacial response normalization done.")

    else:
        logger.info(
            "Skipping spacial normalization as requested in the configuration file..."
        )
        master_flat = spectral_normalized

    print("\n-------------------------\n")

    logger.info("Rejecting outliers and plotting the results...")

    # the below code sets outliers to 1 - these are usually the non-illuminated parts of the flat

    if use_overscan:
        # if overscan was used, set the overscan region to one to avoid
        # explosive values in the final flat

        # Extract the overscan region
        overscan_x_start = detector_params["overscan"]["overscan_x_start"]
        overscan_x_end = detector_params["overscan"]["overscan_x_end"]
        overscan_y_start = detector_params["overscan"]["overscan_y_start"]
        overscan_y_end = detector_params["overscan"]["overscan_y_end"]

        master_flat[
            overscan_y_start:overscan_y_end, overscan_x_start:overscan_x_end
        ] = 1.0

        medianflat_error[
            overscan_y_start:overscan_y_end, overscan_x_start:overscan_x_end
        ] = safe_mean(medianflat_error)

    # TODO: this code is used several times, make a method
    # purge the flat of any outliers once more
    medianflat_error[np.isinf(medianflat_error)] = safe_mean(medianflat_error)
    medianflat_error[np.isnan(medianflat_error)] = safe_mean(medianflat_error)
    medianflat_error[master_flat < 0.5] = safe_mean(medianflat_error)
    medianflat_error[master_flat > 1.5] = safe_mean(medianflat_error)
    medianflat_error[np.isnan(master_flat)] = safe_mean(medianflat_error)
    medianflat_error[np.isinf(master_flat)] = safe_mean(medianflat_error)

    master_flat[master_flat < 0.5] = 1.0
    master_flat[master_flat > 1.5] = 1.0

    master_flat[np.isnan(master_flat)] = 1.0
    master_flat[np.isinf(master_flat)] = 1.0

    if developer_params["debug_plots"]:
        plt.imshow(medianflat_error, cmap="gray")
        plt.title("Error after all normalizations")
        plt.show()

    fig, ax = plt.subplots(
        5 if not flat_params["skip_spacial"] else 3, 2, figsize=(10, 6)
    )

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
        fontsize=12,
        y=1,
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

    master_flat_frame = PyLongslit_frame(
        master_flat, medianflat_error, hdr, "master_flat"
    )

    master_flat_frame.show_frame()
    master_flat_frame.write_to_disc()


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit flatfield procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_flats()


if __name__ == "__main__":
    main()
