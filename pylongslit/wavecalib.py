"""
PyLongslit module for wavelength calibration.
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.modeling.models import Chebyshev2D, Const1D
from astropy.modeling.fitting import LevMarLSQFitter
from numpy.polynomial.chebyshev import chebfit, chebval
import pickle
import os
from astropy.modeling import Fittable1DModel, Parameter
import numpy as np
from sklearn.metrics import r2_score
from itertools import chain
import warnings
import astropy.modeling.fitting
import argparse


class GeneralizedNormal1D(Fittable1DModel):
    """
    This is a generalized Gaussian distribution model for
    fitting the lines in the arc spectrum - it works like a Gaussian
    but has a shape parameter beta that controls the flatness of the peak.
    """

    # the default values here are set to 0 as they are not used ("dummies")
    amplitude = Parameter(default=0)
    mean = Parameter(default=0)
    stddev = Parameter(default=0)
    beta = Parameter(default=0)  # Shape parameter

    @staticmethod
    def evaluate(x, amplitude, mean, stddev, beta):
        return amplitude * np.exp(-((np.abs(x - mean) / stddev) ** beta))


def read_pixtable():
    """
    Read the pixel table from the path specified in the configurations file.

    Returns
    -------
    pixnumber : array
        Pixel numbers.

    wavelength : array
        Wavelengths corresponding to the pixel numbers.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params

    path_to_pixtable = wavecalib_params["center_guess_pixtable"]

    logger.info(
        f"Trying to read center_guess_pixtable table from {path_to_pixtable}..."
    )

    try:
        data = np.loadtxt(path_to_pixtable)
        pixnumber = data[:, 0]
        wavelength = data[:, 1]
        logger.info("Pixtable read successfully.")
    except FileNotFoundError:
        logger.critical(f"File {path_to_pixtable} not found.")
        logger.critical("You have to run the identify routine first.")
        logger.critical(
            "In identify routine, you have to create the center guess pixel table, "
            "and set its' path in the config file."
        )
    return pixnumber, wavelength


def get_master_arc():
    """
    Get the master arc image.

    Returns
    -------
    master_arc : pylongslit.utils.PyLongslit_frame
        Master arc image.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import PyLongslit_frame

    logger.info("Trying to fetch the master arc frame...")

    try:
        master_arc = PyLongslit_frame.read_from_disc("master_arc.fits")
    except FileNotFoundError:
        logger.critical("Master arc not found.")
        logger.critical("You have to run the combine_arcs routine first.")
        exit()

    logger.info("Master arc fetched successfully.")

    return master_arc


def arc_trace_warning(message):
    """
    A helper method for logging warnings during the arc tracing.
    Mostly to avoid code repetition. Appends a pre-designed
    ending to the message. Prints and logs when called.

    Parameters
    ----------
    message : str
        Warning message
    """

    from pylongslit.logger import logger

    logger.warning(message)
    logger.warning(
        "This is expected for some lines, but pay attention "
        "to the upcoming quality assessment plots."
    )


def update_model_parameters(g_model, g_fit):
    """
    Update the arc-line fitting model parameters with the fitted values.

    Helper method for avoiding code repetition.

    Modifies the model in-place.

    Parameters
    ----------
    g_model : `~astropy.modeling.models.GeneralizedNormal1D`
        Generalized normal distribution model.

    g_fit : `~astropy.modeling.models.GeneralizedNormal1D`
        Generalized normal distribution model fitted to the data.
    """
    from pylongslit.parser import wavecalib_params

    g_model.amplitude_0 = g_fit.amplitude_0.value
    g_model.mean_0 = g_fit.mean_0.value
    g_model.stddev_0 = g_fit.stddev_0.value
    g_model.beta_0 = g_fit.beta_0.value
    g_model.amplitude_1 = g_fit.amplitude_1.value

    tolerance_mean = wavecalib_params["TOL_MEAN"]
    tolerance_FWHM = wavecalib_params["TOL_FWHM"]

    g_model.mean_0.bounds = (
        g_model.mean_0.value - tolerance_mean,
        g_model.mean_0.value + tolerance_mean,
    )
    g_model.stddev_0.bounds = (
        g_model.stddev_0.value - tolerance_FWHM * gaussian_fwhm_to_sigma,
        g_model.stddev_0.value + tolerance_FWHM * gaussian_fwhm_to_sigma,
    )


def fit_arc_1d(
    spectral_coords,
    counts,
    fitter,
    g_model,
    R2_threshold=0.99,
    bench_value=None,
    bench_tolerance=1.0,
):
    """
    Method for fitting 1d arc lines.

    Modifies the fitter object in place with the new fit.

    Parameters
    ----------
    spectral_coords : array
        Spectral coordinates.

    counts : array
        Counts for the spectral coordinates.

    fitter : `~astropy.modeling.fitting.LevMarLSQFitter`
        Fitter object

    g_model : `~astropy.modeling.models.GeneralizedNormal1D`
        Generalized gaussian distribution model.

    R2_threshold : float
        Threshold for R2 score. Used to determine if the fit is good.
        Default is 0.99.

    bench_value : float
        Benchmark value for the center. If the center value is too far from
        the benchmark value, the fit is considered bad. Default is None -
        meaning that deviation from the benchmark value is not checked.

    bench_tolerance : float
        Tolerance for the deviaition from the benchmark value. Default is 1.0.

    Returns
    -------
    bool
        True if the fit is good, False otherwise.
    """

    from pylongslit.parser import developer_params

    # Suppress warnings during fitting - we handle these ourselves
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: the below exception is not a robust good fix - refactor
        try:
            g_fit = fitter(g_model, spectral_coords, counts)
        except (TypeError, ValueError, astropy.modeling.fitting.NonFiniteValueError):
            return False

    R2 = r2_score(counts, g_fit(spectral_coords))

    # failed fit
    if R2 < R2_threshold:

        if developer_params["debug_plots"]:

            # debugging - plot all bad fits
            plt.plot(spectral_coords, counts, "x", color="black", label=f"R2: {R2}")
            plt.plot(spectral_coords, g_fit(spectral_coords), color="red")
            plt.legend()
            plt.title(f"Failed at R2: {R2}")
            plt.show()

        if developer_params["verbose_print"]:
            print("fit_1d_fail")

        return False

    # not returned yet - good fit

    if developer_params["verbose_print"]:
        print("Good fit")
    if developer_params["debug_plots"]:
        plt.plot(spectral_coords, counts, "x", color="black", label=f"R2: {R2}")
        plt.plot(spectral_coords, g_fit(spectral_coords), color="green")
        plt.show()

    if bench_value is not None:
        if np.abs(g_fit.mean_0.value - bench_value) > bench_tolerance:
            if developer_params["verbose_print"]:
                print("BIG CHANGE")
                print(g_fit.mean_0.value, bench_value)
            return False

    update_model_parameters(g_model, g_fit)

    return True


def trace_line_tilt(master_arc, center_row, fitter, g_model, FWHM_guess):
    """
    Driver method for tracing the tilt of a single line.

    Parameters
    ----------
    master_arc : pylongslit.utils.PyLongslit_frame
        Master arc image.

    center_row : int
        Spacial index from where to start the fitting. User can shift it from the center, calling it "center-row" is just a convention.

    fitter : `~astropy.modeling.fitting.LevMarLSQFitter`
        Fitter object.

    g_model : `~astropy.modeling.models.GeneralizedNormal1D`
        Generalized Gaussian distribution model.

    FWHM_guess : float
        Initial guess for the FWHM of the line.

    Returns
    -------
    all_centers : array
        Centers of the lines.

    used_spacial : array
        Spacial coordinates used for the fit of the line.

    keep_mask : array
        Mask for keeping the good fits.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params, developer_params

    # Container for fit parameters.
    # Sometimes used for initial guesses for the next row.
    # For now holds more information than needed, but this is nice to have in
    # further developtment or debugging.
    all_params = {}

    # This amount of pixels is used +/- around the spacial pixel
    # in order to create a mean count array for the line (reduces noise).
    pixel_cut_extension = wavecalib_params["pixel_cut_extension"]

    # sanity check for the extension
    if (
        center_row - pixel_cut_extension < 0
        or center_row + pixel_cut_extension > master_arc.data.shape[0]
    ):
        logger.critical(
            "The pixel cut extension is too large for the current center row."
        )
        logger.critical(
            "Decrease the pixel cut extension or change the center row offset in the configuration file."
        )
        exit()

    # this allows the user to manually crop-out any noise end bits.
    start_pixel = wavecalib_params["arcline_start"]
    end_pixel = wavecalib_params["arcline_end"]

    # Sanity checks for the user defined start and end pixels
    if center_row > end_pixel or center_row < start_pixel:
        logger.critical(
            f"Center row of arc frame at {center_row} is outside the defined start and end pixels {start_pixel, end_pixel}."
        )
        logger.critical(
            "Add an offset to the middle cut or change the arc line limits (this is done in the config file)."
        )
        exit()

    # new variable to deal with python indexing - the end pixel is not included
    cut_width = pixel_cut_extension + 1

    # for book-keeping what index corresponds to what spacial pixel
    index_dict = {}
    num_it = 0
    for i in chain(
        range(center_row, end_pixel, cut_width),
        range(center_row - 1, start_pixel, -cut_width),
    ):
        index_dict[str(i)] = num_it
        num_it += 1

    # return array for the centers of the lines
    keep_mask = np.array([], dtype=bool)

    #  "x" values for the fit
    spectral_coords = np.arange(master_arc.data.shape[1])

    # read the parameter that decides when to abort the trace
    TILT_REJECTION_LINE_FRACTION = wavecalib_params["TILT_REJECT_LINE_FRACTION"]
    bad_fit_counter = 0
    bad_fit_threshold = num_it * TILT_REJECTION_LINE_FRACTION
    TILT_TRACE_R2_TOL = wavecalib_params["TILT_TRACE_R2_TOL"]
    jump_tolerance = wavecalib_params["jump_tolerance"]

    # this is the benchmark value to avoid "jumps" between lines -
    # the fitter checks that the next fit is not too far from the previous one
    last_good_center = g_model.mean_0.value

    # container for what spacial pixels the fits were successful for
    used_spacial = []

    # This is the driver loop - we loop over the spacial pixels from the center row
    # and go in both directions until we reach the edge of the detector.

    for i in chain(
        range(center_row, end_pixel, cut_width),
        range(center_row - 1, start_pixel, -cut_width),
    ):

        # if we are starting to loop downwards from going upwards, we need to update the
        # initial guesses back to the center row values manually.
        if i == center_row - 1:
            if developer_params["verbose_print"]:
                print("DETECTED DOWNWARDS MOTION")
            g_model.amplitude_0 = all_params[i + 1]["amplitude"]
            g_model.mean_0 = all_params[i + 1]["center"]
            g_model.stddev_0 = all_params[i + 1]["FWHM"] * gaussian_fwhm_to_sigma
            g_model.beta_0 = all_params[i + 1]["beta"]
            g_model.amplitude_1 = all_params[i + 1]["amplitude_1"]

            tolerance_mean = wavecalib_params["TOL_MEAN"]
            tolerance_FWHM = wavecalib_params["TOL_FWHM"]
            g_model.mean_0.bounds = (
                g_model.mean_0.value - tolerance_mean,
                g_model.mean_0.value + tolerance_mean,
            )
            g_model.stddev_0.bounds = (
                g_model.stddev_0.value - tolerance_FWHM * gaussian_fwhm_to_sigma,
                g_model.stddev_0.value + tolerance_FWHM * gaussian_fwhm_to_sigma,
            )

            last_good_center = g_model.mean_0.value

        # clip out the subimage around the line
        start_pixel = int(g_model.mean_0.value - FWHM_guess)
        end_pixel = int(g_model.mean_0.value + FWHM_guess)

        # if the line is too close to the detector edge, skip it
        if start_pixel < 0 or end_pixel > master_arc.data.shape[1]:
            if developer_params["verbose_print"]:
                print("skipping due to edge at: ", i)
                print(start_pixel, end_pixel)
            continue

        # extract the subimage around the line, and average over the spacial pixels
        center_row_spec = np.mean(
            master_arc.data[
                i - pixel_cut_extension : i + pixel_cut_extension + 1,
                start_pixel:end_pixel,
            ],
            axis=0,
        )
        spectral_coords_sub = spectral_coords[start_pixel:end_pixel]

        # the actual fitting
        keep_bool = fit_arc_1d(
            spectral_coords_sub,
            center_row_spec,
            fitter,
            g_model,
            R2_threshold=TILT_TRACE_R2_TOL,
            bench_value=last_good_center,
            bench_tolerance=jump_tolerance,
        )

        # not successful - this will trigger a few attempts to recover
        if not keep_bool:

            # try minimizing the FWHM - might be due to close lines:

            # clip out the subimage around the line - 2 pixels hardcoded for now
            start_pixel += 2
            end_pixel -= 2

            # too narrow cut - fail the fit
            if start_pixel >= end_pixel:
                keep_bool = False

            else:
                # try the fit again
                center_row_spec = np.mean(
                    master_arc.data[
                        i - pixel_cut_extension : i + pixel_cut_extension + 1,
                        start_pixel:end_pixel,
                    ],
                    axis=0,
                )
                spectral_coords_sub = spectral_coords[start_pixel:end_pixel]

                if developer_params["verbose_print"]:
                    print("RECUTTING AND TRYING AGAIN")

                keep_bool = fit_arc_1d(
                    spectral_coords_sub,
                    center_row_spec,
                    fitter,
                    g_model,
                    R2_threshold=TILT_TRACE_R2_TOL,
                    bench_value=last_good_center,
                    bench_tolerance=jump_tolerance,
                )

            # successful fit - pack-up and continue
            if keep_bool:
                if developer_params["verbose_print"]:
                    print("SUCCESSFUL RE-FIT")

                all_params[i] = {
                    "amplitude": g_model.amplitude_0.value,
                    "center": g_model.mean_0.value,
                    "FWHM": g_model.stddev_0.value * gaussian_sigma_to_fwhm,
                    "beta": g_model.beta_0.value,
                    "amplitude_1": g_model.amplitude_1.value,
                }

                keep_mask = np.append(keep_mask, True)

                last_good_center = g_model.mean_0.value

                used_spacial.append(i)

                continue

            # last resort before failing completely
            else:

                # for several bad fits, might be that the center guess is off
                # if enough good fits are present. Try to fit for the line tilt
                # and use the tilt to estimate the center of the line at the current
                # spacial row

                # at least 10 good fits are needed to try this - hardcoded value for now
                if np.sum(keep_mask) > 10:
                    if developer_params["verbose_print"]:
                        print("ENOUGH GOOD FITS TO RETRY")

                    # extract what is needed for the tilt fit
                    all_centers = np.array(
                        [all_params[key]["center"] for key in all_params.keys()]
                    )

                    good_centers = all_centers[keep_mask]

                    # try the fit
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        coeff = chebfit(
                            used_spacial,
                            good_centers,
                            deg=wavecalib_params["ORDER_SPATIAL_TILT"],
                        )

                    if developer_params["debug_plots"]:
                        plt.plot(used_spacial, good_centers, "o")
                        plt.plot(used_spacial, chebval(used_spacial, coeff))
                        plt.show()

                    # check if the fit is good before using it
                    new_center_guess = chebval(i, coeff)

                    R2 = r2_score(good_centers, chebval(used_spacial, coeff))

                    # bad titl fit - abort
                    if R2 < wavecalib_params["SPACIAL_R2_TOL"]:
                        if developer_params["verbose_print"]:
                            print("R2 TOO LOW FOR NEW CENTER GUESS")
                        keep_bool = False

                    # too big change in the center guess - abort
                    elif (
                        abs(new_center_guess - g_model.mean_0.value)
                        > wavecalib_params["TOL_MEAN"]
                    ):
                        if developer_params["verbose_print"]:
                            print("TOO BIG CHANGE IN NEW CENTER GUESS")
                        keep_bool = False

                    # successful tilt fit - try the fit again with the new center guess
                    # and the new cut
                    else:

                        g_model.mean_0 = new_center_guess

                        if developer_params["verbose_print"]:
                            print("NEW CENTER GUESS", new_center_guess)

                        # try to fit again
                        keep_bool = fit_arc_1d(
                            spectral_coords_sub,
                            center_row_spec,
                            fitter,
                            g_model,
                            R2_threshold=TILT_TRACE_R2_TOL,
                            bench_value=last_good_center,
                            bench_tolerance=jump_tolerance,
                        )

                    # successful fit - pack-up and continue
                    if keep_bool:
                        if developer_params["verbose_print"]:
                            print("SUCCESSFUL RETRY IN FITTED CENTER ESTIMATION")
                        all_params[i] = {
                            "amplitude": g_model.amplitude_0.value,
                            "center": g_model.mean_0.value,
                            "FWHM": g_model.stddev_0.value * gaussian_sigma_to_fwhm,
                            "beta": g_model.beta_0.value,
                            "amplitude_1": g_model.amplitude_1.value,
                        }

                        keep_mask = np.append(keep_mask, True)

                        last_good_center = g_model.mean_0.value

                        used_spacial.append(i)

                        continue

                    # Failed fit
                    else:

                        all_params[i] = {
                            "amplitude": g_model.amplitude_0.value,
                            "center": g_model.mean_0.value,
                            "FWHM": g_model.stddev_0.value * gaussian_sigma_to_fwhm,
                            "beta": g_model.beta_0.value,
                            "amplitude_1": g_model.amplitude_1.value,
                        }

                        bad_fit_counter += 1

                        keep_mask = np.append(keep_mask, False)

                        # too many bad center fits - abort the line all-together
                        if bad_fit_counter > bad_fit_threshold:

                            arc_trace_warning(
                                f"Line trace rejected.\n"
                                "The selection parameters for this are set in the configuration file, and currently are: \n"
                                f"Bad center fit fraction at when to abandon line tracing (bad fits / all fits) : {TILT_REJECTION_LINE_FRACTION}, corresponding to {int(bad_fit_threshold)} fits out of {num_it}. \n"
                                f"R2 tolerance for individual fits of the center: >{TILT_TRACE_R2_TOL}. \n"
                                f"Maximum allowed shift in center since last successful fit: {jump_tolerance}. \n"
                                f"Maximum allowed deviation in FWHM from the initial guess: {wavecalib_params['TOL_FWHM']}."
                            )

                            return None, None, None

                        else:
                            continue

        # this is the successful fit at first try - pack-up and continue

        all_params[i] = {
            "amplitude": g_model.amplitude_0.value,
            "center": g_model.mean_0.value,
            "FWHM": g_model.stddev_0.value * gaussian_sigma_to_fwhm,
            "beta": g_model.beta_0.value,
            "amplitude_1": g_model.amplitude_1.value,
        }

        keep_mask = np.append(keep_mask, True)

        last_good_center = g_model.mean_0.value

        used_spacial.append(i)

    # fitting done - extract results and return

    # extract the centers of the lines
    all_centers = np.array([all_params[key]["center"] for key in all_params.keys()])

    logger.info(
        f"Line traced successfully at pixel {all_params[center_row]['center']}."
    )

    return all_centers, np.array(used_spacial), keep_mask


def show_cyclic_QA_plot(
    fig, ax, title_text=None, x_label=None, y_label=None, show=True
):
    """
    A helper method to avoid code repetition when rendering/showing the QA plots when
    looping throug different arc-tracing routines.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        Figure object.

    ax : array
        Axes objects.

    title_text : str
        Title of the plot.

    x_label : str
        X-axis label. Can be None.

    y_label : str
        Y-axis label. Can be None.

    show : bool
        If True, the plot is shown. Default is True.
    """

    # this removes scientific notation for the y-axis
    # to make more space for the subplots
    for ax_row in ax:
        for ax_col in ax_row:
            formatter = ticker.ScalarFormatter(
                useOffset=False, useMathText=False, useLocale=False
            )
            formatter.set_scientific(False)
            ax_col.yaxis.set_major_formatter(formatter)

    title_text = title_text

    for ax_row in ax:
        for ax_col in ax_row:
            # check if the axis is empty
            if not ax_col.lines:
                ax_col.axis("off")
            else:
                ax_col.legend(fontsize=8)
                ax_col.grid(True)

    fig.suptitle(title_text, fontsize=10, va="top", ha="center")

    # Add a single x-label and y-label for the entire figure
    if x_label is not None:
        fig.text(0.5, 0.04, x_label, ha="center", va="center", fontsize=10)

    if y_label is not None:
        fig.text(
            0.04,
            0.5,
            y_label,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=10,
        )
    if show:
        plt.show(block=True)


def trace_tilts(lines, master_arc, plot_height=6, plot_width=2, figsize=(10, 6)):
    """
    The main driver method for tracing the tilts of the lines in the arc spectrum.

    Firstly, estimates the arc line center position on the spacial axis
    for every line that was re-identified when fitting the 1d wavelength solution.

    Then, calculates the offset of the line from the center row, and fits a polynomial
    to the offsets to trace the tilt of the line.

    Parameters
    ----------
    lines : dict
        Lines dictionary. These are the lines that have been identified
        when fitting the 1d wavelength solution.

    master_arc : pylongslit.utils.PyLongslit_frame
        Master arc image.

    plot_height : int
        Number of rows in the cyclic QA plot. Default is 6.

    plot_width : int
        Number of columns in the cyclic QA plot. Default is 2.

    figsize : tuple
        Figure size. Default is (10, 6).

    Returns
    -------
    good_lines : dict
        Good lines. The shape of the dictionary is:
        {wavelength: (offsets, good_centers, good_spacial_coords, coeff_offset, center_pixel)}
        , where:
        - offsets : Tilt offset from the center row.
        - good_centers : Positions of the succesfully fitted line centers
        - good_spacial_coords : Spacial coordinates corresponding to the good_centers.
        - coeff_offset : Coefficients for the offset polynomial fit.
        - center_pixel : The lines' spectral pixel at the center row.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params, developer_params
    from pylongslit.utils import hist_normalize

    logger.info("Tracing the tilts of the identified lines in the arc spectrum...")

    plt.close("all")

    # get detector shape parameters
    N_ROWS = master_arc.data.shape[0]

    # the offset is needed if the middle of the detector is not a good place
    # to take a sample. This is a user defined parameter in the configuration file.
    center_row_offset = wavecalib_params["offset_middle_cut"]

    center_row = (N_ROWS // 2) + center_row_offset
    # sanity check for the center row
    if center_row < 0 or center_row > N_ROWS:
        logger.critical("The center row is off-setted outside the detector limits.")
        logger.critical(
            "Change the offset_middle_cut parameter in the configuration file."
        )
        exit()

    # this will serve as the "x" axis for the fits
    spectral_coords = np.arange(master_arc.data.shape[1])

    # get the tolerance for the RMS of the tilt line fit
    R2_TOL = wavecalib_params["SPACIAL_R2_TOL"]

    # containers for the good lines and RMS values
    good_lines = {}
    RMS_all = {}

    # general fitting params from the configuration file
    spacial_fit_order = wavecalib_params["ORDER_SPATIAL_TILT"]
    tolerance_mean = wavecalib_params["TOL_MEAN"]
    tolerance_FWHM = wavecalib_params["TOL_FWHM"]
    # this is the amount of pixels used to average over the spacial pixels around the line
    # to reduce noise. This is a user defined parameter in the configuration file.
    # So for every spacial row, we take the average of the pixels in the range
    # [center_row - pixel_cut_extension, center_row + pixel_cut_extension]
    pixel_cut_extension = wavecalib_params["pixel_cut_extension"]

    # counters for QA and debugging
    good_traces = 0
    good_fits = 0

    if developer_params["debug_plots"]:

        # plot the line centers

        all_peak_pix = [lines[key]["peak_pix"] for key in lines.keys()]

        plt.imshow(hist_normalize(master_arc.data), cmap="gray")
        plt.plot(all_peak_pix, np.full_like(all_peak_pix, center_row), "x", color="red")
        plt.show()

    fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)

    # for book-keeping the cyclic plots
    j = 0

    title_text = (
        f"Line Tilt Tracing Results. Green: accepted, Red: rejected. Polynomial order: {spacial_fit_order}.\n"
        f"Acceptance Criteria: R2 of the spacial polynomial fit > {R2_TOL}.\n"
        f"Residuals should be randomly distributed around 0, but some small-scale structure is hard to avoid.\n"
        f"These parameters can be changed in the configuration file."
    )

    for key in lines.keys():
        # extract the initial fit for the arc line from the reidentify procedure
        pixel = lines[key]["peak_pix"]
        wavelength = lines[key]["wavelength"]
        FWHM_guess = lines[key]["FWHM"]
        beta = lines[key]["beta"]
        amplitude = lines[key]["amplitude"]
        constant = lines[key]["constant"]

        # algorithm: for every line - fit the spectral centers of the line through
        # the spacial axis. If enough good fits are present, calculate the offsets
        # of the line compared to the center row, and fit a polynomial to the offsets
        # to trace the tilt of the line. Evaluate the offset fit with R2 and
        # user defined tolerance.

        print("\n-------------------------------------")

        logger.info(f"Tracing line-tilt at pixel {pixel}...")

        start_pixel = pixel - FWHM_guess
        end_pixel = pixel + FWHM_guess

        # cut out the line
        sub_image = master_arc.data[:, int(start_pixel) : int(end_pixel)]
        center_row_spec = np.mean(
            sub_image[
                center_row
                - pixel_cut_extension : (center_row + pixel_cut_extension)
                + 1,
                :,
            ],
            axis=0,
        )
        spectral_coords_sub = spectral_coords[int(start_pixel) : int(end_pixel)]

        # initialize the model and the fitter - these will be used to loop over the spacial pixels
        A_init = amplitude
        mean_init = pixel
        stddev_init = FWHM_guess * gaussian_fwhm_to_sigma
        beta_init = beta

        g_init = GeneralizedNormal1D(
            amplitude=A_init,
            mean=mean_init,
            stddev=stddev_init,
            beta=beta_init,
            bounds={
                # amplitude should be nonzero, and somewhere around max value
                "amplitude": (1, 1.1 * A_init),
                "mean": (pixel - tolerance_mean, pixel + tolerance_mean),
                "stddev": (
                    (FWHM_guess - tolerance_FWHM) * gaussian_fwhm_to_sigma,
                    (FWHM_guess + tolerance_FWHM) * gaussian_fwhm_to_sigma,
                ),
                # beta > 2 flattens peak, beta > 20 is almost a step function
                "beta": (2, 20),
            },
        )

        # a constant model to add to the Gaussian - sometimes needed
        # if a continuum is present in the line spectrum
        const = Const1D(amplitude=constant)
        g_model = g_init + const
        fitter = LevMarLSQFitter()

        if developer_params["debug_plots"]:
            # plot the initial guess

            plt.plot(spectral_coords_sub, center_row_spec, "x", color="black")
            # plot the initial fitting model
            spec_fine = np.linspace(
                spectral_coords_sub[0], spectral_coords_sub[-1], 1000
            )
            plt.plot(spec_fine, g_model(spec_fine), color="red", label="initial guess")
            plt.legend()
            plt.show()
            # Perform 2-pass fit to get a good estimate of the line center

        # this traces the line through the spacial axis - returns
        # None values if the trace was unsuccessful
        centers, spacial_coords_used, mask = trace_line_tilt(
            master_arc, center_row, fitter, g_model, FWHM_guess
        )

        if developer_params["debug_plots"]:
            # debugging the line trace - plot the centers
            if centers is not None and spacial_coords_used is not None:
                plt.close("all")
                plt.plot(spacial_coords_used, centers[mask], "o")
                plt.show()

        # tracing was unsuccessful - abort and move on
        if centers is None:
            continue

        good_traces += 1

        # keep the good traces. We keep the mask as it might be useful
        # in further development or debugging
        good_centers = centers[mask]
        good_spacial_coords = spacial_coords_used

        # calculate the offsets of the line from the center row

        center_pixel = pixel

        offsets = center_pixel - good_centers

        # now do the polynomial fit to the offsets and check the quality

        coeff_offset = chebfit(good_spacial_coords, offsets, deg=spacial_fit_order)

        R2_offsets = r2_score(offsets, chebval(good_spacial_coords, coeff_offset))

        # bad fit - show in QA but do not save
        if R2_offsets < R2_TOL:
            arc_trace_warning(
                f"Good line trace, but the RMS={R2_offsets} of the spacial polynomial is higher than the tolerance {R2_TOL}. "
                f"Current user defined fitting order: {spacial_fit_order}."
            )

            plot_color = "red"

        # good fit - save the results
        else:

            plot_color = "green"
            good_fits += 1

            good_lines[wavelength] = (
                offsets,
                good_centers,
                good_spacial_coords,
                coeff_offset,
                center_pixel,
            )

            RMS_all[wavelength] = R2_offsets

        # plot for QA

        ax[j][0].plot(
            good_spacial_coords,
            offsets,
            "x",
            color="black",
            label=f"Offsets at spectral pixel {np.round(center_pixel,2)}",
        )
        spat_fine = np.linspace(0, N_ROWS, 1000)
        ax[j][0].plot(
            spat_fine,
            chebval(spat_fine, coeff_offset),
            color=plot_color,
            label=f"Fit R2: {np.round(R2_offsets,2)}",
        )

        residuals = offsets - chebval(good_spacial_coords, coeff_offset)

        ax[j][1].plot(
            good_spacial_coords, residuals, "x", color="black", label="Residuals"
        )
        ax[j][1].axhline(0, color="red", linestyle="--")
        ax[j][1].legend(fontsize=8)

        if (
            # this condition checks if the plot has been filled up.
            # Plots if yes, and sets a new plot up
            j
            == (plot_height - 1)
        ):
            show_cyclic_QA_plot(
                fig,
                ax,
                title_text,
                "Spacial Pixels",
                f"Line tilt compared to spacial pixel {center_row} (in pixels)",
            )

            if j != len(lines.keys()) - 1:
                fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)

            j = 0

            continue

        j += 1

    # show the last plot
    show_cyclic_QA_plot(
        fig,
        ax,
        title_text,
        "Spacial Pixels",
        f"Line tilt compared to spacial pixel {center_row} (in pixels)",
    )

    # show the traced lines
    plt.figure(figsize=figsize)
    plt.imshow(hist_normalize(master_arc.data), cmap="gray")

    for key in good_lines.keys():
        plt.plot([good_lines[key][1]], [good_lines[key][2]], "x", color="red")

    plt.title(
        "Traced lines in the arc spectrum.\n"
        "Use the zoom tool to inspect the overall trace quality, and revise the tilt procedure if needed."
    )
    plt.show()

    print("\n-------------------------------------")
    logger.info("Line tilt tracing done.")
    logger.info(f"Number of good traces: {good_traces} out of {len(lines)}.")
    logger.info(f"Number of good fits: {good_fits} out of {good_traces}.")
    print("\n-------------------------------------")

    return good_lines


def reidentify(
    pixnumber, wavelength, master_arc, plot_height=4, plot_width=3, figsize=(10, 6)
):
    """
    Takes the manually identified lines from the arc identification procedure
    and re-identifies them using a normalized Gaussian fit.


    Parameters
    ----------
    pixnumber : array
        Spectral pixel numbers of the hand-identified lines.

    wavelength : array
        Wavelengths corresponding to the pixel numbers.

    master_arc : pylongslit.utils.PyLongslit_frame
        Master arc image.

    plot_height : int
        Number of rows in the cyclic QA plot. Default is 4.

    plot_width : int
        Number of columns in the cyclic QA plot. Default is 3.

    figsize : tuple
        Figure size of the cyclic QA plot. Default is (10, 6).

    Returns
    -------
    line_REID : dict
        Reidentified lines. The format is:
        {wavelength: (center, amplitude, FWHM, beta, constant)}, where:
        - center : Center of the line (spectral pixel).
        - amplitude : Amplitude of the line fit.
        - FWHM : Full width at half maximum of the line fit.
        - beta : Beta parameter of the line fit. (desides the broadness of peak)
        - constant : Constant term value of the line fit.
    """
    from pylongslit.parser import wavecalib_params
    from pylongslit.utils import show_1d_fit_QA

    # tolerance for center shift from hand-identified center to the Gaussian fit
    tol_mean = wavecalib_params["TOL_MEAN"]
    # rough guess of FWHM of lines in pixels
    FWHM = wavecalib_params["FWHM"]
    # tolerance for FWHM deviation from the initial guess to the Gaussian fit
    tol_FWHM = wavecalib_params["TOL_FWHM"]

    # tolerance for the R2 of the fit of the individual lines
    final_r2_tol = wavecalib_params["REIDENTIFY_R2_TOL"]

    # create a container for hand-identified lines
    ID_init = Table(dict(peak=pixnumber, wavelength=wavelength))

    # container for re-identified lines. This will be the final product
    line_REID = {}

    # offset is needed if the mddle of the detector is not a good place
    # to take a sample
    middle_row_offset = wavecalib_params["offset_middle_cut"]

    # we extract the arc spectrum from the middle of master arc (+ offset)
    # +/- extension pixels around the center row, and take the mean.
    # This is done to reduce noise in the line spectrum.
    # The extension is a user defined parameter in the configuration file.
    pixel_cut_extension = wavecalib_params["pixel_cut_extension"]

    middle_row = (master_arc.data.shape[0] // 2) + middle_row_offset
    # limits of the slice
    lower_cut, upper_cut = (
        middle_row - pixel_cut_extension,
        middle_row + pixel_cut_extension,
    )

    # take the mean
    spec_1d = np.mean(master_arc.data[lower_cut : upper_cut + 1, :], axis=0)

    spectral_coords = np.arange(len(spec_1d))

    # this offset value allows to make cyclic subplots, as we use the index
    # together with integer division and module to cycle through subplots
    j_offset = 0

    # this figure is filled with the cyclic QA plots
    fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)
    title_text = (
        f"Reidentification Results. Green: accepted, Red: rejected. \n"
        f"Acceptance Criteria:  R2 > {final_r2_tol}.\n"
        f"Initial guess from FWHM is {FWHM}, allowed deviation from line center guess is {tol_mean}, allowed deviation from FWHM guess is {tol_FWHM}. \n"
        f"All of these parameters can be changed in the configuration file."
    )

    # the main loop - we loop over the hand-identified lines and re-identify them
    for j, peak_pix_init in enumerate(ID_init["peak"]):

        # starts guess limits of the peak
        search_min = int(np.around(peak_pix_init - FWHM))
        search_max = int(np.around(peak_pix_init + FWHM))

        # crop the spectrum around the guess
        cropped_spec = spec_1d[search_min:search_max]
        cropped_spectral_coords = spectral_coords[search_min:search_max]

        # remove any nans and infs from the cropped spectrum
        nan_inf_mask = np.isnan(cropped_spec) | np.isinf(cropped_spec)
        cropped_spectral_coords = cropped_spectral_coords[~nan_inf_mask]
        cropped_spec = cropped_spec[~nan_inf_mask]

        # if empty array - keep looping (this is a rare case but a cheap check)
        if len(cropped_spec) == 0:
            continue

        # TODO - this code for fitter initialization is used in several places - refactor
        # initialize the fitter
        A_init = np.max(cropped_spec)
        mean_init = peak_pix_init
        stddev_init = FWHM * gaussian_fwhm_to_sigma
        beta_init = 2  # beta = 2 means simple Gaussian form

        g_init = GeneralizedNormal1D(
            amplitude=A_init,
            mean=mean_init,
            stddev=stddev_init,
            beta=beta_init,
            bounds={
                # amplitude should be nonzero, and somewhere around max value
                "amplitude": (1, 1.1 * A_init),
                "mean": (peak_pix_init - tol_mean, peak_pix_init + tol_mean),
                "stddev": (
                    (FWHM - tol_FWHM) * gaussian_fwhm_to_sigma,
                    (FWHM + tol_FWHM) * gaussian_fwhm_to_sigma,
                ),
                # beta > 2 flattens peak, beta > 20 is almost a step function
                "beta": (2, 20),
            },
        )

        # a constant model to add to the Gaussian - sometimes needed
        # if a continuum is present in the line spectrum
        const = Const1D(amplitude=0)
        g_model = g_init + const
        fitter = LevMarLSQFitter()

        # bool to check if the line is bad
        bad_line = False
        # this will be used as rejection criteria
        R2 = None

        # Perform 2-pass fit to get a good estimate of the line center
        for i in range(2):

            # perform the fit to recenter and get good start values
            # Suppress warnings during fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g_fit = fitter(g_model, cropped_spectral_coords, cropped_spec)

            R2 = r2_score(cropped_spec, g_fit(cropped_spectral_coords))

            # if R2 of the initial fit is below 0.5, there is practically no
            # chance to recover the line. Abort the trace and move on.
            if (i == 0) and (R2 < 0.5):
                arc_trace_warning(
                    "Line could not be identified with a Gaussian fit. "
                    f"Current FWHM guess is {FWHM}."
                )

                bad_line = True
                break

            # extract the fitted peak position and FWHM:
            fit_center = g_fit.mean_0.value
            FWHM_local = g_fit.stddev_0.value * gaussian_sigma_to_fwhm
            beta = g_fit.beta_0.value

            # get a better estimate of the line position
            start_pixel = int(fit_center - FWHM_local)
            end_pixel = int(fit_center + FWHM_local)

            # case of less points than the number of fitting params
            if (end_pixel - start_pixel) < len(g_model.param_names):
                arc_trace_warning(
                    "Too little points - can't perform the fit.\n"
                    "FWHM is possibly too small or tolerance is too high. Please review the configuration file."
                )

                bad_line = True
                break

            # crop the spectrum around the refined guess
            cropped_spec = spec_1d[start_pixel:end_pixel]
            cropped_spectral_coords = spectral_coords[start_pixel:end_pixel]

            # remove any nans and infs from the cropped spectrum
            nan_inf_mask = np.isnan(cropped_spec) | np.isinf(cropped_spec)
            cropped_spectral_coords = cropped_spectral_coords[~nan_inf_mask]
            cropped_spec = cropped_spec[~nan_inf_mask]

            # update the fitter parameters
            update_model_parameters(g_model, g_fit)

        # set the QA plot color depending on whether the fit was successful
        plot_color = "green"

        if bad_line or R2 < final_r2_tol:
            plot_color = "red"

        # cyclic QA plot - the subplot index is calculated from the main index
        # and the plot width and height.

        subplot_index = (j - j_offset) // plot_width, (j - j_offset) % plot_width

        ax[subplot_index].plot(
            cropped_spectral_coords, cropped_spec, "x", color="black"
        )

        spec_fine = np.linspace(
            cropped_spectral_coords[0], cropped_spectral_coords[-1], 1000
        )
        try:
            ax[subplot_index].plot(
                spec_fine,
                g_fit(spec_fine),
                color=plot_color,
                label="fit R2: {:.2f}, FWHM: {:.2f}".format(R2, FWHM_local),
            )
            ax[subplot_index].axvline(
                mean_init,
                color="black",
                linestyle="--",
                label=f"Initial center guess {mean_init:.2f}",
            )
            ax[subplot_index].plot(
                fit_center,
                g_fit(fit_center),
                "o",
                color=plot_color,
                label=f"Fitted center: {fit_center:.2f}",
            )
        # this is needed to catch the case when the fit is bad and the values
        # don't get defined
        except UnboundLocalError:
            ax[subplot_index].plot(spec_fine, g_fit(spec_fine), color=plot_color)

        if (
            # this condition checks if the plot has been filled up
            # plots if so, and adjust the offset so a new
            # plot can be created and filled up
            (j - j_offset) // plot_width == plot_height - 1
            and (j - j_offset) % plot_width == plot_width - 1
        ):
            show_cyclic_QA_plot(fig, ax, title_text, "Spectral Pixels", "Counts (ADU)")
            j_offset += plot_width * plot_height
            # prepare a new plot, if not the last iteration
            if (j + 1) < len(ID_init):
                fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)

        # if the line is good, save the results
        if not bad_line and R2 > final_r2_tol:
            line_REID[str(j)] = {
                "peak_pix": fit_center,
                "wavelength": wavelength[j],
                "FWHM": FWHM_local,
                "beta": beta,
                "amplitude": g_fit.amplitude_0.value,
                "constant": g_fit.amplitude_1.value,
            }

    # plot the last plot if not filled up
    show_cyclic_QA_plot(fig, ax, title_text, "Spectral Pixels", "Counts (ADU)")

    return line_REID


def fit_1d_solution(line_REID, master_arc):
    """
    Fits a 1d wavelength solution to the re-identified lines.

    Parameters
    ----------
    line_REID : dict
        Reidentified lines. The format is:
        {wavelength: (center, amplitude, FWHM, beta, constant)}, where:
        - center : Center of the line (spectral pixel).
        - amplitude : Amplitude of the line fit.
        - FWHM : Full width at half maximum of the line fit.
        - beta : Beta parameter of the line fit. (desides the broadness of peak)
        - constant : Constant term value of the line fit.

    master_arc : pylongslit.utils.PyLongslit_frame
        Master arc image.

    Returns
    -------
    fit : chebyshev.chebfit
        1d wavelength solution fit.
    """

    from pylongslit.parser import wavecalib_params
    from pylongslit.utils import show_1d_fit_QA

    # order of the 1d wavelength solution fit
    fit_order = wavecalib_params["ORDER_WAVELEN_1D"]

    # lines re-identified, now fit a 1d solution to the re-identified lines
    all_pixels = [line_REID[key]["peak_pix"] for key in line_REID.keys()]
    all_wavelengths = [line_REID[key]["wavelength"] for key in line_REID.keys()]

    fit = chebfit(all_pixels, all_wavelengths, deg=fit_order)
    residuals = all_wavelengths - chebval(all_pixels, fit)

    # get the spectral length of the detector
    N_SPECTRAL = master_arc.data.shape[1]

    pixel_linspace = np.linspace(0, N_SPECTRAL, 1000)
    wavelength_linspace = chebval(pixel_linspace, fit)

    show_1d_fit_QA(
        all_pixels,
        all_wavelengths,
        x_fit_values=pixel_linspace,
        y_fit_values=wavelength_linspace,
        residuals=residuals,
        x_label="Pixels in spectral direction",
        y_label="Wavelength (Å)",
        legend_label="Reidentified lines",
        title=f"1D fit to reidentified lines. Polynomial order: {fit_order}.\n"
        f"Ensure the residuals are randomly distributed around 0, otherwise consider changing the order of the fit in the configuration file.",
    )

    plt.close("all")

    return fit


def fit_2d_tilts(good_lines: dict, figsize=(10, 6)):
    """
    Fits the 2d tilt to the traced tilts.

    Parameters
    ----------
    good_lines : dict
        Good lines. The shape of the dictionary is:
        {wavelength: (offsets, good_centers, good_spacial_coords, coeff_offset, center_pixel)}
        , where:
        - offsets : Tilt offset from the center row.
        - good_centers : Positions of the succesfully fitted line centers
        - good_spacial_coords : Spacial coordinates corresponding to the good_centers.
        - coeff_offset : Coefficients for the offset polynomial fit.
        - center_pixel : The lines' spectral pixel at the center row.

    figsize : tuple
        Figure size. Default is (10, 6).

    Returns
    -------
    fit2D : `~astropy.modeling.models.Chebyshev2D`
        2d tilt fit.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params, developer_params

    logger.info("Preparing to fit a 2d polynomial tilt through whole delector...")

    # extract the polynomial order parameter for the fit in spectral direction
    ORDER_SPECTRAL = wavecalib_params["ORDER_SPECTRAL_TILT"]
    # extract the polynomial order parameter for the fit in spatial direction
    ORDER_SPATIAL = wavecalib_params["ORDER_SPATIAL_TILT"]

    logger.info(
        f"Fitting a 2d tilt solution of order {ORDER_SPECTRAL} in spectral direction and "
        f"order {ORDER_SPATIAL} in spatial direction to reidentified lines..."
    )

    # the following lines extract the good lines from the dictionary
    # and put them in the format that the fitter expects
    offset_values = np.array([])  # z axis
    spectral_pixels = np.array([])  # x axis
    spacial_pixels = np.array([])  # y axis

    for key in good_lines.keys():
        offset_values = np.append(offset_values, good_lines[key][0])
        spectral_pixels = np.append(spectral_pixels, good_lines[key][1])
        spacial_pixels = np.append(spacial_pixels, good_lines[key][2])

    spectral_pixels = spectral_pixels.flatten()
    spacial_pixels = spacial_pixels.flatten()

    if developer_params["debug_plots"]:
        # plot a scatter plot for QA
        plt.scatter(spectral_pixels, spacial_pixels, c=offset_values)
        plt.xlabel("Spectral Pixels")
        plt.ylabel("Spacial Pixels")
        plt.title("Scatter plot of the reidentified lines")
        plt.colorbar(label="Offsets (Pixel)")
        plt.show()

    # set up the fitting model and perform the fit

    coeff_init = Chebyshev2D(
        x_degree=ORDER_SPECTRAL,
        y_degree=ORDER_SPATIAL,
    )

    fitter = LevMarLSQFitter(calc_uncertainties=True)

    fit2D = fitter(coeff_init, spectral_pixels, spacial_pixels, offset_values)

    residuals = offset_values - fit2D(spectral_pixels, spacial_pixels)
    RMS = np.sqrt(np.mean(residuals**2))

    if developer_params["debug_plots"]:
        # 3d residuals for debugging - too abstract for the user
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            spacial_pixels, spectral_pixels, residuals, c=residuals, cmap="viridis"
        )
        ax.set_ylabel("Spectral Pixels")
        ax.set_xlabel("Spacial Pixels")
        ax.set_zlabel("Residuals (Å)")
        ax.set_title("Residuals of the 2D Fit")
        fig.colorbar(sc, label="Residuals (Å)")
        plt.show()

    # plot QA

    fig, axs = plt.subplots(2, 1, figsize=figsize)

    fig.suptitle(
        "2D tilt fit residuals. Ensure the residuals are randomly distributed around 0, otherwise consider changing the order of the fit in the configuration file.\n"
        f"Small scale residual structure (≈ 0.01 pixels) is hard to avoid, and should be okay.\n"
        f"Current 2d polynomial fit order: spectral: {ORDER_SPECTRAL}, spatial: {ORDER_SPATIAL}.\n"
        f"RMS of the residuals: {RMS}.",
        fontsize = 10
    )

    axs[0].plot(spacial_pixels, residuals, "x")
    axs[0].set_xlabel("Spacial Pixels")
    axs[0].set_ylabel("Tilt in pixels.")
    axs[0].axhline(0, color="red", linestyle="--")

    axs[1].plot(spectral_pixels, residuals, "x")
    axs[1].set_xlabel("Spectral Pixels")
    axs[1].set_ylabel("Tilt in pixels.")
    axs[1].axhline(0, color="red", linestyle="--")

    plt.show()

    return fit2D


def construct_detector_map(fit2D_REID):
    """
    Evaluate a 2D fit at every pixel of the detector to get.

    Parameters
    ----------
    fit2D_REID : `~astropy.modeling.models.Chebyshev2D`
        2D fit model.

    Returns
    -------
    map : 2D array
        A detector array with the fit evaluated at every pixel.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import detector_params

    logger.info("Constructing the detector map...")

    N_SPACIAL = (
        detector_params["xsize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["ysize"]
    )

    N_SPECTRAL = (
        detector_params["ysize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["xsize"]
    )

    # create the spacial and spectral lines, use to construct a meshgrid
    spectral_line = np.linspace(0, N_SPECTRAL, N_SPECTRAL)
    spacial_line = np.linspace(0, N_SPACIAL, N_SPACIAL)

    X, Y = np.meshgrid(spectral_line, spacial_line)
    map = fit2D_REID(X, Y)

    return map


def plot_tilt_2D_QA(fit2D_REID, good_lines: dict, figsize=(10, 6)):
    """
    Plots a QA of the 2D tilt fit. This is done by plotting the individual
    line fits and the 2D fit at the same pixels as the individual fits.

    Parameters
    ----------
    fit2D_REID : `~astropy.modeling.models.Chebyshev2D`
        2D fit model.

    good_lines : dict
        Good lines. The shape of the dictionary is:
        {wavelength: (offsets, good_centers, good_spacial_coords, coeff_offset, center_pixel)}
        , where:
        - offsets : Tilt offset from the center row.
        - good_centers : Positions of the succesfully fitted line centers
        - good_spacial_coords : Spacial coordinates corresponding to the good_centers.
        - coeff_offset : Coefficients for the offset polynomial fit.
        - center_pixel : The lines' spectral pixel at the center row.

    figsize : tuple
        Figure size. Default is (10, 6).
    """
    # extract the needed parameters from the good lines dictionary
    centers_1d = [good_lines[key][1] for key in good_lines.keys()]
    spatials_1d = [good_lines[key][2] for key in good_lines.keys()]
    coeffs_1d = [good_lines[key][3] for key in good_lines.keys()]
    fitted_centers = [good_lines[key][4] for key in good_lines.keys()]

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # plot the individual line fits
    for i, coeffs in enumerate(coeffs_1d):
        central_spec = fitted_centers[i]
        axs[0].plot(
            spatials_1d[i],
            chebval(spatials_1d[i], coeffs),
            ".",
            label=f"Central spec: {int(central_spec)}",
        )

    axs[0].set_title("Individually traced line tilts.")
    axs[0].set_xlabel("Spatial Pixels")
    axs[0].set_ylabel("Line titls in pixels.")
    axs[0].legend(fontsize = 8)

    for i, spectral_array in enumerate(centers_1d):
        # evaluate the 2d fit at the same pixels as the individual fits
        offsets_2d = fit2D_REID(spectral_array, spatials_1d[i])
        axs[1].plot(spatials_1d[i], offsets_2d, ".")

    axs[1].set_title(
        "2D tilt fit evaluated at the same pixels as the individual traced lines."
    )
    axs[1].set_xlabel("Spatial Pixels")
    axs[1].legend()

    fig.suptitle(
        "2D tilt fit QA. Left: individually traced line tilts. Right: 2D tilt fit evaluated at the same pixels as the individual traced lines.\n"
        "The right side should resemble the left side, preferibly with irregularities smoothened out.\n"
        "If not, review the whole tilt fitting procedure."
    )
    plt.show()


def construct_wavelen_map(wavelen_fit, tilt_fit):
    """
    Constructs a wavelength map from the 2D tilt and the 1D wavelength solution.

    Parameters
    ----------
    wavelen_fit : chebyshev.chebfit
        1d wavelength solution fit.

    tilt_fit : `~astropy.modeling.models.Chebyshev2D`
        2D tilt fit model.

    Returns
    -------
    map : 2D array
        A detector array with the wavelength solution evaluated at every pixel.
    """

    from pylongslit.parser import detector_params
    from pylongslit.utils import wavelength_sol

    # TODO: this meshgrid build is repeated, refactor
    N_SPACIAL = (
        detector_params["xsize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["ysize"]
    )

    N_SPECTRAL = (
        detector_params["ysize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["xsize"]
    )

    spectral_line = np.linspace(0, N_SPECTRAL, N_SPECTRAL)
    spacial_line = np.linspace(0, N_SPACIAL, N_SPACIAL)

    X, Y = np.meshgrid(spectral_line, spacial_line)

    map = wavelength_sol(X, Y, wavelen_fit, tilt_fit)

    return map


def plot_tiltmap(tilt_map, figsize=(10, 6)):
    """
    Plots the tilt map.

    Parameters
    ----------
    tilt_map : 2D array
        Tilt map (detector array with the tilt evaluated at every pixel).

    figsize : tuple
        Figure size. Default is (10, 6).
    """

    plt.figure(figsize=figsize)
    plt.imshow(tilt_map, origin="lower")
    plt.colorbar(label="Spacial offset from center pixel (in pixels)")
    plt.title(
        "Detector tilt map.\n"
        "Inspect the map for any irregularities - it should be a smooth continuum.\n"
    )
    plt.xlabel("Pixels in spectral direction")
    plt.ylabel("Pixels in spatial direction")
    plt.show()


def plot_wavemap(wavelength_map, figsize=(10, 6)):
    """
    Plots the wavelength map.

    Parameters
    ----------
    wavelength_map : 2D array
        Wavelength map (detector array with the wavelength solution evaluated at every pixel).

    figsize : tuple
        Figure size. Default is (10, 6).
    """

    plt.figure(figsize=figsize)
    plt.imshow(wavelength_map, origin="lower")
    plt.colorbar(label="Wavelength (Å)")
    plt.title(
        "Wavelength map (wavelengths mapped to every pixel of the detector)\n"
        "Inspect the map for any irregularities - it should be a smooth continuum.\n"
        "Also, check if the wavelengths are as expected for your instrument."
    )
    plt.xlabel("Pixels in spectral direction")
    plt.ylabel("Pixels in spatial direction")
    plt.show()


def plot_wavelengthcalib_QA(good_lines: dict, wave_fit, tilt_fit):
    """
    A wrapper method to plot all the final QA plots for the wavelength calibration.

    Parameters
    ----------
    good_lines : dict
        Good lines. The shape of the dictionary is:
        {wavelength: (offsets, good_centers, good_spacial_coords, coeff_offset, center_pixel)}
        , where:
        - offsets : Tilt offset from the center row.
        - good_centers : Positions of the succesfully fitted line centers
        - good_spacial_coords : Spacial coordinates corresponding to the good_centers.
        - coeff_offset : Coefficients for the offset polynomial fit.
        - center_pixel : The lines' spectral pixel at the center row.

    wave_fit : chebyshev.chebfit
        1d wavelength solution fit.

    tilt_fit : `~astropy.modeling.models.Chebyshev2D`
        2D tilt fit model.
    """

    plot_tilt_2D_QA(tilt_fit, good_lines)

    tilt_map = construct_detector_map(tilt_fit)

    plot_tiltmap(tilt_map)

    wave_map = construct_wavelen_map(wave_fit, tilt_fit)

    plot_wavemap(wave_map)


def simulate_wavelengthcalib_variance(
    wave_fit, wave_fit_error, tilt_fit, tilt_fit_error, n_sims=1000
):
    """
    PURELY EXPERIMENTAL - DO NOT USE
    """

    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params
    from pylongslit.parser import detector_params

    logger.info("Simulating wavelength calibration variance...")
    logger.info(f"Number of simulations: {n_sims}")

    N_SPACIAL = (
        detector_params["xsize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["ysize"]
    )

    N_SPECTRAL = (
        detector_params["ysize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["xsize"]
    )

    wavemap_sum = np.zeros(
        (
            N_SPACIAL,
            N_SPECTRAL,
        )
    )

    wavemap_sum_squares = np.zeros(
        (
            N_SPACIAL,
            N_SPECTRAL,
        )
    )

    wavemap_diffs = np.zeros(
        (
            N_SPACIAL,
            N_SPECTRAL,
        )
    )

    first_param_list = []

    wavemap_initial = construct_wavelen_map(wave_fit, tilt_fit)

    for _ in range(n_sims):
        # for _ in tqdm(range(n_sims), desc="Estimating wavelength calibration variance"):
        # simulate the variance in the wavelength map by sampling from the
        # normal distribution of the fit parameters
        random_wave_fit = np.random.normal(wave_fit, wave_fit_error, wave_fit.shape)
        first_param_list.append(random_wave_fit[0])

        random_tilt_params = np.random.normal(
            tilt_fit.parameters, tilt_fit_error, tilt_fit_error.shape
        )

        random_tilt_fit = tilt_fit.copy()
        random_tilt_fit.parameters = random_tilt_params

        wave_map = construct_wavelen_map(random_wave_fit, random_tilt_fit)
        wavemap_sum += wave_map
        wavemap_sum_squares += wave_map**2
        print(np.max(wavemap_sum_squares))

        wavemap_diffs = wavemap_initial - wave_map

    wavemap_mean = wavemap_sum / n_sims
    variance_wavemap = wavemap_sum_squares / n_sims - wavemap_mean**2
    error_wavemap = np.sqrt(variance_wavemap) / np.sqrt(n_sims)

    plt.imshow(wavemap_mean, origin="lower")
    plt.colorbar(label="Mean wavelength map (Å)")
    plt.show()

    plt.imshow(error_wavemap, origin="lower")
    plt.colorbar(label="Error in wavelength map (Å)")
    plt.show()

    wavemap_diffs = wavemap_diffs / n_sims
    plt.imshow(wavemap_diffs, origin="lower")
    plt.colorbar(label="Difference in wavelength map (Å)")
    plt.show()

    plt.hist(first_param_list, bins=int(np.sqrt(n_sims)))
    plt.title(
        f"Distribution of the first parameter of the wavelength fit. Original value: {wave_fit[0]}"
    )
    plt.show()


def calculate_final_fit_rms(wave_fit, tilt_fit, good_lines):
    """
    Calculates the residual rms of the final wavelength detector map.

    Parameters
    ----------
    wave_fit : chebyshev.chebfit
        1d wavelength solution fit.

    tilt_fit : `~astropy.modeling.models.Chebyshev2D`
        2D tilt fit model.

    good_lines : dict
        Good lines. The shape of the dictionary is:
        {wavelength: (offsets, good_centers, good_spacial_coords, coeff_offset, center_pixel)}
        , where:
        - offsets : Tilt offset from the center row.
        - good_centers : Positions of the succesfully fitted line centers
        - good_spacial_coords : Spacial coordinates corresponding to the good_centers.
        - coeff_offset : Coefficients for the offset polynomial fit.
        - center_pixel : The lines' spectral pixel at the center row.

    Returns
    -------
    RMS : array
        2d detector shaped array with the wavelength solution
        RMS at every pixel of the detector.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import wavelength_sol
    from pylongslit.parser import detector_params

    logger.info("Calculating the final wavelength calibration error...")

    residuals = np.array([])

    # for every traced line, evaluate the wavelength solution at the line
    # coordinates and calculate the residuals.
    for key in good_lines.keys():
        good_x = good_lines[key][1]
        good_y = good_lines[key][2]
        wavelength = float(key)

        wavesol_wavelengths = wavelength_sol(good_x, good_y, wave_fit, tilt_fit)

        residuals = np.append(residuals, wavesol_wavelengths - wavelength)

    # calculate the RMS.
    residuals = np.array(residuals).flatten()
    RMS = np.sqrt(np.mean(residuals**2))

    # create a constant error array with the RMS value. The constant array
    # may seem redundant, but it is a good interface for later development.
    N_SPACIAL = (
        detector_params["xsize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["ysize"]
    )

    N_SPECTRAL = (
        detector_params["ysize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["xsize"]
    )

    error_array = np.full((N_SPACIAL, N_SPECTRAL), RMS)

    return error_array


def write_waveimage_to_disc(wavelength_map, master_arc):
    """
    DEPRECATED

    Write the wavelength calibration results (waveimage) to disc.

    Parameters
    ----------
    wavelength_map : 2D array
        Wavelength map.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir
    from pylongslit.utils import write_to_fits

    logger.info("Writing wavelength calibration results to disc...")

    # steal header from master_arc
    header = master_arc.header
    write_to_fits(wavelength_map, header, "wavelength_map.fits", output_dir)

    logger.info("Wavelength calibration results written to disc.")


# TODO: the below 8 methods have a lot of code repetition - refactor


def write_reided_lines_to_disc(lines):
    """
    Backs up the reidentified lines to disc (machine-readable format only).

    Name of the file is reidentified_lines.pkl and it is stored in the output directory
    defined in the configuration file.

    Parameters
    ----------
    lines : dict
        Reidentified lines. The format is:
        {wavelength: (center, amplitude, FWHM, beta, constant)}, where:
        - center : Center of the line (spectral pixel).
        - amplitude : Amplitude of the line fit.
        - FWHM : Full width at half maximum of the line fit.
        - beta : Beta parameter of the line fit. (desides the broadness of peak)
        - constant : Constant term value of the line fit.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Writing reidentified lines to disc...")

    # get current working directory
    cwd = os.getcwd()

    # change to output directory dir
    os.chdir(output_dir)

    # Write lines to disk
    with open("reidentified_lines.pkl", "wb") as file:
        pickle.dump(lines, file)

    # change back to the original working directory
    # this helps if user uses relative paths
    os.chdir(cwd)

    logger.info("Reidentified lines written to disc.")


def write_wavelen_fit_to_disc(fit1d):
    """
    Backs up the 1D fit results to disc (machine-readable format only).

    Name of the file is wavelen_fit.pkl and it is stored in the output directory
    defined in the configuration file.

    Parameters
    ----------
    fit1d : chebyshev.chebfit
        1d wavelength solution fit.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Writing wavelen fit results to disc...")

    # get current working directory
    cwd = os.getcwd()

    # change to output directory dir
    os.chdir(output_dir)

    # Write fit1d to disk
    with open("wavelen_fit.pkl", "wb") as file:
        pickle.dump(fit1d, file)

    # change back to the original working directory
    # this helps if user uses relative paths
    os.chdir(cwd)


    logger.info(
        f"2D tilt fit results written to disc in {output_dir}, filename wavelen_fit.pkl."
    )


def write_tilt_fit_to_disc(fit2D_REID):
    """
    Backs up the 2D fit results to disc (machine-readable format only).

    Name of the file is tilt_fit.pkl and it is stored in the output directory
    defined in the configuration file.

    Parameters
    ----------
    fit2D_REID : `~astropy.modeling.models.Chebyshev2D`
        2D fit model.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Writing tilt fit results to disc...")

    # get current working directory
    cwd = os.getcwd()

    # change to output directory dir
    os.chdir(output_dir)

    # Write fit2D_REID to disk
    with open("tilt_fit.pkl", "wb") as file:
        pickle.dump(fit2D_REID, file)

    # change back to the original working directory
    # this helps if user uses relative paths
    os.chdir(cwd)

    logger.info(
        f"2D tilt fit results written to disc in {output_dir}, filename tilt_fit.pkl."
    )


def write_good_tilt_lines_to_disc(good_lines):
    """
    Backs up the Traced tilt lines to disc (machine-readable format only).

    Name of the file is good_lines.pkl and it is stored in the output directory
    defined in the configuration file.

    Parameters
    ----------
    good_lines : dict
        Traced tilt lines. The shape of the dictionary is:
        {wavelength: (offsets, good_centers, good_spacial_coords, coeff_offset, center_pixel)}
        , where:
        - offsets : Tilt offset from the center row.
        - good_centers : Positions of the succesfully fitted line centers.
        - good_spacial_coords : Spacial coordinates corresponding to the good_centers.
        - coeff_offset : Coefficients for the offset polynomial fit.
        - center_pixel : The lines' spectral pixel at the center.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Writing Traced tilt lines to disc...")

    # get current working directory
    cwd = os.getcwd()

    # change to output directory dir
    os.chdir(output_dir)

    # Write good_lines to disk
    with open("good_lines.pkl", "wb") as file:
        pickle.dump(good_lines, file)

    # change back to the original working directory
    # this helps if user uses relative paths
    os.chdir(cwd)

    logger.info("Traced tilt lines written to disc.")


def get_reided_lines_from_disc():
    """
    Load the reidentified lines from disc.

    Returns
    -------
    lines : dict
        Reidentified lines. The format is:
        {wavelength: (center, amplitude, FWHM, beta, constant)}, where:
        - center : Center of the line (spectral pixel).
        - amplitude : Amplitude of the line fit.
        - FWHM : Full width at half maximum of the line fit.
        - beta : Beta parameter of the line fit. (desides the broadness of peak)
        - constant : Constant term value of the line fit.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Loading reidentified lines from disc...")

    # get current working directory
    cwd = os.getcwd()

    # change to output directory dir
    os.chdir(output_dir)

    # Load lines from disk
    try:
        with open("reidentified_lines.pkl", "rb") as file:
            lines = pickle.load(file)
    except FileNotFoundError:
        logger.error(
            'Reidentified lines not found. Please run the full wavelength calibration routine, with the "reuse" parameters set to false in the configuration file.'
        )
        exit()

    logger.info("Reidentified lines loaded.")

    # change back to the original working directory
    # this helps if user uses relative paths
    os.chdir(cwd)

    return lines


def get_wavelen_fit_from_disc():
    """
    Load the 1D fit results from disc.

    Returns
    -------
    fit1d : `~astropy.modeling.models.Chebyshev1D`
        1D fit model.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Loading 1D wavelength solution from disc...")

    # get current working directory
    cwd = os.getcwd()

    # change to output directory dir
    os.chdir(output_dir)

    # Load the fit from disk
    try:
        with open("wavelen_fit.pkl", "rb") as file:
            fit1d = pickle.load(file)
    except FileNotFoundError:
        logger.error(
            'Wavelength solution not found. Please run the full wavelength calibration routine, with the "reuse" parameters set to false in the configuration file.'
        )
        exit()

    logger.info("Wavelength solution loaded.")

    # change back to the original working directory
    # this helps if user uses relative paths
    os.chdir(cwd)

    return fit1d


def get_tilt_fit_from_disc():
    """
    Load the 2D fit results from disc.

    Returns
    -------
    fit2D_REID : `~astropy.modeling.models.Chebyshev2D`
        2D fit model.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Loading 2D tilt solution from disc...")

    # get current working directory
    cwd = os.getcwd()

    # change to output directory dir
    os.chdir(output_dir)

    # Load the fit from disc from disk
    try:
        with open("tilt_fit.pkl", "rb") as file:
            fit2D_REID = pickle.load(file)
    except FileNotFoundError:
        logger.error(
            'Tilt solution not found. Please run the full wavelength calibration routine, with the "reuse" parameters set to false in the configuration file.'
        )
        exit()

    logger.info("Tilt solution loaded.")

    # change back to the original working directory
    # this helps if user uses relative paths
    os.chdir(cwd)

    return fit2D_REID


def get_good_tilt_lines_from_disc():
    """
    Load the Traced tilt lines from disc.

    Returns
    -------
    good_lines : dict
        Traced tilt lines. The shape of the dictionary is:
        {wavelength: (offsets, good_centers, good_spacial_coords, coeff_offset, center_pixel)}
        , where:
        - offsets : Tilt offset from the center row.
        - good_centers : Positions of the succesfully fitted line centers.
        - good_spacial_coords : Spacial coordinates corresponding to the good_centers.
        - coeff_offset : Coefficients for the offset polynomial fit.
        - center_pixel : The lines' spectral pixel at the center.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Loading Traced tilt lines from disc...")

    # get current working directory
    cwd = os.getcwd()

    # change to output directory dir
    os.chdir(output_dir)

    # Load good_lines from disk
    try:
        with open("good_lines.pkl", "rb") as file:
            good_lines = pickle.load(file)
    except FileNotFoundError:
        logger.error(
            'Traced lines not found. Please run the full wavelength calibration routine, with the "reuse" parameters set to false in the configuration file.'
        )
        exit()

    logger.info("Traced tilt lines loaded.")

    # change back to the original working directory
    # this helps if user uses relative paths
    os.chdir(cwd)

    return good_lines


def construct_wavemap_frame(wave_fit, tilt_fit, error):
    """
    Method to construct a PyLongslit frame with the wavelength map.

    Parameters
    ----------
    wave_fit : chebyshev.chebfit
        1d wavelength solution fit.

    tilt_fit : `~astropy.modeling.models.Chebyshev2D`
        2D tilt fit model.

    error : 2D array
        Error array with the same shape as the detector array.
    """

    from pylongslit.utils import PyLongslit_frame
    from pylongslit.logger import logger

    # we steal the header from the master arc
    try:
        master_arc = PyLongslit_frame.read_from_disc("master_arc.fits")
    except FileNotFoundError:
        logger.error("Master arc not found. Please run the combine arcs routine first.")
        exit()

    hdr = master_arc.header

    wave_map = construct_wavelen_map(wave_fit, tilt_fit)

    wave_map_frame = PyLongslit_frame(wave_map, error, hdr, "wavelength_map")

    wave_map_frame.show_frame()
    wave_map_frame.write_to_disc()


def run_wavecalib():
    """
    Run the wavelength calibration routine.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params

    logger.info("Starting wavelength calibration routine...")

    pixnumber, wavelength = read_pixtable()

    master_arc = get_master_arc()

    print("\n-----------------------------\n")
    if wavecalib_params["reuse_reided_lines"]:
        logger.warning("Reusing reidentified lines from disc.")
        logger.warning("This is set in the configuration file.")
        lines = get_reided_lines_from_disc()
    else:
        logger.info("Reidentifying the lines...")
        lines = reidentify(pixnumber, wavelength, master_arc)
        write_reided_lines_to_disc(lines)
        logger.info("Lines reidentified.")
    print("\n-----------------------------\n")

    print("\n-----------------------------\n")
    if wavecalib_params["reuse_1d_sol"]:
        logger.warning("Reusing 1d wavelength solution from disc.")
        logger.warning("This is set in the configuration file.")
        wave_sol = get_wavelen_fit_from_disc()
    else:
        logger.info("Fitting the 1D wavelength solution...")
        wave_sol = fit_1d_solution(lines, master_arc)
        write_wavelen_fit_to_disc(wave_sol)
        logger.info("1D wavelength solution fit done.")
    print("\n-----------------------------\n")

    print("\n-----------------------------\n")
    if wavecalib_params["reuse_line_traces"]:
        logger.warning("Reusing tilt traces from disc.")
        logger.warning("This is set in the configuration file.")
        good_lines = get_good_tilt_lines_from_disc()
    else:
        logger.info("Starting tilt tracing...")
        good_lines = trace_tilts(lines, master_arc)
        write_good_tilt_lines_to_disc(good_lines)
        logger.info("Tilt tracing done.")
    print("\n-----------------------------\n")

    print("\n-----------------------------\n")
    if wavecalib_params["reuse_2d_tilt_fit"]:
        logger.warning("Reusing 2d tilt fit from disc.")
        logger.warning("This is set in the configuration file.")
        fit_2d_tilt_results = get_tilt_fit_from_disc()
    else:
        logger.info("Fitting the 2D tilt solution...")
        fit_2d_tilt_results = fit_2d_tilts(good_lines)
        write_tilt_fit_to_disc(fit_2d_tilt_results)
        logger.info("2D tilt solution fit done.")
    print("\n-----------------------------\n")


    print("\n-----------------------------\n")

    logger.info("All fits done. Building QA plots...")

    plot_wavelengthcalib_QA(good_lines, wave_sol, fit_2d_tilt_results)

    logger.info("Creating the wavelength map framw...")

    error = calculate_final_fit_rms(wave_sol, fit_2d_tilt_results, good_lines)

    construct_wavemap_frame(wave_sol, fit_2d_tilt_results, error)

    logger.info("Wavelength calibration routine done.")
    print("\n-----------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit wavecalibration procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_wavecalib()


if __name__ == "__main__":
    main()
