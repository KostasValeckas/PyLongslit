from logger import logger
from parser import wavecalib_params, output_dir, detector_params
import numpy as np
from astropy.table import Table
from utils import open_fits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.modeling.models import Chebyshev2D, Const1D, Chebyshev1D
from astropy.modeling.fitting import LevMarLSQFitter
from numpy.polynomial.chebyshev import chebfit, chebval
from utils import write_to_fits
from utils import show_1d_fit_QA
import pickle
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from scipy.optimize import root
from scipy.optimize import minimize
from astropy.modeling import Fittable1DModel, Parameter
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import r2_score
from itertools import chain
import warnings


class GeneralizedNormal1D(Fittable1DModel):
    """
    This is a generalized normal distribution model for
    fitting the lines in the arc spectrum - it works like a Gaussian
    but has a shape parameter beta that controls the flatness of the peak.
    """

    # the default values here are set to 0 as they are not used
    amplitude = Parameter(default=0)
    mean = Parameter(default=0)
    stddev = Parameter(default=0)
    beta = Parameter(default=0)  # Shape parameter

    @staticmethod
    def evaluate(x, amplitude, mean, stddev, beta):
        return amplitude * np.exp(-((np.abs(x - mean) / stddev) ** beta))


def read_pixtable():
    """
    Read the pixel table from the path specified in the config file.

    Returns
    -------
    pixnumber : array
        Pixel numbers.

    wavelength : array
        Wavelengths corresponding to the pixel numbers.
    """

    path_to_pixtable = wavecalib_params["pixtable"]

    logger.info(f"Trying to read pixtable table from {path_to_pixtable}...")

    try:
        data = np.loadtxt(path_to_pixtable)
        pixnumber = data[:, 0]
        wavelength = data[:, 1]
        logger.info("Pixtable read successfully.")
    except FileNotFoundError:
        logger.critical(f"File {path_to_pixtable} not found.")
        logger.critical("You have to run the identify routine first.")
        logger.critical(
            "In identify routine, you have to create the pixel table, "
            "and set its' path in the config file."
        )
    return pixnumber, wavelength


def get_master_arc():
    """
    Get the master arc image.

    Returns
    -------
    master_arc : array
        Master arc image.
    """

    logger.info("Trying to fetch the master arc frame...")

    try:
        master_arc = open_fits(output_dir, "master_arc.fits")
    except FileNotFoundError:
        logger.critical("Master arc not found.")
        logger.critical("You have to run the combine_arcs routine first.")
        exit()

    logger.info("Master arc fetched successfully.")

    return master_arc


def arc_trace_warning(message):
    """
    A helper method for logging warnings during the arc tracing.
    Mostly to avoid code repetition.
    """

    logger.warning(message)
    logger.warning(
        "This is expected for some lines, but pay attention "
        "to the upcoming quality assessment plots."
    )


def fit_arc_1d(spectral_coords, center_row_spec, fitter, g_model, R2_threshold=0.99):
    """
    Method for fitting seperate 1d arc lines.

    Modifies the fitter object in place with the new fit.

    Parameters
    ----------
    spectral_coords : array
        Spectral coordinates.

    center_row_spec : array
        Center row spectrum.

    fitter : `~astropy.modeling.fitting.LevMarLSQFitter`
        Fitter object.

    g_model : `~astropy.modeling.models.GeneralizedNormal1D`
        Generalized normal distribution model.

    R2_threshold : float
        Threshold for R2 score. Used to determine if the fit is good.
        Default is 0.99.

    Returns
    -------
    bool
        True if the fit is good, False otherwise.
    """

    # Suppress warnings during fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_fit = fitter(g_model, spectral_coords, center_row_spec)

    R2 = r2_score(center_row_spec, g_fit(spectral_coords))

    if R2 < R2_threshold:
        return False

    # update the fitter parameters
    g_model.amplitude_0 = g_fit.amplitude_0.value
    g_model.mean_0 = g_fit.mean_0.value
    g_model.stddev_0 = g_fit.stddev_0.value
    g_model.beta_0 = g_fit.beta_0.value
    g_model.amplitude_1 = g_fit.amplitude_1.value

    return True


def trace_line_tilt(
    sub_image,
    spectral_coords,
    N_ROWS,
    center_row,
    fitter,
    g_model,
):
    """
    Traces the tilt of a single line in the arc spectrum.

    Loops over the rows of the sub_image and fits the line in each row.

    Parameters
    ----------
    sub_image : array
        Sub image of the arc spectrum holding the line.

    spectral_coords : array
        Spectral coordinates.

    N_ROWS : int
        Number of rows in the sub image.

    center_row : int
        Center row of the sub image.

    fitter : `~astropy.modeling.fitting.LevMarLSQFitter`
        Fitter object.

    g_model : `~astropy.modeling.models.GeneralizedNormal1D`
        Generalized normal distribution model.

    Returns
    -------
    all_centers_sorted : array
        Sorted centers of the line.

    keep_mask : array
        Mask for keeping the good fits for each row.
    """
    # container for fit parameters.
    # Sometimes used for initial guesses for the next row.
    # For holds more information than needed, but this is nice to have in
    # further developtment or debugging.
    all_params = {}
    # mask for keeping the good fits, will be updated in the loop
    keep_mask = np.ones(N_ROWS, dtype=bool)

    # read the parameter that decides when to abort the trace - i.e.
    # when the fit is bad for more than a certain number of rows
    TILT_REJECTION_LINE_FRACTION = wavecalib_params["TILT_REJECT_LINE_FRACTION"]
    bad_fit_counter = 0
    bad_fit_threshold = N_ROWS * TILT_REJECTION_LINE_FRACTION

    # R2 tolerance for the fit of every row
    TILT_TRACE_R2_TOL = wavecalib_params["TILT_TRACE_R2_TOL"]

    # Loop from center and up and then down
    for i in chain(range(center_row, N_ROWS), range(center_row - 1, -1, -1)):

        # if we are starting to loop downwards, we need to update the initial guesses
        # back to the center row values manually.
        if i == center_row - 1:
            g_model.amplitude_0 = all_params[i + 1]["amplitude"]
            g_model.mean_0 = all_params[i + 1]["center"]
            g_model.stddev_0 = all_params[i + 1]["FWHM"] / gaussian_sigma_to_fwhm
            g_model.beta_0 = all_params[i + 1]["beta"]
            g_model.amplitude_1 = all_params[i + 1]["amplitude_1"]

        center_row_spec = sub_image[i, :]

        keep_bool = fit_arc_1d(
            spectral_coords,
            center_row_spec,
            fitter,
            g_model,
            R2_threshold=TILT_TRACE_R2_TOL,
        )

        all_params[i] = {
            "amplitude": g_model.amplitude_0.value,
            "center": g_model.mean_0.value,
            "FWHM": g_model.stddev_0.value * gaussian_sigma_to_fwhm,
            "beta": g_model.beta_0.value,
            "amplitude_1": g_model.amplitude_1.value,
        }

        if not keep_bool:

            bad_fit_counter += 1
            if bad_fit_counter > bad_fit_threshold:

                arc_trace_warning(
                    f"Too many bad fits with R2 below {TILT_TRACE_R2_TOL} in the tilt trace. "
                    "The selection parameters for this are: "
                    f"TILT_REJECT_LINE_FRACTION = {TILT_REJECTION_LINE_FRACTION}  "
                    f"and TILT_TRACE_R2_TOL. = {TILT_TRACE_R2_TOL}"
                )

                return None, None

            keep_mask[i] = False

    # extract the centers of the lines
    all_centers = np.array([all_params[key]["center"] for key in all_params.keys()])

    # rows and used for sorting the centers only, as they make it easy to
    # track the original order
    rows = np.array([row for row in all_params.keys()])
    sorted_indices = np.argsort(rows)
    all_centers_sorted = all_centers[sorted_indices]

    logger.info(
        f"Line traced successfully at pixel {all_params[center_row]['center']}."
    )

    return all_centers_sorted, keep_mask


def trace_tilts(pixel_array, wavelength_array, master_arc):
    """
    Trace the tilts of the lines in the arc spectrum.
    """
    logger.info("Tracing the tilts of the lines in the arc spectrum...")

    # get detector shape parameters
    N_ROWS = master_arc[0].data.shape[0]
    center_row = N_ROWS // 2
    spacial_coords = np.arange(N_ROWS)

    # get the rough guess of the FWHM of the lines
    FWHM_guess = wavecalib_params["FWHM"]

    # get the tolerance for the RMS of the tilt line fit
    RMS_TOL = wavecalib_params["SPACIAL_RMS_TOL"]

    # containers for the good lines and RMS values
    good_lines = {}
    RMS_all = {}

    spacial_fit_order = wavecalib_params["ORDER_SPATIAL_REID"]

    for pixel, wavelength in zip(pixel_array, wavelength_array):

        # algorithm: crop the spectrum around the line, fit a Gaussian to the line
        # in the center row to refine crop position, then trace the line center every row.
        # Fit a polynomial to the line centers to get the tilt of the line.
        # Use RMS of the polynomial fit as a quality metric.
        # If the RMS is too high, discard the line. Else, redo the fit
        # but now for the offset from the center row.

        print("\n-------------------------------------")

        logger.info(f"Tracing line-tilt at pixel {pixel}...")

        # first guess - this will be refined
        start_pixel = int(pixel - FWHM_guess)
        end_pixel = int(pixel + FWHM_guess)

        sub_image = master_arc[0].data[:, start_pixel:end_pixel]

        spectral_coords = np.arange(start_pixel, end_pixel)

        center_row_spec = sub_image[center_row, :]

        # initialize the fitter - this fitter will be used for all the fits
        A_init = np.max(center_row_spec)
        mean_init = pixel
        stddev_init = FWHM_guess * gaussian_fwhm_to_sigma
        beta_init = 2  # beta = 2 means simple Gaussian form
        TOL_REID_FWHM = wavecalib_params["TOL_REID_FWHM"]

        g_init = GeneralizedNormal1D(
            amplitude=A_init,
            mean=mean_init,
            stddev=stddev_init,
            beta=beta_init,
            # for bounds, accept a deviation of double from the initial guesses
            bounds={
                "amplitude": (1, 2 * A_init),
                "mean": (pixel - 2 * FWHM_guess, pixel + 2 * FWHM_guess),
                "stddev": (
                    TOL_REID_FWHM * gaussian_fwhm_to_sigma / 2,
                    2 * TOL_REID_FWHM * gaussian_fwhm_to_sigma,
                ),
                # beta > 2 flattens peak, beta > 20 is almost a step function
                "beta": (0.1, 20),
            },
        )

        # a constant model to add to the Gaussian - sometimes needed
        # if a continuum is present in the line spectrum
        const = Const1D(amplitude=0)
        g_model = g_init + const
        fitter = LevMarLSQFitter()

        # perform the fit to recenter and get good start values
        # Suppress warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g_fit = fitter(g_model, spectral_coords, center_row_spec)

        R2 = r2_score(center_row_spec, g_fit(spectral_coords))

        # if R2 of the initial fit is below 0.5, there is practically no
        # chance to recover the line. Abort the trace and move on.
        if R2 < 0.5:
            arc_trace_warning(
                "Line could not be identified with a Gaussian fit. "
                f"Current FWHM guess is {FWHM_guess}."
            )
            continue

        # extract the fitted peak position and FWHM:
        fit_center = g_fit.mean_0.value
        FWHM_local = g_fit.stddev_0.value * gaussian_sigma_to_fwhm

        # get a better estimate of the center
        start_pixel = int(fit_center - FWHM_local)
        end_pixel = int(fit_center + FWHM_local)

        sub_image = master_arc[0].data[:, start_pixel:end_pixel]
        center_row_spec = sub_image[center_row, :]
        spectral_coords = np.arange(start_pixel, end_pixel)

        # update the fitter parameters
        g_model.amplitude_0 = g_fit.amplitude_0.value
        g_model.mean_0 = g_fit.mean_0.value
        g_model.stddev_0 = g_fit.stddev_0.value
        g_model.beta_0 = g_fit.beta_0.value
        g_model.amplitude_1 = g_fit.amplitude_1.value

        # Do a second pass fit to get a better estimate of the line center
        # Suppress warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # perform the fit to recenter and get good start values
            g_fit = fitter(g_model, spectral_coords, center_row_spec)

        # extract the fitted peak position and FWHM:
        fit_center = g_fit.mean_0.value
        FWHM_local = g_fit.stddev_0.value * gaussian_sigma_to_fwhm

        # get a better estimate of the center
        start_pixel = int(fit_center - FWHM_local)
        end_pixel = int(fit_center + FWHM_local)

        sub_image = master_arc[0].data[:, start_pixel:end_pixel]
        center_row_spec = sub_image[center_row, :]
        spectral_coords = np.arange(start_pixel, end_pixel)

        # update the fitter parameters
        g_model.amplitude_0 = g_fit.amplitude_0.value
        g_model.mean_0 = g_fit.mean_0.value
        g_model.stddev_0 = g_fit.stddev_0.value
        g_model.beta_0 = g_fit.beta_0.value
        g_model.amplitude_1 = g_fit.amplitude_1.value

        # now fit all rows:

        centers, mask = trace_line_tilt(
            sub_image,
            spectral_coords,
            N_ROWS,
            center_row,
            fitter,
            g_model,
        )

        # tracing was unsuccessful - abort and move on
        if centers is None:
            continue

        # keep the good and the bad fits for QA
        good_centers = centers[mask]
        good_spacial_coords = spacial_coords[mask]

        bad_centers = centers[~mask]
        bad_spacial_coords = spacial_coords[~mask]

        # do a poly fit to the good centers
        coeff = chebfit(good_spacial_coords, good_centers, deg=3)

        RMS = np.sqrt(
            np.mean((good_centers - chebval(good_spacial_coords, coeff)) ** 2)
        )
        # if the RMS is too high, discard the line
        if RMS > RMS_TOL:
            arc_trace_warning(
                f"Good line trace, but the RMS={RMS} of the spacial polynomial is higher than the tolerance {RMS_TOL}. "
                f"Current user defined spacial fitting order: {spacial_fit_order}."
            )
            continue

        else:
            # if the initial poly fit is good, fit the offsets.
            # Check RMS again.
            center_pixel = chebval(center_row, coeff)

            offsets = good_centers - center_pixel

            coeff_offset = chebfit(good_spacial_coords, offsets, deg=3)

            RMS = np.sqrt(
                np.mean((offsets - chebval(good_spacial_coords, coeff_offset)) ** 2)
            )

            # if the RMS is too high, discard the line
            if RMS > RMS_TOL:
                arc_trace_warning(
                    f"Good line trace, but the RMS={RMS} of the spacial polynomial is higher than the tolerance {RMS_TOL}. "
                    f"Current user defined fitting order: {spacial_fit_order}."
                )
                continue

            good_lines[wavelength] = (
                offsets,
                good_centers,
                good_spacial_coords,
                coeff_offset,
                center_pixel,
            )

            RMS_all[wavelength] = RMS

    return good_lines


def show_reidentify_QA_plot(fig, ax, TOL_REID, TOL_REID_FWHM, FWHM):
    """
    Prepare the master reidentify plot and show it.

    I.e. set the title, x-label, y-label and grid.
    Do some formatting for of the axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.

    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object.

    TOL_REID : float
        Tolerance for pixel shift from hand-identified lines to Gaussian fit.

    TOL_REID_FWHM : float
        Tolerance for FWHM of the Gaussian fit.

    FWHM : float
        Rough guess of FWHM of lines in pixels.
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

    title_text = (
        f"Reidentification Results. Green: accepted, Red: rejected. \n"
        f"Acceptance Criteria: \n"
        f"Allowing deviation of {TOL_REID} in pixels from centrum guess. "
        f"\n Allowing FWHM deviation of {TOL_REID_FWHM} pixels from initial FWHM guess of {FWHM} \n"
        "Check that the accepted fits are in fact good fits."
    )

    fig.suptitle(title_text, fontsize=11, va="top", ha="center")

    # Add a single x-label and y-label for the entire figure
    fig.text(
        0.5, 0.04, "Pixels in spectral direction", ha="center", va="center", fontsize=12
    )
    fig.text(
        0.04,
        0.5,
        "Counts (ADU)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    plt.show()


# TODO: see if this can be optimized runtime-wise
def reidentify(pixnumber, wavelength, master_arc):
    """

    Parameters
    ----------
    pixnumber : array
        Pixel numbers.

    wavelength : array
        Wavelengths corresponding to the pixel numbers.

    master_arc : array
        Master arc image.

    Returns
    -------
    line_REID : dict
        Reidentified lines.
    """
    # number of reidentification slices along the spacial direction
    N_REID = wavecalib_params["N_REID"]
    # how many pixels to sum over at every step
    STEP_REID = wavecalib_params["STEP_REID"]
    # tolerance for pixel shift from hand-identified lines to Gaussian fit
    TOL_REID = wavecalib_params["TOL_REID_MEAN"]
    # rough guess of FWHM of lines in pixels
    FWHM = wavecalib_params["FWHM"]
    # tolerance for FWHM of the Gaussian fit
    TOL_REID_FWHM = wavecalib_params["TOL_REID_FWHM"]

    # create a container for hand-identified lines
    ID_init = Table(dict(peak=pixnumber, wavelength=wavelength))

    # container for re-identified lines. This will be the final product
    line_REID = {}

    # these are the coordinates of the middle of the slices used in the reidentification
    # these correspond to the spatial coordinates, functioning as y-axis in 2d-fitting
    spatialcoord = np.arange(0, N_REID * STEP_REID, STEP_REID) + STEP_REID / 2

    # the following parameters are used for QA plots

    # this offset value allows to make cyclic subplots, as we use the index
    # together with integer division and module to cycle through subplots
    j_offset = 0

    plot_height = 4
    plot_width = 3
    figsize = (24, 20)

    fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)

    # we only do one QA plot - around the middle slice
    plot_at_index = N_REID // 2

    # the fits for every slice are very likely very similar, we save the values
    # for the first slice and load them as initial guesses for the rest of the slices
    guess_cache = {}

    # loop over the spatial slices, and reidentify the lines
    for i in tqdm(range(0, N_REID), desc="Reidentifying lines", unit="slice"):
        # limits of the slice
        lower_cut, upper_cut = i * STEP_REID, (i + 1) * STEP_REID
        # sum over the slice
        reidentify_i = np.sum(master_arc[0].data[lower_cut:upper_cut, :], axis=0)

        # container for the reidentified lines for this slice
        peak_gauss_REID = []

        # re-identify every hand-identified line
        for j, peak_pix_init in enumerate(ID_init["peak"]):
            start_time = time.time()
            # limits of the peak
            search_min = int(np.around(peak_pix_init - FWHM * 2))
            search_max = int(np.around(peak_pix_init + FWHM * 2))
            # crop the spectrum around the line
            cropped_spec = reidentify_i[search_min:search_max]
            # dummy x_array around cropped line for later fitting
            x_cropped = np.arange(len(cropped_spec)) + search_min

            # remove any nans and infs from the cropped spectrum
            nan_inf_mask = np.isnan(cropped_spec) | np.isinf(cropped_spec)
            x_cropped = x_cropped[~nan_inf_mask]
            cropped_spec = cropped_spec[~nan_inf_mask]

            # if empty array - keep looping
            if len(cropped_spec) == 0:
                continue
            else:
                # set up some initial guesses for the Gaussian fit
                # amplitude:
                A_init = np.max(cropped_spec)
                # mean (line centrum - initial gues is same as the hand-identified line):
                mean_init = peak_pix_init
                # sigma (width of the line) - convert from user defined FWHM:
                stddev_init = FWHM * gaussian_fwhm_to_sigma

                if i == 1:
                    # build a Generalized Normal fitter with an added constant
                    g_init = GeneralizedNormal1D(
                        amplitude=A_init,
                        mean=mean_init,
                        stddev=stddev_init,
                        beta=2,  # Initial guess for beta
                        bounds={
                            "amplitude": (0, 2 * np.max(cropped_spec)),
                            "stddev": (0, 2 * TOL_REID_FWHM * gaussian_fwhm_to_sigma),
                            # beta > 2 flattens peak, beta > 20 is almost a step function
                            "beta": (2, 20),
                        },
                    )
                else:
                    if str(peak_pix_init) in guess_cache.keys():
                        g_init = GeneralizedNormal1D(
                            amplitude=guess_cache[str(peak_pix_init)][0],
                            mean=guess_cache[str(peak_pix_init)][1],
                            stddev=guess_cache[str(peak_pix_init)][2],
                            beta=guess_cache[str(peak_pix_init)][
                                3
                            ],  # Initial guess for beta
                            bounds={
                                "amplitude": (0, 2 * np.max(cropped_spec)),
                                "stddev": (
                                    0,
                                    2 * TOL_REID_FWHM * gaussian_fwhm_to_sigma,
                                ),
                                # beta > 2 flattens peak, beta > 20 is almost a step function
                                "beta": (2, 20),
                            },
                        )
                    else:
                        # build a Generalized Normal fitter with an added constant
                        g_init = GeneralizedNormal1D(
                            amplitude=A_init,
                            mean=mean_init,
                            stddev=stddev_init,
                            beta=2,  # Initial guess for beta
                            bounds={
                                "amplitude": (0, 2 * np.max(cropped_spec)),
                                "stddev": (
                                    0,
                                    2 * TOL_REID_FWHM * gaussian_fwhm_to_sigma,
                                ),
                                # beta > 2 flattens peak, beta > 20 is almost a step function
                                "beta": (2, 20),
                            },
                        )
                const = Const1D(amplitude=0)
                g_model = g_init + const

                # perform the fit
                fitter = LevMarLSQFitter()
                fit_start_time = time.time()
                g_fit = fitter(g_model, x_cropped, cropped_spec)
                fit_end_time = time.time()

                # extract the fitted peak position and FWHM:
                fit_center = g_fit.mean_0.value
                fitted_FWHM = g_fit.stddev_0.value * gaussian_sigma_to_fwhm

                # this is the cyclic indexing mechanism for the QA plots
                if i == plot_at_index:
                    subplot_index = (j - j_offset) // plot_width, (
                        j - j_offset
                    ) % plot_width

                # check if the fitted peak is within the tolerance of the hand-identified peak
                if (
                    # deviation from center
                    abs(fit_center - peak_pix_init) > TOL_REID
                    # FWHM is too large or too small
                    or abs(fitted_FWHM - FWHM) > TOL_REID_FWHM
                    # amplitude is too small
                    or g_fit.amplitude_0.value < 1
                ):
                    peak_gauss_REID.append(np.nan)
                    if i == plot_at_index:
                        # plot the rejected fits
                        # plot the spectrum
                        ax[subplot_index].plot(
                            x_cropped, cropped_spec, "x", color="black"
                        )
                        # plot the fit
                        x_fine = np.linspace(x_cropped[0], x_cropped[-1], 1000)
                        ax[subplot_index].plot(x_fine, g_fit(x_fine), color="red")
                else:
                    peak_gauss_REID.append(fit_center)
                    if i == 1:
                        guess_cache[str(peak_pix_init)] = (
                            g_fit.amplitude_0.value,
                            g_fit.mean_0.value,
                            g_fit.stddev_0.value,
                            g_fit.beta_0.value,
                        )
                    if i == plot_at_index:
                        # plot the accepted fits
                        # plot the spectrum
                        ax[subplot_index].plot(
                            x_cropped, cropped_spec, "x", color="black"
                        )
                        # plot the fit
                        x_fine = np.linspace(x_cropped[0], x_cropped[-1], 1000)
                        ax[subplot_index].plot(x_fine, g_fit(x_fine), color="green")

                if (
                    # this condition checks if the plot has been filled up
                    # plots if so, and adjust the offset so a new
                    # plot can be created and filled up
                    (j - j_offset) // plot_width == plot_height - 1
                    and (j - j_offset) % plot_width == plot_width - 1
                    and i == plot_at_index
                ):
                    show_reidentify_QA_plot(fig, ax, TOL_REID, TOL_REID_FWHM, FWHM)
                    j_offset += plot_width * plot_height
                    # prepare a new plot
                    fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)
                end_loop_time = time.time()

        # last plot - plot the last plots even if master plot is not filled up
        if i == plot_at_index:
            show_reidentify_QA_plot(fig, ax, TOL_REID, TOL_REID_FWHM, FWHM)

        n_good_reids = np.sum(~np.isnan(peak_gauss_REID))

        if n_good_reids == 0:
            logger.warning(
                f"No lines reidentified in slice {i}. "
                "Check the tolerance values in the config file if this warning continues to appear."
            )
            continue

        # store the reidentified lines in the master container

        # mask for infs and nans
        mask = np.isfinite(peak_gauss_REID)

        # Ensure ID_init['wavelength'] is an array-like object
        wavelengths = np.array(ID_init["wavelength"])
        # Same for peak_gauss_REID
        peak_gauss_REID = np.array(peak_gauss_REID)

        # store the reidentified lines, theiwavelength and y-value
        # y-value is the same for the whole slice
        line_REID[str(i)] = Table(
            dict(
                peak_pix=peak_gauss_REID[mask],
                wavelength=wavelengths[mask],
                # extends the spacial coordinate to pair every spectral coordinate
                spacial=spatialcoord[i] * np.ones(np.sum(mask)),
            )
        )

    return line_REID


def fit_1d_QA(line_REID: dict):
    """
    Fit the reidentified lines through detector middle for QA.

    Parameters
    ----------
    line_REID : dict
        Reidentified lines.
    """

    # extract the polynomial order parameter for the fit
    ORDER_WAVELEN_REID = wavecalib_params["ORDER_WAVELEN_REID"]
    logger.info(
        f"Fitting a 1d wavelength solution of order {ORDER_WAVELEN_REID} to reidentified lines..."
    )

    # find the middle key of line_REID dictionary
    middle_key = len(line_REID) // 2

    # ready the data
    pixels = line_REID[str(middle_key)]["peak_pix"]
    wavelengths = line_REID[str(middle_key)]["wavelength"]

    # fit the data
    coeff = chebfit(pixels, wavelengths, deg=ORDER_WAVELEN_REID)

    last_pixel = (
        detector_params["xsize"]
        if detector_params["dispersion"]["spectral_dir"] == "x"
        else detector_params["ysize"]
    )

    # evaluate fit for the QA
    x_fit_values = np.linspace(0, last_pixel, 1000)
    y_fit_values = chebval(x_fit_values, coeff)

    # prepare residuals for the QA plot
    residuals = wavelengths - chebval(pixels, coeff)

    # plot data
    show_1d_fit_QA(
        pixels,
        wavelengths,
        x_fit_values=x_fit_values,
        y_fit_values=y_fit_values,
        residuals=residuals,
        x_label="Pixels in spectral direction",
        y_label="Wavelength (Å)",
        legend_label="Reidentified lines",
        title=f"1D fit of reidentified lines through the middle of the detector.\n"
        f"Polynomial order : {ORDER_WAVELEN_REID}.\n"
        "Check the fit and residuals for any irregularities.\n"
        "If needed, change relative parameters in the config file.",
    )


def fit_2d(good_lines: dict):

    logger.info("Preparing to fit a 2d polynomial through whole delector...")

    # extract the polynomial order parameter for the fit in spectral direction
    ORDER_WAVELEN_REID = wavecalib_params["ORDER_WAVELEN_REID"]
    # extract the polynomial order parameter for the fit in spatial direction
    ORDER_SPATIAL_REID = wavecalib_params["ORDER_SPATIAL_REID"]

    logger.info(
        f"Fitting a 2d wavelength solution of order {ORDER_WAVELEN_REID} in spectral direction and "
        f"order {ORDER_SPATIAL_REID} in spatial direction to reidentified lines..."
    )

    N_SPACIAL = (
        detector_params["xsize"]
        if detector_params["dispersion"]["spectral_dir"] == "y"
        else detector_params["ysize"]
    )

    spacial_pixel_column = np.arange(0, N_SPACIAL)

    offset_values = np.array([])
    spectral_pixels = np.array([])
    spacial_pixels = np.array([])

    for key in good_lines.keys():
        offset_values = np.append(offset_values, good_lines[key][0])
        spectral_pixels = np.append(spectral_pixels, good_lines[key][1])
        spacial_pixels = np.append(spacial_pixels, good_lines[key][2])

    spectral_pixels = spectral_pixels.flatten()
    wavelength_values = offset_values.flatten()
    spacial_pixels = spacial_pixels.flatten()

    print("spec", spectral_pixels.shape)
    print("spacial", spacial_pixels.shape)
    print("wave", wavelength_values.shape)

    """
    # flatten the lists
    spectral_pixels = np.concatenate(spectral_pixels)
    spacial_pixels = np.concatenate(spacial_pixels)
    wavelength_values = np.concatenate(wavelength_values)
    

    wavelength_dict = {}

    for wavelength in np.unique(wavelength_values):
        mask = wavelength_values == wavelength
        spectral_coords = spectral_pixels[mask]
        spacial_coords = spacial_pixels[mask]
        wavelength_dict[wavelength] = {"spectral_coords": spectral_coords, "spacial_coords": spacial_coords}
    """

    # set up the fitting model

    coeff_init = Chebyshev2D(
        x_degree=ORDER_WAVELEN_REID,
        y_degree=ORDER_SPATIAL_REID,
    )

    fitter = LevMarLSQFitter()

    fit2D_REID = fitter(coeff_init, spectral_pixels, spacial_pixels, wavelength_values)

    residuals = wavelength_values - fit2D_REID(spectral_pixels, spacial_pixels)
    RMS = np.sqrt(np.mean(residuals**2))

    plt.figure(figsize=(10, 8))
    plt.plot(spacial_pixels, residuals, "x")
    plt.xlabel("Spacial Pixels")
    plt.ylabel("Residuals (Å)")
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Residuals of the 2D Fit. RMS: {RMS}")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(spectral_pixels, residuals, "x")
    plt.xlabel("Specral Pixels")
    plt.ylabel("Residuals (Å)")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals of the 2D Fit")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.scatter(spectral_pixels, spacial_pixels, c=residuals, cmap="viridis", s=5)
    plt.colorbar(label="Residuals (Å)")
    plt.xlabel("Spectral Pixels")
    plt.ylabel("Spatial Pixels")
    plt.title("Residuals of the 2D Fit")
    plt.show()

    return fit2D_REID


def construct_wavelength_map(fit2D_REID, master_arc):
    """
    Construct the wavelength map by evaluating the fit at every pixel.

    Parameters
    ----------
    fit2D_REID : `~astropy.modeling.models.Chebyshev2D`
        2D fit model.

    master_arc : array
        Master arc image.

    Returns
    -------
    wavelength_map : 2D array
        Wavelength map.
    """

    # get the shape of the master arc - use it for wavelength map
    x, y = master_arc[0].data.shape

    # create a grid of coordinates
    spac_coords, spec_coords = np.mgrid[:x, :y]

    # evaluate the fit at every pixel
    logger.info("Constructing a wavelength map through whole detector...")
    logger.info("Evaluating the 2D fit at every pixel...")
    wavelength_map = fit2D_REID(spec_coords, spac_coords)

    # check for negative values or nans or infs in the wavelength map
    if (
        np.any(wavelength_map < 0)
        or np.any(np.isnan(wavelength_map))
        or np.any(np.isinf(wavelength_map))
    ):
        logger.error(
            "Negative values, NaNs or Infs found in the wavelength map. "
            "Check the fit and the data."
        )

    return wavelength_map


def plot_verticals(fit2D_REID, good_lines: dict, figsize=(18, 12)):
    """
    Plot vertical slices of the wavelength map for QA of the lines tilts.

    Parameters
    ----------
    wavelength_map : 2D array
        Wavelength map.
    """

    # unpack goodlines:
    offsets_1d = [good_lines[key][0] for key in good_lines.keys()]
    centers_1d = [good_lines[key][1] for key in good_lines.keys()]
    spatials_1d = [good_lines[key][2] for key in good_lines.keys()]
    coeffs_1d = [good_lines[key][3] for key in good_lines.keys()]
    fitted_centers = [good_lines[key][4] for key in good_lines.keys()]

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for i, coeffs in enumerate(coeffs_1d):
        central_spec = fitted_centers[i]
        axs[0].plot(
            spatials_1d[i],
            chebval(spatials_1d[i], coeffs),
            label=f"Central spec: {central_spec}",
        )

    axs[0].set_title("Individual line tilt fits")
    axs[0].set_xlabel("Spatial Pixels")
    axs[0].set_ylabel("Fitted Offsets from Spatial Center")
    axs[0].legend()

    for i, spectral_array in enumerate(centers_1d):
        offsets_2d = fit2D_REID(spectral_array, spatials_1d[i])
        axs[1].plot(spatials_1d[i], offsets_2d)

    axs[1].set_title("2D Fit detector tilt fit")
    axs[1].set_xlabel("Spatial Pixels")
    axs[1].set_ylabel("Fitted Offsets from Spatial Center")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_wavemap(wavelength_map, figsize=(18, 12)):

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


def plot_wavelengthcalib_QA(wavelength_map, good_lines: dict, fit2d_REID):
    """
    Plot the wavelength calibration QA plots.

    Parameters
    ----------
    wavelength_map : 2D array
        Wavelength map.
    """

    logger.info("Preparing wavelength calibration QA plots...")
    plot_wavemap(wavelength_map)
    plot_verticals(fit2d_REID, good_lines)


def write_waveimage_to_disc(wavelength_map, master_arc):
    # TODO this is not used anymore - keep until done developing,
    # then remove it
    """
    Write the wavelength calibration results (waveimage) to disc.

    Parameters
    ----------
    wavelength_map : 2D array
        Wavelength map.
    """

    logger.info("Writing wavelength calibration results to disc...")

    # steal header from master_arc
    header = master_arc[0].header
    write_to_fits(wavelength_map, header, "wavelength_map.fits", output_dir)

    logger.info("Wavelength calibration results written to disc.")


def write_fit2d_REID_to_disc(fit2D_REID):
    """
    Write the 2D fit results to disc.

    Parameters
    ----------
    fit2D_REID : `~astropy.modeling.models.Chebyshev2D`
        2D fit model.
    """

    logger.info("Writing 2D fit results to disc...")

    # change to output directory dir
    os.chdir(output_dir)

    # Write fit2D_REID to disk
    with open("fit2D_REID.pkl", "wb") as file:
        pickle.dump(fit2D_REID, file)

    # change back to original directory

    os.chdir("..")

    logger.info(
        f"2D fit results written to disc in {output_dir}, filename fit2D_REID.pkl."
    )


def load_fit2d_REID_from_disc():
    """
    Load the 2D fit results from disc.

    Returns
    -------
    fit2D_REID : `~astropy.modeling.models.Chebyshev2D`
        2D fit model.
    """

    logger.info("Loading 2D wavelength solution from disc...")

    # change to output directory dir
    os.chdir(output_dir)

    # Load fit2D_REID from disk
    with open("fit2D_REID.pkl", "rb") as file:
        fit2D_REID = pickle.load(file)

    logger.info("Wavelength solution loaded.")

    # change back to original directory
    os.chdir("..")

    return fit2D_REID


def run_wavecalib():
    """
    Run the wavelength calibration routine.
    """
    logger.info("Starting wavelength calibration routine...")

    pixnumber, wavelength = read_pixtable()

    master_arc = get_master_arc()

    logger.info("Reidentifying the lines...")

    # reidentified_lines = reidentify(pixnumber, wavelength, master_arc)

    # logger.info("Reidentification done.")
    # logger.info("Starting the fitting routine...")

    # fit_1d_QA(reidentified_lines)

    good_lines = trace_tilts(pixnumber, wavelength, master_arc)

    fit_2d_results = fit_2d(good_lines)

    write_fit2d_REID_to_disc(fit_2d_results)

    wavelength_map = construct_wavelength_map(fit_2d_results, master_arc)

    plot_wavelengthcalib_QA(wavelength_map, good_lines, fit_2d_results)

    write_waveimage_to_disc(wavelength_map, master_arc)

    logger.info("Wavelength calibration routine done.")
    print("\n-----------------------------\n")


#
if __name__ == "__main__":
    run_wavecalib()
    # trace_tilts()
