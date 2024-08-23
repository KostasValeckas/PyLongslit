# =============================================================================
# reidentify
# =============================================================================
# For REIDENTIFY, I used astropy.modeling.models.Chebyshev2D
# Read idarc

from logger import logger
from parser import wavecalib_params, output_dir
import numpy as np
from astropy.table import Table, Column
from utils import open_fits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.modeling.models import Gaussian1D, Chebyshev2D, Const1D
from astropy.modeling.fitting import LevMarLSQFitter
import matplotlib


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
            formatter = ticker.ScalarFormatter(useOffset=False, useMathText=False, useLocale=False)
            formatter.set_scientific(False)
            ax_col.yaxis.set_major_formatter(formatter)


    title_text = (
        f"Reidentification Results. Green: accepted, Red: rejected. \n" 
        f"Acceptance Criteria: \n"
        f"Allowing deviation of {TOL_REID} in pixels from centrum guess. "
        f"\n Allowing FWHM deviation of {TOL_REID_FWHM} pixels from initial FWHM guess of {FWHM} \n"
        "Check that the accepted fits are in fact good fits."
    )
    

    fig.suptitle(title_text, fontsize=11, va='top', ha='center')

    # Add a single x-label and y-label for the entire figure
    fig.text(0.5, 0.04, 'Pixels in spectral direction', ha='center', va='center', fontsize=12)
    fig.text(0.04, 0.5, 'Counts (ADU)', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.show()
    


def reidentify(pixnumber, wavelength, master_arc):
    """

    Parameters
    ----------
    pixnumber : array
        Pixel numbers.

    wavelength : array
        Wavelengths corresponding to the pixel numbers.
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

    # create a container for the reidentified lines
    line_REID = np.zeros((len(ID_init), N_REID - 1))

    # centers of every step spacially
    spatialcoord = np.arange(0, (N_REID - 1) * STEP_REID, STEP_REID) + STEP_REID / 2

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

    # loop over the spatial slices, and reidentify the lines
    for i in range(0, N_REID):
        # limits of the slice
        lower_cut, upper_cut = i * STEP_REID, (i + 1) * STEP_REID
        # sum over the slice
        reidentify_i = np.sum(master_arc[0].data[lower_cut:upper_cut, :], axis=0)

        logger.info(
            f"Reidentifying slice (spacial coordinates) {lower_cut}:{upper_cut}..."
        )

        # container for the reidentified lines
        peak_gauss_REID = []



        # re-identify every hand-identified line
        for j, peak_pix_init in enumerate(ID_init["peak"]):
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

                # build a Gaussian fitter with an added constant
                g_init = Gaussian1D(
                    amplitude=A_init,
                    mean=mean_init,
                    stddev=stddev_init,
                    bounds={
                        "amplitude": (0, 2 * np.max(cropped_spec)),
                        "stddev": (0, TOL_REID),
                    },
                )
                const = Const1D(amplitude=0)
                g_model = g_init + const

                # perform the fit
                fitter = LevMarLSQFitter()
                g_fit = fitter(g_model, x_cropped, cropped_spec)

                # extract the fitted peak position and FWHM:
                fit_center = g_fit.mean_0.value
                fitted_FWHM = g_fit.stddev_0.value * gaussian_sigma_to_fwhm

                # this is the cyclic indexing mechanism for the QA plots
                if i == plot_at_index:
                    subplot_index = (j - j_offset) // plot_width, (j - j_offset) % plot_width
            

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
                        ax[subplot_index].plot(x_cropped, cropped_spec, "x", color='black')
                        # plot the fit
                        x_fine = np.linspace(x_cropped[0], x_cropped[-1], 1000)
                        ax[subplot_index].plot(x_fine, g_fit(x_fine), color='red')
                else:
                    peak_gauss_REID.append(fit_center)
                    if i == plot_at_index:
                        # plot the accepted fits
                        # plot the spectrum
                        ax[subplot_index].plot(x_cropped, cropped_spec, "x", color='black')
                        # plot the fit
                        x_fine = np.linspace(x_cropped[0], x_cropped[-1], 1000)
                        ax[subplot_index].plot(x_fine, g_fit(x_fine), color='green')

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
        
        # last plot - plot the last plots even if master plot is not filled up
        if i == plot_at_index:
            show_reidentify_QA_plot(fig, ax, TOL_REID, TOL_REID_FWHM, FWHM)

        # slice done 
        logger.info(f"Slice {i} reidentified. Reidentified {np.sum(~np.isnan(peak_gauss_REID))} lines out of {len(ID_init)}.\n----")
    logger.info("Re-identification completed. Starting fitting procedure...")


def run_wavecalib(): 
    """
    Run the wavelength calibration routine.
    """
    logger.info("Starting wavelength calibration routine...")

    pixnumber, wavelength = read_pixtable()

    master_arc = get_master_arc()

    reidentify(pixnumber, wavelength, master_arc)


if __name__ == "__main__":
    run_wavecalib()
