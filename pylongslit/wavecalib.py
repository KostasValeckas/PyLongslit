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
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

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

    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params

    path_to_pixtable = wavecalib_params["center_guess_pixtable"]

    logger.info(f"Trying to read center_guess_pixtable table from {path_to_pixtable}...")

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

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir
    from pylongslit.utils import open_fits

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

    from pylongslit.logger import logger

    logger.warning(message)
    logger.warning(
        "This is expected for some lines, but pay attention "
        "to the upcoming quality assessment plots."
    )


def update_model_parameters(g_model, g_fit):
    """
    Update the model parameters with the fitted values.

    Helper method for avoiding code repetition.

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

    g_model.mean_0.bounds = (g_model.mean_0.value - tolerance_mean, g_model.mean_0.value + tolerance_mean)
    g_model.stddev_0.bounds = (g_model.stddev_0.value - tolerance_FWHM * gaussian_fwhm_to_sigma,  g_model.stddev_0.value + tolerance_FWHM * gaussian_fwhm_to_sigma)

    


def fit_arc_1d(spectral_coords, center_row_spec, fitter, g_model, R2_threshold=0.99,
                bench_value=None,
                    bench_tolerance=1.0):
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
        Fitter object.trace_tilt

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
        #TODO: the below exception is not a good fix - refactor
        try:
            g_fit = fitter(g_model, spectral_coords, center_row_spec)
        except (TypeError, ValueError, astropy.modeling.fitting.NonFiniteValueError):
            return False

    R2 = r2_score(center_row_spec, g_fit(spectral_coords))



    if R2 < R2_threshold:
        
        #plt.plot(spectral_coords, center_row_spec, "x", color="black", label = f"R2: {R2}")
        #plt.plot(spectral_coords, g_fit(spectral_coords), color="red")
        #plt.legend()
        #plt.title(f"Failed at R2: {R2}")
        #plt.show()

        #print("fit_1d_fail")

        return False
    
    #print("Good fit")

    #plt.plot(spectral_coords, center_row_spec, "x", color="black", label = f"R2: {R2}")
    #plt.plot(spectral_coords, g_fit(spectral_coords), color="green")  
    #plt.show()

    if bench_value is not None:
        if np.abs(g_fit.mean_0.value - bench_value) > bench_tolerance:
            #print("BIG CHANGE")
            #print(g_fit.mean_0.value, bench_value)
            return False


    update_model_parameters(g_model, g_fit)

    return True


def trace_line_tilt(
    master_arc,
    N_ROWS,
    center_row,
    fitter,
    g_model,
    FWHM_guess
):
    
    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params

    # container for fit parameters.
    # Sometimes used for initial guesses for the next row.
    # For holds more information than needed, but this is nice to have in
    # further developtment or debugging.
    all_params = {}
    # mask for keeping the good fits, will be updated in the loop


    #TODO this is very hacked, fix later

    pixel_cut_extension = wavecalib_params["pixel_cut_extension"]

    start_pixel = wavecalib_params["arcline_start"]
    end_pixel = wavecalib_params["arcline_end"]

    if center_row > end_pixel or center_row < start_pixel:
        logger.critical(f"Center row of arc frame at {center_row} is outside the defined start and end pixels {start_pixel, end_pixel}.")
        logger.critical("Add an offset to the middle cut or change the arc line limits (this is done in the config file).")
        exit(-1)

    cut_width = pixel_cut_extension + 1

    index_dict = {}
    num_it = 0
    for i in chain(range(center_row, end_pixel, cut_width), range(center_row - 1, start_pixel, -cut_width)): 
        index_dict[str(i)] = num_it
        num_it += 1

    keep_mask = np.array([], dtype = bool)

    spectral_coords = np.arange(master_arc.data.shape[1])

    # read the parameter that decides when to abort the trace - i.e.
    # when the fit is bad for more than a certain number of rows
    TILT_REJECTION_LINE_FRACTION = wavecalib_params["TILT_REJECT_LINE_FRACTION"]
    bad_fit_counter = 0
    bad_fit_threshold = (num_it * TILT_REJECTION_LINE_FRACTION)

    # R2 tolerance for the fit of every row
    TILT_TRACE_R2_TOL = wavecalib_params["TILT_TRACE_R2_TOL"]

    last_good_center = g_model.mean_0.value

    used_spacial =[]

    # Loop from center and up and then down with a jump of 5
    for i in chain(range(center_row, end_pixel, cut_width), range(center_row - 1, start_pixel, -cut_width)):

        # if we are starting to loop downwards, we need to update the initial guesses
        # back to the center row values manually.
        if i == center_row - 1:
            #print("DETECTED DOWNWARDS MOTION")
            g_model.amplitude_0 = all_params[i + 1]["amplitude"]
            g_model.mean_0 = all_params[i + 1]["center"]
            g_model.stddev_0 = all_params[i + 1]["FWHM"] * gaussian_fwhm_to_sigma
            g_model.beta_0 = all_params[i + 1]["beta"]
            g_model.amplitude_1 = all_params[i + 1]["amplitude_1"]
        
            tolerance_mean = wavecalib_params["TOL_MEAN"]
            tolerance_FWHM = wavecalib_params["TOL_FWHM"]
            g_model.mean_0.bounds = (g_model.mean_0.value - tolerance_mean, g_model.mean_0.value + tolerance_mean)
            g_model.stddev_0.bounds = (g_model.stddev_0.value - tolerance_FWHM * gaussian_fwhm_to_sigma, g_model.stddev_0.value + tolerance_FWHM * gaussian_fwhm_to_sigma)

            last_good_center = g_model.mean_0.value

        # clip out the subimage around the line
        start_pixel = int(
            g_model.mean_0.value - FWHM_guess
        )
        end_pixel = int(
            g_model.mean_0.value + FWHM_guess
        )

        if start_pixel < 0 or end_pixel > master_arc.shape[1]:
            #print("skipping due to edge at: ", i)
            #print(start_pixel, end_pixel)
            #all_centers = np.array([all_params[key]["center"] for key in all_params.keys()])
            #good_centers = all_centers[keep_mask]
            #plt.plot(used_spacial, good_centers, "o")
            #plt.show()
            continue
        #print("now at: ", i)

        # np.mean(sub_image[center_row-2:(center_row+2) + 1, :], axis = 0)
        center_row_spec = np.mean(master_arc[i-pixel_cut_extension: i+pixel_cut_extension + 1, start_pixel:end_pixel], axis = 0)
        spectral_coords_sub = spectral_coords[start_pixel:end_pixel]

        jump_tolerance = wavecalib_params["jump_tolerance"]

        keep_bool = fit_arc_1d(
            spectral_coords_sub,
            center_row_spec,
            fitter,
            g_model,
            R2_threshold=TILT_TRACE_R2_TOL,
            bench_value=last_good_center,
            bench_tolerance=jump_tolerance
        )

        if not keep_bool:

            # try minimizing the FWHM - might be due to close lines:

            # clip out the subimage around the line
            start_pixel += 2
            end_pixel -= 2

            if start_pixel >= end_pixel:
                keep_bool = False

            else:

                center_row_spec = np.mean(master_arc[i-pixel_cut_extension: i+pixel_cut_extension + 1, start_pixel:end_pixel], axis = 0)
                spectral_coords_sub = spectral_coords[start_pixel:end_pixel]


                #print("RECUTTING AND TRYING AGAIN")

                keep_bool = fit_arc_1d(
                    spectral_coords_sub,
                    center_row_spec,
                    fitter,
                    g_model,
                    R2_threshold=TILT_TRACE_R2_TOL,
                    bench_value=last_good_center,
                    bench_tolerance=jump_tolerance
                )

            if keep_bool:
                #print("SUCCESSFUL RE-FIT")
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

            else:

                # for several bad fits, might be that the center guess is off
                # if enough good fits are present, retry the fit

                if np.sum(keep_mask) > 10:
                    #print("ENOUGH GOOD FITS TO RETRY")

                    all_centers = np.array([all_params[key]["center"] for key in all_params.keys()])
                    
                    good_centers = all_centers[keep_mask]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        coeff = chebfit(used_spacial, good_centers, deg=wavecalib_params["ORDER_SPATIAL_TILT"])
                    #plt.plot(used_spacial, good_centers, "o")
                    #plt.plot(used_spacial, chebval(used_spacial, coeff))
                    #plt.show()

                    new_center_guess = chebval(i, coeff)

                    R2 = r2_score(good_centers, chebval(used_spacial, coeff))

                    if R2 < wavecalib_params["SPACIAL_R2_TOL"]:
                        #print("R2 TOO LOW FOR NEW CENTER GUESS")
                        keep_bool = False

                    elif abs(new_center_guess - g_model.mean_0.value) > wavecalib_params["TOL_MEAN"]:
                        #print("TOO BIG CHANGE IN NEW CENTER GUESS")
                        keep_bool = False

                    
                    else:

                        g_model.mean_0 = new_center_guess

                        #print("NEW CENTER GUESS", new_center_guess)

                        temp = used_spacial.copy()
                        min, max = np.min(temp), np.max(temp)
                        temp = np.linspace(min, max, 1000)
                        temp_extra = np.linspace(max, max+100, 1000)

                        #plt.plot(temp, chebval(temp, coeff))
                        #plt.plot(temp_extra, chebval(temp_extra, coeff))
                        #plt.show()  


                        keep_bool = fit_arc_1d(
                            spectral_coords_sub,
                            center_row_spec,
                            fitter,
                            g_model,
                            R2_threshold=TILT_TRACE_R2_TOL,
                            bench_value=last_good_center,
                            bench_tolerance=jump_tolerance 
                        )

                    if keep_bool:
                        #print("SUCCESSFUL RETRY IN FITTED CENTER ESTIMATION")
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

    # extract the centers of the lines
    all_centers = np.array([all_params[key]["center"] for key in all_params.keys()])

    logger.info(
        f"Line traced successfully at pixel {all_params[center_row]['center']}."
    )

    return all_centers, np.array(used_spacial), keep_mask


def show_cyclic_QA_plot(fig, ax, title_text=None, x_label=None, y_label=None, show=True):
    """ """

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
            # some plots might not have a legend, ignore the warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                ax_col.legend()

    fig.suptitle(title_text, fontsize=11, va="top", ha="center")

    # Add a single x-label and y-label for the entire figure
    if x_label is not None:
        fig.text(0.5, 0.04, x_label, ha="center", va="center", fontsize=12)

    if y_label is not None:
        fig.text(
            0.04,
            0.5,
            y_label,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=12,
        )
    if show: plt.show(block = True)


def trace_tilts(lines, master_arc):
    """
    Trace the tilts of the lines in the arc spectrum.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params, developer_params

    logger.info("Tracing the tilts of the lines in the arc spectrum...")

    plt.close("all")    

    # get detector shape parameters
    N_ROWS = master_arc[0].data.shape[0]
    
    #the offset is needed if the middle of the detector is not a good place
    # to take a sample
    center_row_offset = wavecalib_params["offset_middle_cut"]
    
    center_row = (N_ROWS // 2) + center_row_offset
    spacial_coords = np.arange(N_ROWS)
    spectral_coords = np.arange(master_arc[0].data.shape[1])

    # get the tolerance for the RMS of the tilt line fit
    R2_TOL = wavecalib_params["SPACIAL_R2_TOL"]

    # containers for the good lines and RMS values
    good_lines = {}
    RMS_all = {}

    spacial_fit_order = wavecalib_params["ORDER_SPATIAL_TILT"]

    tolerance_mean = wavecalib_params["TOL_MEAN"]

    tolerance_FWHM = wavecalib_params["TOL_FWHM"]

    # counters for QA and debugging
    good_traces = 0
    good_fits = 0

    plot_height = 6
    plot_width = 2
    figsize = (24, 20)

    if developer_params["debug_plots"]:

        # sometimes used for debugging - plot the line centers

        all_peak_pix = [lines[key]["peak_pix"] for key in lines.keys()]

        plt.imshow(master_arc[0].data, aspect="auto")
        plt.plot(all_peak_pix, np.full_like(all_peak_pix, center_row), "x", color="red")
        plt.show()
        #exit()

    fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)

    # for book-keeping the cyclic plots
    j = 0

    title_text = (
        f"Line Tilt Tracing Results. Green: accepted, Red: rejected. Polynomial order: {spacial_fit_order}.\n"
        f"Acceptance Criteria: R2 of the spacial polynomial fit > {R2_TOL}.\n"
        f"These parameters can be changed in the configuration file."
    )

    for key in lines.keys():
        pixel = lines[key]["peak_pix"]
        wavelength = lines[key]["wavelength"]
        FWHM_guess = lines[key]["FWHM"]
        beta = lines[key]["beta"]
        amplitude = lines[key]["amplitude"]
        constant = lines[key]["constant"]

        # algorithm: crop the spectrum around the line, fit a Gaussian to the line
        # in the center row to refine crop position, then trace the line center every row.
        # Fit a polynomial to the line centers to get the tilt of the line.
        # Use RMS of the polynomial fit as a quality metric.
        # If the RMS is too high, discard the line. Else, redo the fit
        # but now for the offset from the center row.

        print("\n-------------------------------------")

        logger.info(f"Tracing line-tilt at pixel {pixel}...")

        # first guess - this will be refined
        start_pixel = pixel - FWHM_guess
        end_pixel = pixel + FWHM_guess

        pixel_cut_extension = wavecalib_params["pixel_cut_extension"]


        sub_image = master_arc[0].data[:, int(start_pixel):int(end_pixel)]
        center_row_spec = np.mean(sub_image[center_row-pixel_cut_extension:(center_row+pixel_cut_extension) + 1, :], axis = 0)
        spectral_coords_sub = spectral_coords[int(start_pixel):int(end_pixel)]


        # initialize the fitter - this fitter will be used for all the fits
        A_init = amplitude
        mean_init = pixel
        stddev_init = FWHM_guess * gaussian_fwhm_to_sigma
        beta_init = beta  

        # TODO - maybe bounds should be set in the config file?
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

        if False:

            plt.plot(spectral_coords_sub, center_row_spec, "x", color="black")
            # plot the initial fitting model 
            spec_fine = np.linspace(
                spectral_coords_sub[0], spectral_coords_sub[-1], 1000
            )
            plt.plot(
                spec_fine,
                g_model(spec_fine),
                color="red",
                label="initial guess"
            )
            plt.legend()
            plt.show()
            # Perform 2-pass fit to get a good estimate of the line center



        centers, spacial_coords_used, mask = trace_line_tilt(
            master_arc[0].data,
            N_ROWS,
            center_row,
            fitter,
            g_model,
            FWHM_guess
        )

        if False:
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

        # do a poly fit to the good centers
        coeff = chebfit(good_spacial_coords, good_centers, deg=spacial_fit_order)

        R2_spat = r2_score(good_centers, chebval(good_spacial_coords, coeff))
        # prepare color for QA plot

        plot_color = "green"

        # if the RMS is too high, discard the line
        if R2_spat < R2_TOL:
            arc_trace_warning(
                f"Good line trace, but the RMS={R2_spat} of the spacial polynomial is higher than the tolerance {R2_TOL}. "
                f"Current user defined spacial fitting order: {spacial_fit_order}."
            )
            plot_color = "red"

        else:
            # if the initial poly fit is good, fit the offsets.

            # calculate center pixel and use it to find offsets
            center_pixel = chebval(center_row, coeff)

            offsets = center_pixel - good_centers

            coeff_offset = chebfit(good_spacial_coords, offsets, deg=spacial_fit_order)

            # fit should be good, as it is the same as the good_centers, just
            # with an offset. But still check just to be sure.
            R2_offsets = r2_score(offsets, chebval(good_spacial_coords, coeff_offset))

            good_fits += 1

            if False:
                residuals = good_centers - chebval(good_spacial_coords, coeff)
                plt.plot(good_spacial_coords, residuals, "x", color="black")
                plt.show()

                        # if the RMS is too high, discard the line
            if R2_offsets < R2_TOL:
                arc_trace_warning(
                    f"Good line trace, but the RMS={R2_offsets} of the spacial polynomial is higher than the tolerance {R2_TOL}. "
                    f"Current user defined fitting order: {spacial_fit_order}."
                )

                plot_color = "red"

            else:

                good_lines[wavelength] = (
                    offsets,
                    good_centers,
                    good_spacial_coords,
                    coeff_offset,
                    center_pixel,
                )

            RMS_all[wavelength] = R2_offsets


        ax[j][0].plot(good_spacial_coords, good_centers, "x", color="black")
        # plot the fit
        spat_fine = np.linspace(0, N_ROWS, 1000)
        ax[j][0].plot(
            spat_fine,
            chebval(spat_fine, coeff),
            color=plot_color,
            label=f"Fit R2: {R2_spat}",
        )

        residuals = good_centers - chebval(good_spacial_coords, coeff)

        ax[j][1].plot(
            good_spacial_coords, residuals, "x", color="black", label="Residuals"
        )
        ax[j][1].axhline(0, color="red", linestyle="--")
        ax[j][1].legend()
    
        if (
            # this condition checks if the plot has been filled up
            # plots if so, and adjust the offset so a new
            # plot can be created and filled up
            j == (plot_height - 1)
        ):
            show_cyclic_QA_plot(
                fig, ax, title_text, "Spacial Pixels", "Spectral Pixels"
            )

            j = 0

            fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)
            
            continue

        j += 1

    show_cyclic_QA_plot(fig, ax, title_text, "Spacial Pixels", "Spectral Pixels")

    print("\n-------------------------------------")
    logger.info("Line tilt tracing done.")
    logger.info(f"Number of good traces: {good_traces} out of {len(lines)}.")
    logger.info(f"Number of good fits: {good_fits} out of {good_traces}.")
    print("\n-------------------------------------")

    return good_lines


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
    from pylongslit.parser import wavecalib_params
    from pylongslit.utils import show_1d_fit_QA

    # tolerance for pixel shift from hand-identified lines to Gaussian fit
    tol_mean = wavecalib_params["TOL_MEAN"]
    # rough guess of FWHM of lines in pixels
    FWHM = wavecalib_params["FWHM"]
    # tolerance for FWHM of the Gaussian fit
    tol_FWHM = wavecalib_params["TOL_FWHM"]

    final_r2_tol = wavecalib_params["REIDENTIFY_R2_TOL"]

    fit_order = wavecalib_params["ORDER_WAVELEN_1D"]

    # create a container for hand-identified lines
    ID_init = Table(dict(peak=pixnumber, wavelength=wavelength))

    # container for re-identified lines. This will be the final product
    line_REID = {}

    # we extract the arc spectrum from the middle of master arc
    # +/- 2 pixels, as we assume the variation in tilts there is negligible

    # offset is needed if the mddle of the detector is not a good place
    # to take a sample
    middle_row_offset = wavecalib_params["offset_middle_cut"]
    pixel_cut_extension = wavecalib_params["pixel_cut_extension"]

    middle_row = (master_arc[0].data.shape[0] // 2) + middle_row_offset
    # limits of the slice
    lower_cut, upper_cut = middle_row - pixel_cut_extension, middle_row + pixel_cut_extension

    # sum over the slice
    spec_1d = np.mean(master_arc[0].data[lower_cut:upper_cut+1, :], axis=0)

    spectral_coords = np.arange(len(spec_1d))

    # this offset value allows to make cyclic subplots, as we use the index
    # together with integer division and module to cycle through subplots
    j_offset = 0

    plot_height = 4
    plot_width = 3
    figsize = (24, 20)

    fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)

    title_text = (
        f"Reidentification Results. Green: accepted, Red: rejected. \n"
        f"Acceptance Criteria:  R2 > {final_r2_tol}.\n"
        f"Initial guess from FWHM is {FWHM}, allowed deviation from line center guess is {tol_mean}, allowed deviation from FWHM guess is {tol_FWHM}. \n"
        f"All of these parameters can be changed in the configuration file."
    )

    # re-identify every hand-identified line
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

        # if empty array - keep looping
        if len(cropped_spec) == 0:
            continue

        # initialize the fitter - this fitter will be used for all the fits
        A_init = np.max(cropped_spec)
        mean_init = peak_pix_init
        stddev_init = FWHM * gaussian_fwhm_to_sigma
        beta_init = 2  # beta = 2 means simple Gaussian form

        # TODO - maybe bounds should be set in the config file?
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

        # Perform 2-pass fit to get a good estimate of the line center

        bad_line = False

        R2 = None

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

            # get a better estimate of the center
            start_pixel = int(fit_center - FWHM_local)
            end_pixel = int(fit_center + FWHM_local)

            if (end_pixel - start_pixel) < len(g_model.param_names):
                arc_trace_warning(
                    "Less points in line spectrum than the spacial fit order. "
                    "FWHM is possibly too small or tolerance is too high."
                )

                bad_line = True
                break

            # crop the spectrum around the guess
            cropped_spec = spec_1d[start_pixel:end_pixel]
            cropped_spectral_coords = spectral_coords[start_pixel:end_pixel]

            # remove any nans and infs from the cropped spectrum
            nan_inf_mask = np.isnan(cropped_spec) | np.isinf(cropped_spec)
            cropped_spectral_coords = cropped_spectral_coords[~nan_inf_mask]
            cropped_spec = cropped_spec[~nan_inf_mask]

            # update the fitter parameters
            update_model_parameters(g_model, g_fit)

        plot_color = "green"

        if bad_line or R2 < final_r2_tol:
            plot_color = "red"

        subplot_index = (j - j_offset) // plot_width, (j - j_offset) % plot_width

        ax[subplot_index].plot(
            cropped_spectral_coords, cropped_spec, "x", color="black"
        )
        # plot the fit
        spec_fine = np.linspace(
            cropped_spectral_coords[0], cropped_spectral_coords[-1], 1000
        )
        try:
            ax[subplot_index].plot(
                spec_fine,
                g_fit(spec_fine),
                color=plot_color,
                label="fit R2: {:.2f}, FWHM: {:.2f}".format(R2,FWHM_local),
            )
            ax[subplot_index].axvline(
                mean_init, color="black", linestyle="--", label=f"Initial center guess {mean_init:.2f}"
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
            ax[subplot_index].plot(
                spec_fine,
                g_fit(spec_fine),
                color=plot_color
            )


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
            if (j+1) < len(ID_init):
                fig, ax = plt.subplots(plot_height, plot_width, figsize=figsize)

        if not bad_line and R2 > final_r2_tol:
            line_REID[str(j)] = {
                "peak_pix": fit_center,
                "wavelength": wavelength[j],
                "FWHM": FWHM_local,
                "beta": beta,
                "amplitude": g_fit.amplitude_0.value,
                "constant": g_fit.amplitude_1.value
            }

    # plot the last plot if not filled up
    show_cyclic_QA_plot(fig, ax, title_text, "Spectral Pixels", "Counts (ADU)")

    all_pixels = [line_REID[key]["peak_pix"] for key in line_REID.keys()]
    all_wavelengths = [line_REID[key]["wavelength"] for key in line_REID.keys()]
    all_fwhm = [line_REID[key]["FWHM"] for key in line_REID.keys()]

    fit = chebfit(all_pixels, all_wavelengths, deg=fit_order)
    residuals = all_wavelengths - chebval(all_pixels, fit)
    # Design matrix for Chebyshev polynomials
    X = np.polynomial.chebyshev.chebvander(all_pixels, fit_order)

    sigma_squared = np.var(residuals)

    # Calculate the covariance matrix for the coefficients
    # Covariance matrix: C = sigma^2 * (X^T X)^-1
    XT_X_inv = np.linalg.inv(X.T @ X)
    cov_matrix = sigma_squared * XT_X_inv

    # The standard errors of the coefficients are the square roots of the diagonal elements of the covariance matrix
    errors = np.sqrt(np.diag(cov_matrix))

    pixel_linspace = np.linspace(0, len(spec_1d), 1000)
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
        title=f"1D fit of reidentified lines. User-set polynomial order: {fit_order}.",
    )

    # used when tracing tilts
    fwhm_mean = np.mean(all_fwhm)

    plt.close("all")    

    return fit, line_REID, fwhm_mean, errors


def fit_2d_tilts(good_lines: dict, figsize=(18, 12)):

    from pylongslit.logger import logger
    from pylongslit.parser import wavecalib_params, detector_params

    logger.info("Preparing to fit a 2d polynomial tilt through whole delector...")

    # extract the polynomial order parameter for the fit in spectral direction
    ORDER_SPECTRAL = wavecalib_params["ORDER_SPECTRAL_TILT"]
    # extract the polynomial order parameter for the fit in spatial direction
    ORDER_SPATIAL = wavecalib_params["ORDER_SPATIAL_TILT"]

    logger.info(
        f"Fitting a 2d tilt solution of order {ORDER_SPECTRAL} in spectral direction and "
        f"order {ORDER_SPATIAL} in spatial direction to reidentified lines..."
    )

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

    # the following lines extract the good lines from the dictionary
    # and put them in the format that the fitter expects
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

    # plot a scatter plot for QA
    plt.scatter(spectral_pixels, spacial_pixels, c=wavelength_values)
    plt.xlabel("Spectral Pixels")
    plt.ylabel("Spacial Pixels")
    plt.title("Scatter plot of the reidentified lines")
    plt.colorbar(label="Wavelength (Å)")
    plt.show()

    # take a slice at spacial pixel x = 1400
    # to see the linearity of the reidentified lines


    """
    plt.plot(spacial_pixels, "+")
    plt.show()

    for i in range(10):
        mask = spacial_pixels.astype(int) == np.random.choice(spacial_pixels)
        print("mask: ", np.sum(mask))
        spacial_1400 = spacial_pixels[mask]
        spectral_1400 = spectral_pixels
        wavelength_1400 = wavelength_values

        plt.plot(spectral_1400, wavelength_1400)
    plt.show()

    for i in range(10):
        maske = spectral_pixels.astype(int) == int(np.random.choice(spectral_pixels))
        print("mask: ", np.sum(maske))
        spacial_1400 = spacial_pixels
        spectral_1400 = spectral_pixels[maske]
        wavelength_1400 = wavelength_values

        plt.plot(spacial_1400, wavelength_1400)
    
    plt.show()

    exit()
    """

    # set up the fitting model

    coeff_init = Chebyshev2D(
        x_degree=ORDER_SPECTRAL,
        y_degree=ORDER_SPATIAL,
    )

    fitter = LevMarLSQFitter(calc_uncertainties=True)

    fit2D = fitter(coeff_init, spectral_pixels, spacial_pixels, wavelength_values)
    cov = fitter.fit_info['param_cov']
    param_errors = np.sqrt(np.diag(cov))

    residuals = wavelength_values - fit2D(spectral_pixels, spacial_pixels)
    RMS = np.sqrt(np.mean(residuals**2))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(spacial_pixels, spectral_pixels, residuals, c=residuals, cmap='viridis')
    ax.set_ylabel("Spectral Pixels")
    ax.set_xlabel("Spacial Pixels")
    ax.set_zlabel("Residuals (Å)")
    ax.set_title("Residuals of the 2D Fit")
    fig.colorbar(sc, label="Residuals (Å)")
    plt.show()

    # plot QA

    _, axs = plt.subplots(2, 1, figsize=figsize)

    axs[0].plot(spacial_pixels, residuals, "x")
    axs[0].set_xlabel("Spacial Pixels")
    axs[0].set_ylabel("Residuals (Å)")
    axs[0].axhline(0, color="red", linestyle="--")
    axs[0].set_title(f"Residuals of the 2D Fit (Spacial Pixels). RMS: {RMS}")

    axs[1].plot(spectral_pixels, residuals, "x")
    axs[1].set_xlabel("Spectral Pixels")
    axs[1].set_ylabel("Residuals (Å)")
    axs[1].axhline(0, color="red", linestyle="--")
    axs[1].set_title("Residuals of the 2D Fit (Spectral Pixels)")

    plt.show()

    return fit2D, param_errors


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

    spectral_line = np.linspace(0, N_SPECTRAL, N_SPECTRAL)
    spacial_line = np.linspace(0, N_SPACIAL, N_SPACIAL)

    X, Y = np.meshgrid(spectral_line, spacial_line)
    map = fit2D_REID(X, Y)

    return map


def plot_tilt_2D_QA(fit2D_REID, good_lines: dict, figsize=(18, 12)):

    centers_1d = [good_lines[key][1] for key in good_lines.keys()]
    spatials_1d = [good_lines[key][2] for key in good_lines.keys()]
    coeffs_1d = [good_lines[key][3] for key in good_lines.keys()]
    fitted_centers = [good_lines[key][4] for key in good_lines.keys()]

    _, axs = plt.subplots(1, 2, figsize=figsize)

    for i, coeffs in enumerate(coeffs_1d):
        central_spec = fitted_centers[i]
        axs[0].plot(
            spatials_1d[i],
            chebval(spatials_1d[i], coeffs),
            "o",
            label=f"Central spec: {central_spec}",
        )

    axs[0].set_title("Individual line tilt fits")
    axs[0].set_xlabel("Spatial Pixels")
    axs[0].set_ylabel("Fitted Offsets from Spatial Center")
    axs[0].legend()

    for i, spectral_array in enumerate(centers_1d):
        offsets_2d = fit2D_REID(spectral_array, spatials_1d[i])
        axs[1].plot(spatials_1d[i], offsets_2d, "o")

    axs[1].set_title("2D Fit detector tilt fit")
    axs[1].set_xlabel("Spatial Pixels")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def construct_wavelen_map(wavelen_fit, tilt_fit, original_orientation=False):

    from pylongslit.parser import detector_params
    from pylongslit.utils import wavelength_sol

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


def plot_tiltmap(tilt_map, figsize=(18, 12)):

    plt.figure(figsize=figsize)
    plt.imshow(tilt_map, origin="lower")
    plt.colorbar(label="Spacial offset from center pixel (in pixels)")
    plt.title(
        "Tilt map (Offset from spacial center mapped to every pixel of the detector)\n"
        "Inspect the map for any irregularities - it should be a smooth continuum.\n"
    )
    plt.xlabel("Pixels in spectral direction")
    plt.ylabel("Pixels in spatial direction")
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


def plot_wavelengthcalib_QA(good_lines: dict, wave_fit, tilt_fit):

    plot_tilt_2D_QA(tilt_fit, good_lines)

    tilt_map = construct_detector_map(tilt_fit)

    plot_tiltmap(tilt_map)

    wave_map = construct_wavelen_map(wave_fit, tilt_fit)

    plot_wavemap(wave_map)

def simulate_wavelengthcalib_variance(wave_fit, wave_fit_error, tilt_fit, tilt_fit_error, n_sims=1000):

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
    #for _ in tqdm(range(n_sims), desc="Estimating wavelength calibration variance"):
        # simulate the variance in the wavelength map by sampling from the
        # normal distribution of the fit parameters
        random_wave_fit = np.random.normal(wave_fit, wave_fit_error, wave_fit.shape)
        first_param_list.append(random_wave_fit[0])

        random_tilt_params = np.random.normal(tilt_fit.parameters, tilt_fit_error, tilt_fit_error.shape)

        random_tilt_fit = tilt_fit.copy()
        random_tilt_fit.parameters = random_tilt_params

        wave_map = construct_wavelen_map(random_wave_fit, random_tilt_fit)
        wavemap_sum += wave_map
        wavemap_sum_squares += wave_map**2
        print(np.max(wavemap_sum_squares))

        wavemap_diffs = wavemap_initial - wave_map

    wavemap_mean = wavemap_sum / n_sims
    variance_wavemap = wavemap_sum_squares/n_sims - wavemap_mean**2
    error_wavemap = np.sqrt(variance_wavemap)/np.sqrt(n_sims)

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
    plt.title(f"Distribution of the first parameter of the wavelength fit. Original value: {wave_fit[0]}")  
    plt.show()

def calculate_final_fit_rms(wave_fit, tilt_fit, good_lines):
    
    from pylongslit.logger import logger
    from pylongslit.utils import wavelength_sol
    from pylongslit.parser import detector_params


    residuals = np.array([])
    
    for key in good_lines.keys():
        good_x = good_lines[key][1]
        good_y= good_lines[key][2]
        wavelength = float(key)

        wavesol_wavelengths = wavelength_sol(good_x, good_y, wave_fit, tilt_fit)

        residuals = np.append(residuals, wavesol_wavelengths - wavelength)  

    print(residuals)

    
    residuals = np.array(residuals).flatten()
    RMS = np.sqrt(np.mean(residuals**2))

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
    # TODO this is not used anymore - keep until done developing,
    # then remove it
    """
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
    header = master_arc[0].header
    write_to_fits(wavelength_map, header, "wavelength_map.fits", output_dir)

    logger.info("Wavelength calibration results written to disc.")


def write_wavelen_fit_to_disc(fit1d):

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Writing wavelen fit results to disc...")

    # change to output directory dir
    os.chdir(output_dir)

    # Write fit2D_REID to disk
    with open("wavelen_fit.pkl", "wb") as file:
        pickle.dump(fit1d, file)

    # change back to original directory

    os.chdir("..")

    logger.info(
        f"2D tilt fit results written to disc in {output_dir}, filename wavelen_fit.pkl."
    )


def write_tilt_fit_to_disc(fit2D_REID):
    """ """
    from pylongslit.logger import logger
    from pylongslit.parser import output_dir


    logger.info("Writing tilt fit results to disc...")

    # change to output directory dir
    os.chdir(output_dir)

    # Write fit2D_REID to disk
    with open("tilt_fit.pkl", "wb") as file:
        pickle.dump(fit2D_REID, file)

    # change back to original directory

    os.chdir("..")

    logger.info(
        f"2D tilt fit results written to disc in {output_dir}, filename tilt_fit.pkl."
    )


def write_good_tilt_lines_to_disc(good_lines):
    """
    Write the good tilt lines to disc.

    Parameters
    ----------
    good_lines : dict
        Good tilt lines.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Writing good tilt lines to disc...")

    # change to output directory dir
    os.chdir(output_dir)

    # Write good_lines to disk
    with open("good_lines.pkl", "wb") as file:
        pickle.dump(good_lines, file)

    # change back to original directory

    os.chdir("..")

    logger.info("Good tilt lines written to disc.")


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

    # change to output directory dir
    os.chdir(output_dir)

    # Load fit2D_REID from disk
    with open("wavelen_fit.pkl", "rb") as file:
        fit1d = pickle.load(file)

    logger.info("Wavelength solution loaded.")

    # change back to original directory
    os.chdir("..")

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

    # change to output directory dir
    os.chdir(output_dir)

    # Load fit2D_REID from disk
    with open("tilt_fit.pkl", "rb") as file:
        fit2D_REID = pickle.load(file)

    logger.info("Tilt solution loaded.")

    # change back to original directory
    os.chdir("..")

    return fit2D_REID


def get_good_tilt_lines_from_disc():
    """
    Load the good tilt lines from disc.

    Returns
    -------
    good_lines : dict
        Good tilt lines.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Loading good tilt lines from disc...")

    # change to output directory dir
    os.chdir(output_dir)

    # Load good_lines from disk
    with open("good_lines.pkl", "rb") as file:
        good_lines = pickle.load(file)

    logger.info("Good tilt lines loaded.")

    # change back to original directory
    os.chdir("..")

    return good_lines

def construct_wavemap_frame(wave_fit, tilt_fit, error):
    
    from pylongslit.utils import PyLongslit_frame

    master_arc = PyLongslit_frame.read_from_disc("master_arc.fits")

    hdr = master_arc.header

    wave_map = construct_wavelen_map(wave_fit, tilt_fit)

    wave_map_frame = PyLongslit_frame(wave_map, error, hdr, "wavelength_map")

    wave_map_frame.show_frame(normalize=False)
    wave_map_frame.write_to_disc()






def load_fit2d_REID_from_disc():
    """
    Load the 2D fit results from disc.

    Returns
    -------
    fit2D_REID : `~astropy.modeling.models.Chebyshev2D`
        2D fit model.
    """

    if False:

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

    raise NotImplementedError


def run_wavecalib():
    """
    Run the wavelength calibration routine.
    
    """

    from pylongslit.logger import logger
    from pylongslit.utils import hist_normalize

    logger.info("Starting wavelength calibration routine...")

    pixnumber, wavelength = read_pixtable()

    master_arc = get_master_arc()

    logger.info("Reidentifying the lines...")

    wave_sol, lines, fwhm_mean, wavesol_errors = reidentify(pixnumber, wavelength, master_arc)


    write_wavelen_fit_to_disc(wave_sol)

    logger.info("Reidentification done.")
    logger.info("Starting tilt tracing...")


    good_lines = trace_tilts(lines, master_arc)

    write_good_tilt_lines_to_disc(good_lines)
    #good_lines = get_good_tilt_lines_from_disc()



    plt.imshow(hist_normalize(master_arc[0].data), aspect="auto")

    for key in good_lines.keys():
        plt.plot([good_lines[key][1]], [good_lines[key][2]], "x", color="red")

    plt.show()


    fit_2d_tilt_results, tilt_param_errors = fit_2d_tilts(good_lines)

    write_tilt_fit_to_disc(fit_2d_tilt_results)

    """

    # TODO: while developing just read the products from disc

    
    fit_2d_tilt_results = get_tilt_fit_from_disc()
    wave_sol = get_wavelen_fit_from_disc()

    """

    plot_wavelengthcalib_QA(good_lines, wave_sol, fit_2d_tilt_results)

    error = calculate_final_fit_rms(wave_sol, fit_2d_tilt_results, good_lines)

    construct_wavemap_frame(wave_sol, fit_2d_tilt_results, error)



    # write_waveimage_to_disc(wavelength_map, master_arc)

    logger.info("Wavelength calibration routine done.")
    print("\n-----------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit wavecalibration procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()
    
    from pylongslit import set_config_file_path
    
    set_config_file_path(args.config)
    
    run_wavecalib()


if __name__ == "__main__":
    main()

