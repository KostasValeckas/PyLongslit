from astropy.modeling.models import Gaussian1D
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
import matplotlib.pyplot as plt
from astropy.modeling.fitting import LevMarLSQFitter
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import r2_score
import warnings
import astropy.modeling.fitting
from scipy.interpolate import make_lsq_spline, BSpline
from astropy.modeling import Fittable1DModel, Parameter


class Cauchy1D(Fittable1DModel):
    """
    This is a Cauchy distribution model for fitting the object trace.

    The model is defined as:
    amplitude / (1 + ((x - mean) / gamma) ** 2), where gamma is the FWHM/2.
    """

    # the default values here are set to 0 as they are not used ("dummies")
    amplitude = Parameter(default=0)
    mean = Parameter(default=0)
    gamma = Parameter(default=0)

    @staticmethod
    def evaluate(x, amplitude, mean, gamma):
        return amplitude / (1 + ((x - mean) / gamma) ** 2)


def choose_obj_centrum_obj_trace(file_list):
    """
    A wrapper for `choose_obj_centrum` that is used in the object-finding routine.

    Parameters
    ----------
    file_list : list
        A list of filenames.

    Returns
    -------
    center_dict : dict
        A dictionary containing the user clicked object centers.
        Format: {filename: (x, y)}
    """

    from pylongslit.utils import choose_obj_centrum

    # used for more readable plotting code
    plot_title = (
        lambda file: f"Object position estimation for {file}.\n"
        "Press on the object on a bright point."
        "\nYou can try several times. Press 'q' or close plot when done."
        "\nIf you want to skip this file, simply do not choose a center before closing."
    )

    titles = [plot_title(file) for file in file_list]

    return choose_obj_centrum(file_list, titles)


def estimate_signal_to_noise(data, fitted_amplitude, sky_left, sky_right):
    """
    A very simple and approximate signal to noise estimation.

    Takes the mean of the absolute values of the sky data as the noise.

    Takes the fitted amplitude of the object Gaussian fit as the signal.

    Parameters
    ----------
    data : array
        The data array.

    fitted_amplitude : float
        The amplitude of the fitted Gaussian.

    Returns
    -------
    float
        The signal to noise ratio.
    """

    sky_data = np.concatenate((data[:sky_left], data[sky_right:]))
    noise = np.median(np.abs(sky_data))

    return fitted_amplitude / noise


def find_obj_one_column(obj_x, obj_val, refined_center, params):
    """
    Performs a Gaussian fit to a single column of the detector image to
    estimate the object center and FWHM.

    Parameters
    ----------
    obj_x : array
        The spatial pixel array.

    obj_val : array
        The data array.

    refined_center : float
        The estimated object center.

    params : dict
        A dictionary containing the fit parameters - this is the dictionary
        that is fetched directly from the configuration file (['trace']['object']
        or ['trace']['standard'].

    Returns
    -------
    g_fit : 'astropy.modeling.models.Gaussian1D'
        The fitted Gaussian model object.

    good_fit : bool
        A boolean value indicating whether the fit was good or not.

    R2 : float
        The R2 score of the fit.
    """

    from pylongslit.logger import logger

    # estimate the amplitude guess for initial guess for the Gaussian fitter
    amplitude_guess = np.max(obj_val)

    # construct tuples of min_max values for the fitters
    # The amplitude should not deviate from max value by much, and not be negative
    amplitude_interval = (0.1, 1.1 * amplitude_guess)
    # allow the center to vary by a user defined parameter
    mean_interval = (
        refined_center - params["center_thresh"],
        refined_center + params["center_thresh"],
    )

    # construct the fitter after user choice
    if params["model"] == "Gaussian":
        stddev_interval = (
            (params["fwhm_guess"] - params["fwhm_thresh"]) * gaussian_fwhm_to_sigma,
            (params["fwhm_guess"] + params["fwhm_thresh"]) * gaussian_fwhm_to_sigma,
        )
        # stddev should not be zero or negative
        if stddev_interval[0] <= 0:
            stddev_interval = (0.1, stddev_interval[1])
        fit_model = Gaussian1D(
            amplitude=amplitude_guess,
            mean=refined_center,
            stddev=params["fwhm_guess"] / 2,
            bounds={
                "amplitude": amplitude_interval,
                "mean": mean_interval,
                "stddev": stddev_interval,
            },
        )

    elif params["model"] == "Cauchy":
        # for Cauchy, the gamma is the FWHM/2
        gamma_interval = (
            (params["fwhm_guess"] - params["fwhm_thresh"]) / 2,
            (params["fwhm_guess"] + params["fwhm_thresh"]) / 2,
        )
        # gamma should not be zero or negative
        if gamma_interval[0] <= 0:
            gamma_interval = (0.1, gamma_interval[1])
        fit_model = Cauchy1D(
            amplitude=amplitude_guess,
            mean=refined_center,
            gamma=params["fwhm_guess"] / 2,
            bounds={
                "amplitude": amplitude_interval,
                "mean": mean_interval,
                "gamma": gamma_interval,
            },
        )

    else:

        logger.error(
            "Unrecognized model for object trace - options are 'Gaussian' or 'Cauchy'."
        )
        logger.error("Please check the configuration file.")
        exit()

    fitter = LevMarLSQFitter()

    # perform the fit
    # Suppress warnings during fitting - we handle these ourselves
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: the below exception is not a robust good fix - refactor
        try:
            g_fit = fitter(fit_model, obj_x, obj_val)
        except (TypeError, ValueError, astropy.modeling.fitting.NonFiniteValueError):
            return False

    # user defined R2 is used as a threshold for the fit
    R2 = r2_score(obj_val, g_fit(obj_x))

    if params["fit_R2"] > R2:
        good_fit = False
    else:
        good_fit = True

    return g_fit, good_fit, R2


def find_obj_position(
    spectral, signal_to_noise_array, snr_threshold, minimum_connected_pixels=10
):
    """
    NOT USED

    Estimes the object start and end by searching for a connected region of pixels
    with a signal to noise ratio above a threshold. Used for as a first guess
    for the interactive adjustment of the object limits in the method
    `interactive_adjust_obj_limits`.

    Parameters
    ----------
    spectral : array
        The spectral pixel array.

    signal_to_noise_array : array
        An array containing the signal to noise ratio for each pixel.

    snr_threshold : float
        The signal to noise threshold.

    minimum_connected_pixels : int
        The minimum number of connected pixels above the threshold.
        Default is 10.

    Returns
    -------
    start_pixel : int
        The estimated spectral pixel where the object starts.

    end_pixel : int
        The estimated spectral pixel where the object ends.
    """

    start_index = None
    consecutive_count = 0

    # loop through the signal to noise array and find the start index
    # from where the next 10 pixels have a signal to noise ratio above the threshold
    for i, snr in enumerate(signal_to_noise_array):
        if snr > snr_threshold:
            if start_index is None:
                start_index = i
            consecutive_count += 1
        else:
            start_index = None
            consecutive_count = 0

        if consecutive_count >= minimum_connected_pixels:
            break

    # now the same "backwards"
    end_index = None
    consecutive_count = 0

    for i in range(len(signal_to_noise_array) - 1, -1, -1):
        snr = signal_to_noise_array[i]
        if snr > snr_threshold:
            if end_index is None:
                end_index = i
            consecutive_count += 1
        else:
            end_index = None
            consecutive_count = 0

        if consecutive_count >= minimum_connected_pixels:
            break

    # convert snr array indexes to spectral indexes
    start_pixel = spectral[start_index]
    end_pixel = spectral[end_index]

    return start_pixel, end_pixel


def interactive_adjust_obj_limits(
    image,
    spectral,
    center_data,
    good_fit_array,
    params,
    figsize=(10, 6),
):
    """
    A interactive method that allows the user to adjust the object limits.

    Used for refining the object limits after the initial guess from the method
    `find_obj_position` before fittitng the object trace.

    Parameters
    ----------
    image : array
        The detector image.

    spectral : array
        The spectral pixel array.

    center_data : array
        The estimated object centers.

    good_fit_array : array
        An array containing boolean values for each pixel,
        indicating whether the fit for center was good or not.

    params : dict
        A dictionary containing the fit parameters - this is the dictionary
        that is fetched directly from the configuration file (['trace']['object']
        or ['trace']['standard'].

    figsize : tuple
        The figure size. Default is (10, 6).

    Returns
    -------
    start_index : int
        The spectral pixel where the object starts.

    end_index : int
        The spectral pixel where the object ends.
    """

    from pylongslit.utils import hist_normalize
    from pylongslit.parser import developer_params

    # start at the edges of the object
    start_index = 1
    end_index = image.shape[1] - 1

    # normalize detector image data for plotting
    image = hist_normalize(image)

    # create masked arrays for good and bad center fits
    good_center_data = np.ma.array(center_data, mask=~good_fit_array)
    x_good = np.ma.array(spectral, mask=~good_fit_array)

    bad_center_data = np.ma.array(center_data, mask=good_fit_array)
    x_bad = np.ma.array(spectral, mask=good_fit_array)

    # plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # this method updates the plot - it is called in the key press event
    # in matplotlib (see below).
    def update_plot(start_index, end_index):
        ax1.clear()
        ax1.plot(
            x_good,
            good_center_data,
            "x",
            label="Estimated object center - good fits",
            color="green",
        )
        ax1.plot(
            x_bad,
            bad_center_data,
            "x",
            label="Estimated object center - bad fits",
            color="red",
        )
        ax1.axvline(x=start_index, color="red", linestyle="--", label="Object start")
        ax1.axvline(x=end_index, color="red", linestyle="--", label="Object end")

        ax1.set_title(
            "Adjust object limits with arrowkeys to exclude parts that are too noisy, if needed.\n"
            "Red points are not used, even when theyin the selected interval.\n"
            "Press up/down to change lower boundary, right/left to change upper boundary.\n"
            f"Press 'q' or close window when done."
        )
        ax1.legend()
        ax1.set_ylabel("Spacial pixel")

        # plot the object image for visual referrence
        ax2.clear()
        ax2.imshow(image, cmap="gray", label="Detector Image")
        ax2.scatter(
            x_good,
            good_center_data,
            marker="X",
            color="green",
            label="Estimated object center (accepted)",
            alpha=0.5,
        )
        ax2.scatter(
            x_bad,
            bad_center_data,
            marker="X",
            color="red",
            label="Estimated object center (rejected)",
            alpha=0.5,
        )
        ax2.axvline(x=start_index, color="red", linestyle="--", label="Object start")
        ax2.axvline(x=end_index, color="red", linestyle="--", label="Object end")
        ax2.set_xlabel("Spectral pixel")
        ax2.set_ylabel("Spatial pixel")
        ax2.legend()
        # we invert since python by default has (0,0) in the upper left corner
        ax2.invert_yaxis()
        ax2.set_title(
            "Detector image with estimated object center - use zoom tool for better view."
        )

        # setting the x-axis to be shared between the two plots
        ax1.set_xlim(ax2.get_xlim())
        ax1.set_xticks([])

        fig.canvas.draw()

    # we move by the cut size, as this corresponds to one point in data
    pixel_cut = params["spectral_pixel_extension"]

    # we attachh this to the interactive plot, it calls update_plot every time
    # a key is pressed
    def on_key(event):
        # this allows accesing the SNR_threshold variable outside the scope
        nonlocal start_index
        nonlocal end_index
        if event.key == "up":
            if start_index < end_index:
                start_index += 1 + 2 * pixel_cut
        elif event.key == "down":
            if start_index > 1 + 2 * pixel_cut:
                start_index -= 1 + 2 * pixel_cut
        elif event.key == "right":
            if end_index < image.shape[1] - (1 + 2 * pixel_cut):
                end_index += 1 + 2 * pixel_cut
        elif event.key == "left":
            if end_index > start_index:
                end_index -= 1 + 2 * pixel_cut
        update_plot(start_index, end_index)

    # call the update_plot method to plot the initial state
    update_plot(start_index, end_index)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.legend()
    plt.show()

    if developer_params["verbose_print"]:
        print(
            f"Object start spectral pixel: {start_index}, object end spectral pixel: {end_index}"
        )

    return start_index, end_index


def fit_distribution_parameter(
    use_bsplie, parameter, params, good_x, good_y, data, filename
):
    """
    This is a helper method to 'find_obj_frame' that fits either the object centers or FWHMs
    through the whole detector image.

    Parameters
    ----------
    use_bsplie : bool
        A boolean value indicating whether to use B-spline fitting or not.

    parameter : "Object Center" or "FWHM"
        A string indicating whether to fit the object centers or FWHMs.
        Only "Object Center" or "FWHM" are allowed.

    params : dict
        A dictionary containing the fit parameters - this is the dictionary
        that is fetched directly from the configuration file (['trace']['object']
        or ['trace']['standard'].

    good_x : array
        The spectral pixels used for the fit.

    good_y : array
        The object centers or FWHMs corresponding to good_x.

    data : array
        The detector image.

    filename : str
        The filename of the observation. Used for QA purposes.

    Returns
    -------
    spectral_pixels : array
        The spectral pixel array for whole detector.

    param_fit_pix : array
        The fitted object centers or
        FWHMs through the whole detector image.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import show_1d_fit_QA
    from pylongslit.parser import developer_params

    # sanity check on the fit parameter variable
    if parameter not in ["Object Center", "FWHM"]:
        logger.error(
            "Unrecognized parameter for fitting in fit_distribution_parameter."
        )
        logger.error(
            "Please check the code, or contact the developers if you have not changed the code."
        )
        exit()

    fit_degree = (
        params["fit_order_trace"]
        if parameter == "Object Center"
        else params["fit_order_fwhm"]
    )

    # if the user chooses to use the bspline. This is not recommended for the object trace
    # and should only be used if the trace can not be fitted with a regular polynomial.
    if use_bsplie:
        logger.warning(f"B-spline fitting is selected for {parameter} tracing.")
        logger.warning(
            f"This should only be used when the {parameter} can not be fitted with a regular polynomial."
        )
        logger.warning(
            "This won't work well if there are many unfitted object centers."
        )
        logger.warning("Watch out for overfitting.")

        n_knots = params["knots_bspline"]

        # Create the knots array
        t = np.concatenate(
            (
                np.repeat(good_x[0], fit_degree + 1),  # k+1 knots at the beginning
                np.linspace(good_x[1], good_x[-2], n_knots),  # interior knots
                np.repeat(good_x[-1], fit_degree + 1),  # k+1 knots at the end
            )
        )
        # fit and construct the spline
        spl = make_lsq_spline(good_x, good_y, t=t, k=fit_degree)
        bspline = BSpline(spl.t, spl.c, spl.k)

        show_1d_fit_QA(
            good_x[1:-1],
            good_y[1:-1],
            x_fit_values=good_x[1:-1],
            y_fit_values=bspline(good_x[1:-1]),
            residuals=good_y[1:-1] - bspline(good_x[1:-1]),
            x_label="Spectral Pixel",
            y_label=f"{parameter} in pixels",
            legend_label=f"Individually fitted {parameter}'s",
            title=f"{parameter} B-spline fit for {filename} with {n_knots} interior knots, degree {fit_degree} (this is set in the configuration file).\n"
            "You should aim for very little to no large-scale structure in the residuals, "
            "with the lowest amount of knots possible.",
        )

        # now evaluate the trace through whole detector
        # for the bspline fits we need to extrapolate beyond the fit, so the
        # containers are filled up in several steps
        spectral_pixels = np.arange(data.shape[1])
        param_fit_pix = np.zeros_like(spectral_pixels, dtype=float)

        good_spline_start = int(good_x[1]) + 1
        good_spline_end = int(good_x[-2]) - 1

        param_fit_pix[good_spline_start:good_spline_end] = bspline(
            spectral_pixels[good_spline_start:good_spline_end]
        )

        # we take the first and last 10 points of good data points and extrapolate the ends
        # with a line since the B-spline is ususally diverging at the ends
        start_fit = chebfit(good_x[:10], good_y[:10], deg=1)
        end_fit = chebfit(good_x[-11:-1], good_y[-11:-1], deg=1)

        # insert the linear extrapolations
        param_fit_pix[:good_spline_start] = chebval(
            spectral_pixels[:good_spline_start], start_fit
        )
        param_fit_pix[good_spline_end:] = chebval(
            spectral_pixels[good_spline_end:], end_fit
        )

        if developer_params["debug_plots"]:
            plt.axvline(
                x=good_spline_start,
                color="blue",
                linestyle="--",
                label="Spline/Extrapolation",
            )
            plt.axvline(
                x=good_spline_end,
                color="blue",
                linestyle="--",
                label="Spline/Extrapolation",
            )
            plt.plot(spectral_pixels, param_fit_pix)
            plt.show()

    # regular Chebyshev fitting - the default appraoch
    else:

        param_fit = chebfit(good_x, good_y, deg=fit_degree)

        # dummy x array for plotting the fit
        x_fit = np.linspace(good_x[0], good_x[-1], 1000)

        # evaluate the fit at the used pixel interval
        param_fit_val = chebval(x_fit, param_fit)

        # evaluate the fit at used pixels for residuals
        param_fit_pts = chebval(good_x, param_fit)

        # residuals
        resid = good_y - param_fit_pts

        # now evaluate the trace through whole detector and return
        spectral_pixels = np.arange(data.shape[1], dtype=float)
        param_fit_pix = chebval(spectral_pixels, param_fit)

        show_1d_fit_QA(
            good_x,
            good_y,
            x_fit_values=x_fit,
            y_fit_values=param_fit_val,
            residuals=resid,
            x_label="Spectral Pixel",
            y_label="Counts (ADU)",
            legend_label=f"Individually fitted {parameter}'s",
            title=f"{parameter} fitting QA for {filename}.\n Ensure that the residuals are random and the fit is generally correct."
            f"\nIf not, adjust the fit parameters in the config file. Current fit order is {fit_degree}.",
        )

    return spectral_pixels, param_fit_pix


def objmodel_QA(
    data, params, centers_fit_pix, fwhm_fit_pix, filename, figsize=(10, 6)
):
    """
    Construct the object model and overlay the raw data for QA purposes.

    Parameters
    ----------
    data : array
        The detector image.

    params : dict
        A dictionary containing the fit parameters - this is the dictionary
        that is fetched directly from the configuration file (['trace']['object']
        or ['trace']['standard'].

    centers_fit_pix : array
        The fitted object centers.

    fwhm_fit_pix : array
        The fitted FWHMs.s

    filename : str
        The filename of the observation.

    figsize : tuple
        The figure size. Default is (10, 6).
    """

    from pylongslit.parser import developer_params

    obj_model = np.zeros_like(data)
    x_spat = np.arange(data.shape[0])
    # loop through the spectral pixels and construct the model
    # we set amplitude to 1 for simplicity
    for i in range(data.shape[1]):
        obj_model[:, i] = (
            Cauchy1D().evaluate(x_spat, 1, centers_fit_pix[i], fwhm_fit_pix[i] / 2)
            if params["model"] == "Cauchy"
            else Gaussian1D().evaluate(
                x_spat, 1, centers_fit_pix[i], fwhm_fit_pix[i] * gaussian_fwhm_to_sigma
            )
        )

        if developer_params["debug_plots"] and i % 100 == 0:
            print(f"Model at column {i} constructed.")
            print(f"Center: {centers_fit_pix[i]}, FWHM: {fwhm_fit_pix[i]}")
            plt.plot(x_spat, obj_model[:, i], label=f"Model at column {i}")
            plt.show()

    # plot the QA for the object model
    fig, ax = plt.subplots(3, 1, figsize=figsize)
    ax1, ax2, ax3 = ax

    ax1.imshow(data, cmap="cool", label="Detector image")
    ax1.set_title("Detector image.")

    ax2.imshow(obj_model, cmap="hot", label="Object model")
    ax2.set_title("Object model.")

    ax3.imshow(data, cmap="cool")
    ax3.imshow(obj_model, cmap="hot", alpha=0.3)
    ax3.set_title("Object model overlayed on detector image.")

    fig.text(0.5, 0.04, "Spectral pixels", ha="center", va="center", fontsize=12)
    fig.text(
        0.04,
        0.5,
        "Spacial pixels",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    fig.suptitle(
        f"QA for the object model in {filename}. The object model is constructed from the fitted object centers and FWHMs.\n"
        f"The object model should be a good representation of the object in the detector image. \n"
        f"It is okay if it exceeds the extent of the object in spectral direction.\n"
        f"If the model is not a good representation, revise the object trace routine - pay attention to warnings and QA plots."
    )

    plt.show()


def find_obj_frame(filename, spacial_center, params, figsize=(10, 6)):
    """
    Driver method for finding an object in a single frame.

    First, uses `find_obj_one_column` to find the object in
    columns of the detector image.

    Then, uses `interactive_adjust_obj_limits` to interactively adjust the object limits.

    Finally, fits a Chebyshev polynomial to the object centers and FWHMs,
    and shows QA for the results.

    Parameters
    ----------
    filename : str
        The filename of the observation.

    spacial_center : float
        The user-guess for the object center.

    params : dict
        A dictionary containing the fit parameters - this is the dictionary
        that is fetched directly from the configuration file (['trace']['object']
        or ['trace']['standard'].

    figsize : tuple
        The QA figure size. Default is (10, 6).

    Returns
    -------
    spectral_pixels : array
        The spectral pixel array.

    centers_fit_pix : array
        The fitted object centers.

    fwhm_fit_pix : array
        The fitted FWHMs.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import developer_params
    from pylongslit.utils import PyLongslit_frame, check_crr_and_sky
    from pylongslit.utils import estimate_sky_regions

    logger.info(f"Starting object tracing on {filename}...")

    # get the frame data
    frame = PyLongslit_frame.read_from_disc(filename)

    check_crr_and_sky(frame.header, filename)

    data = frame.data

    # final containers for the results
    good_fit_array = np.array([], dtype=bool)
    spectral = np.array([])
    centers = np.array([])
    FWHMs = np.array([])

    # instead of looping through every spectral pixel, the user can take a mean
    # of a slice +/- the cut extension value to get more robust results.
    cut_extension = params["spectral_pixel_extension"]
    # check that the cut is not some large value that would corrupt the quality
    if cut_extension > len(data[1]) // 20:
        logger.warning(f"The spectral pixel extension is set to {cut_extension}")
        logger.warning(
            "This is a large value and may corrupt the quality of the object trace."
        )
        logger.warning("Consider adjusting the value in the config file.")

    # plot QA for 9 equally spaced columns
    fig, ax = plt.subplots(3, 3, figsize=figsize)
    # for plot indexing
    plot_nr = 0

    num_it = 0
    # TODO hacked way to count the number of iterations, same as in tilt fits in wavecalid
    # consider a utils method for this
    for i in range(cut_extension, data.shape[1], cut_extension + 1):
        num_it += 1

    # find 9 equally spaced indices to which plot QA fits, cut the edges a bit
    # , handle the unlicely case where the number of columns is less than 100
    if data.shape[1] < 100:
        qa_indices = np.linspace(0, data.shape[1], 9).astype(int)
    else:
        qa_indices = np.linspace(50, data.shape[1] - 50, 9).astype(int)

    if developer_params["verbose_print"]:
        print(qa_indices)

    # loop through the columns and find obj in each
    logger.info(f"Finding object in {filename}...")

    # these are the detector pixel values on the spacial axis
    x_spat = np.arange(data.shape[0])

    # loop through the columns - the tqdm is a progress bar
    for i in tqdm(
        range(cut_extension, data.shape[1], cut_extension + 1),
        desc=f"Fitting object trace for {filename}",
    ):

        if cut_extension == 0:
            val = data[:, i]
        # take a  mean of the slice if there is a user defined cut extension
        else:
            val = np.mean(data[:, i - cut_extension : i + cut_extension + 1], axis=1)

        # get the area only around the object
        refined_center, sky_left, sky_right = estimate_sky_regions(
            val, spacial_center, params["fwhm_guess"], params["fwhm_thresh"]
        )

        obj_x = x_spat[sky_left:sky_right]
        obj_val = val[sky_left:sky_right]

        # check if the object is in the frame
        if len(obj_x) == 0 or len(obj_val) == 0:
            continue

        # perform the Gaussian fit
        try:
            g_fit, good_fit, R2 = find_obj_one_column(
                obj_x, obj_val, refined_center, params
            )
        # TODO: make this more robust
        except TypeError:
            continue

        # extract the fit values used from the fitter:
        center = g_fit.mean.value
        # FWHM depends on the model used
        FWHM = (
            g_fit.gamma.value * 2
            if params["model"] == "Cauchy"
            else g_fit.stddev.value * gaussian_sigma_to_fwhm
        )

        # check if the column is one of the QA columns. The complicated if statement
        # checks if the column is within the interval of +/- cut extension from the QA indices
        if (
            any(
            (i - qa_index) in np.arange(0, cut_extension + 1)
            for qa_index in qa_indices
            )
            and plot_nr < 9
        ):

            if developer_params["verbose_print"]:
                print(f"Plotting QA for column {i}...")

            obj_x_linspace = np.linspace(obj_x[0], obj_x[-1], 1000)

            # plot the fit
            row = plot_nr // 3
            col = plot_nr % 3
            ax[row, col].plot(obj_x, obj_val, ".-", label=f"Data at column {i}", markersize=4)
            ax[row, col].plot(
            obj_x_linspace,
            g_fit(obj_x_linspace),
            label=f"Fit with R2:{R2:.2f}\n FWHM {FWHM:.2f}",
            c="red" if not good_fit else "green",
            linewidth=0.8,
            )
            ax[row, col].axvline(
            center,
            color="green",
            linestyle="--",
            label=f"Fitted center {center:.2f}",
            linewidth=0.8,
            )
            ax[row, col].legend(fontsize=8)
            ax[row, col].grid(True, linewidth=0.5)
            ax[row, col].tick_params(axis="both", which="major", labelsize=8)
            plot_nr += 1

        # append the results to the containers
        spectral = np.append(spectral, i)
        centers = np.append(centers, center)
        FWHMs = np.append(FWHMs, FWHM)
        good_fit_array = np.append(good_fit_array, good_fit)

    fig.suptitle(
        f"QA for a sample of object center fits in {filename}. Green - accepted, red - rejected. Model\n"
        f"Fit model chosen: {params['model']}. Current rejection threshold is R2 > {params['fit_R2']},\n"
        f"with allowed center deviation of {params['center_thresh']} pixels away from max data value,\n"
        f"and {params['fwhm_thresh']} pixels from FWHM guess. Initial FWHM guess is {params['fwhm_guess']}. All of these can be changed in the configuration file.",
        fontsize=12,
    )
    fig.text(0.5, 0.04, "Spacial Pixels", ha="center", va="center", fontsize=12)
    fig.text(
        0.04,
        0.5,
        "Counts",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    plt.show()

    # center estimation done, now interactive adjustment of object limits

    logger.info("Starting interactive user refinement of object limits...")
    logger.info("Follow the instructions in the plot.")
    # interactive user refinment of object limits
    obj_start_pixel, obj_end_pixel = interactive_adjust_obj_limits(
        data, spectral, centers, good_fit_array, params
    )

    # the indexes are not on the same grid as the detector array,
    # so we have to convert to array indexes
    obj_start_index = np.argmin(np.abs(spectral - obj_start_pixel))
    obj_end_index = np.argmin(np.abs(spectral - obj_end_pixel))

    # now the object and fwhm fitting

    # for centers and FWHMs, mask everything below obj_start_index and above obj_end_index,
    # and only keep the good center fits
    good_fits_cropped = good_fit_array[obj_start_index:obj_end_index]

    good_x = spectral[obj_start_index:obj_end_index][good_fits_cropped]
    good_centers = centers[obj_start_index:obj_end_index][good_fits_cropped]
    good_FWHMs = FWHMs[obj_start_index:obj_end_index][good_fits_cropped]

    # the actual fits are performed in the helper method
    logger.info("Fitting object center...")

    spectral_pixels, centers_fit_pix = fit_distribution_parameter(
        params["use_bspline_obj"],
        "Object Center",
        params,
        good_x,
        good_centers,
        data,
        filename,
    )

    logger.info("Fitting object FWHM...")

    spectral_pixels, fwhm_fit_pix = fit_distribution_parameter(
        params["use_bspline_fwhm"], "FWHM", params, good_x, good_FWHMs, data, filename
    )

    # lastly, construct the object model for QA

    objmodel_QA(data, params, centers_fit_pix, fwhm_fit_pix, filename, figsize=figsize)

    return spectral_pixels, centers_fit_pix, fwhm_fit_pix


def get_params(filename):
    """
    Checks the filename and returns the correct parameters for the object trace,
    depending on whether the frame is a standard star or a science frame.

    Parameters
    ----------
    filename : str
        The filename of the observation.

    Returns
    -------
    params : dict
        A dictionary containing the fit parameters - this is the dictionary
        that is fetched directly from the configuration file (['trace']['object']
        or ['trace']['standard']).
    """

    from pylongslit.logger import logger
    from pylongslit.parser import trace_params

    # sanity check for the filename
    if "science" not in filename and "standard" not in filename:
        logger.error(f"Unrecognized file type for {filename}.")
        logger.error("Make sure not to manually rename any files.")
        logger.error(
            "Restart from the reduction procedure. Contact the developers if the problem persists."
        )
        exit()
    # the parameters are different for standard and object frames,
    # as the apertures usually differ in size
    if "standard" in filename:
        params = trace_params["standard"]
        logger.info("This is a standard star frame.")
    else:
        params = trace_params["object"]
        logger.info("This is a science frame.")

    return params


def find_obj(center_dict):
    """
    Driver method for object finding in every frame.

    Loops through the frames and calls `find_obj_frame` for every frame.

    Parameters
    ----------
    center_dict : dict
        A dictionary containing the user clicked object centers.
        Format: {filename: (x, y)}

    Returns
    -------
    obj_dict : dict
        A dictionary containing the results of the object finding routine.
        Format: {filename: (spectral_pixel, object_center, fwhm)}
    """

    from pylongslit.logger import logger

    # this is the container for the results
    obj_dict = {}

    # loop through the files
    for filename, center in center_dict.items():

        params = get_params(filename)

        # we only need the spatial center
        spacial_center = center[1]
        spectral, centers_fit_val, fwhm_fit_val = find_obj_frame(
            filename, spacial_center, params
        )
        obj_dict[filename] = (spectral, centers_fit_val, fwhm_fit_val)

        logger.info(f"Object tracing routine finished for {filename}.")
        print("----------------------------\n")

    logger.info("Object tracing routine finished for all frames.")

    return obj_dict


def write_obj_trace_results(obj_dict):
    """
    Writes the object trace results to a file in the output directory.

    Parameters
    ----------
    obj_dict : dict
        A dictionary containing the results of the object finding routine.
        Format: {filename: (good_x, centers_fit_val, fwhm_fit_val)}
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    for filename, (good_x, centers_fit_val, fwhm_fit_val) in obj_dict.items():

        # prepare a filename
        filename = filename.replace("reduced_", "obj_").replace(".fits", ".dat")

        logger.info(f"Writing object trace results to {filename}...")

        # get current working directory
        cwd = os.getcwd()

        # change to output directory
        os.chdir(output_dir)

        # write to the file
        with open(filename, "w") as f:
            for x, center, fwhm in zip(good_x, centers_fit_val, fwhm_fit_val):
                f.write(f"{x}\t{center}\t{fwhm}\n")

        # close the file
        f.close()

        # change back to the original directory
        # useful if the user is using relative paths in the config file
        os.chdir(cwd)

        logger.info(
            f"Object trace results written to directory {output_dir}, filename: {filename}."
        )


def run_obj_trace():
    """
    Driver method for the object tracing routine.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import get_reduced_frames

    logger.info("Starting object tracing routine...")
    logger.info("Fetching reduced frames...")
    filenames = get_reduced_frames()

    # get the user-guess for the object center
    center_dict = choose_obj_centrum_obj_trace(filenames)

    obj_dict = find_obj(center_dict)

    logger.info("Writing object trace results...")

    write_obj_trace_results(obj_dict)

    logger.info("Object tracing routine finished.")
    print("----------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit object-tracing procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_obj_trace()


if __name__ == "__main__":
    main()
