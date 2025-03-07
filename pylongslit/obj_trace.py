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
from astropy.modeling.models import Const1D
import warnings
import astropy.modeling.fitting
from scipy.interpolate import make_lsq_spline, BSpline



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

    from pylongslit.wavecalib import GeneralizedNormal1D

    # estimate the amplitude guess for initial guess for the Gaussian fitter
    amplitude_guess = np.max(obj_val)

    # construct tuples of min_max values for the Gaussian fitter

    # The amplitude should not deviate from max value by much, and not be negative
    amplitude_interval = (0.1, 1.1 * amplitude_guess)
    # allow the center to vary by a user defined parameter
    mean_interval = (refined_center - params["center_thresh"], refined_center + params["center_thresh"]) 
    # allow the stddev vary by a user defined parameter
    gamma_interval = (
        (params["fwhm_guess"] - params["fwhm_thresh"]) * gaussian_fwhm_to_sigma,
        (params["fwhm_guess"] + params["fwhm_thresh"]) * gaussian_fwhm_to_sigma,
    )
    beta_interval = (2, 2)
    
    # construct the fitter 
    # TODO: this has some repeated code from wavecalib - consider a utils method
    g_init = GeneralizedNormal1D(
        amplitude=amplitude_guess,
        mean=refined_center,
        stddev=params["fwhm_guess"] * gaussian_fwhm_to_sigma,
        beta=2,
        bounds={
            "amplitude": amplitude_interval,
            "mean": mean_interval,
            "stddev": stddev_interval,
            "beta": beta_interval,
        },
    )

    # a constant model to add to the Gaussian - sometimes needed
    # if sky is not subtracted properly - as this adds a continuum
    const = Const1D(amplitude=0)
    g_model = g_init + const
    fitter = LevMarLSQFitter()

    # perform the fit
    # Suppress warnings during fitting - we handle these ourselves
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #TODO: the below exception is not a robust good fix - refactor
        try:
            g_fit = fitter(g_model, obj_x, obj_val)
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
    signal_to_noise_array,
    SNR_initial_guess,
    good_fit_array,
    figsize=(18, 12),
):
    """
    A interactive method that allows the user to adjust the object limits
    by adjusting the signal to noise threshold and visually inspecting the results.

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

    signal_to_noise_array : array
        The signal to noise ratio for each center.

    SNR_initial_guess : float
        The initial guess for the signal to noise threshold.

    good_fit_array : array
        An array containing boolean values for each pixel,
        indicating whether the fit for center was good or not.

    figsize : tuple
        The figure size. Default is (18, 12).

    Returns
    -------
    start_index : int
        The spectral pixel where the object starts.

    end_index : int
        The spectral pixel where the object ends.
    """

    from pylongslit.utils import hist_normalize
    from pylongslit.parser import developer_params

    # renaming for clarity in code later on
    SNR_threshold = SNR_initial_guess

    # estimate the start and end index of the object
    start_index, end_index = find_obj_position(spectral, signal_to_noise_array, SNR_threshold)

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
            "Adjust SNR threshold with arrowkeys to exclude parts that are too noisy, if needed.\n"
            "Press up/down arrows for small changes (+/- 0.1), right/left for large changes (+/- 1).\n"
            f"Current SNR threshold: {SNR_threshold}. Press 'q' or close window when done."
        )
        ax1.legend()
        ax1.set_ylabel("Spacial pixel")

        # plot the object image for visual referrence
        ax2.clear()
        ax2.imshow(image, cmap="gray", label="Detector Image")
        ax2.scatter(
            spectral,
            center_data,
            marker="X",
            color="red",
            label="Estimated object center",
            #s=0.7,
            alpha=0.5,
        )
        ax2.axvline(x=start_index, color="red", linestyle="--", label="Object start")
        ax2.axvline(x=end_index, color="red", linestyle="--", label="Object end")
        ax2.set_xlabel("Spectral pixel")
        ax2.set_ylabel("Spatial pixel")
        ax2.legend()
        # we invert since python by default has (0,0) in the upper left corner
        ax2.invert_yaxis()
        ax2.set_title("Detector image with estimated object center - use zoom tool for better view.")

        # setting the x-axis to be shared between the two plots
        ax1.set_xlim(ax2.get_xlim())
        ax1.set_xticks([])

        fig.canvas.draw()

    # we attachh this to the interactive plot, it calls update_plot every time
    # a key is pressed
    def on_key(event):
        # this allows accesing the SNR_threshold variable outside the scope
        nonlocal SNR_threshold
        nonlocal start_index
        nonlocal end_index
        if event.key == "up":
            SNR_threshold += 0.1
        elif event.key == "down":
            SNR_threshold -= 0.1
        elif event.key == "right":
            SNR_threshold += 1
        elif event.key == "left":
            SNR_threshold -= 1
        start_index, end_index = find_obj_position(spectral, signal_to_noise_array, SNR_threshold)
        update_plot(start_index, end_index)

    # call the update_plot method to plot the initial state
    update_plot(start_index, end_index)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.legend()
    plt.show()

    if developer_params["verbose_print"]:
        print(f"Object start spectral pixel: {start_index}, object end spectral pixel: {end_index}")

    return start_index, end_index



def find_obj_frame(filename, spacial_center, params, figsize=(18, 18)):
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
        The QA figure size. Default is (18, 18).

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
    from pylongslit.utils import PyLongslit_frame
    from pylongslit.utils import estimate_sky_regions
    from pylongslit.utils import show_1d_fit_QA, hist_normalize
    from pylongslit.wavecalib import GeneralizedNormal1D

    logger.info(f"Starting object tracing on {filename}...")

    # get the frame data
    frame = PyLongslit_frame.read_from_disc(filename)
    # check if frame has been skysubtracted and crr removed, else throw
    # a warning
    if not frame.header["CRRREMOVD"]:
        logger.warning(f"{filename} has not been cosmic ray removed.")
        logger.warning("This may affect the quality of the object trace.")
        logger.warning("Consider running the cosmic ray removal routine - but continuing for now...")

    if not frame.header["SKYSUBBED"] and not frame.header["BCGSUBBED"]:
        logger.warning(f"{filename} has not been sky subtracted.")
        logger.warning("This may affect the quality of the object trace.")
        logger.warning("Consider running the sky subtraction or A-B background subtraction routines - but continuing for now...")

    data = frame.data

    # final containers for the results
    good_fit_array = np.array([], dtype=bool)
    spectral = np.array([])
    centers = np.array([])
    FWHMs = np.array([])
    signal_to_noise_array = np.array([])
    # beta params (this decides the shape of the fitted distribution)
    betas = np.array([])

    # instead of looping through every spectral pixel, the user can take a mean
    # of a slice +/- the cut extension value to get more robust results.  
    cut_extension = params["spectral_pixel_extension"]
    #check that the cut is not some large value that would corrupt the quality
    if cut_extension > len(data[1])//20:
        logger.warning(f"The spectral pixel extension is set to {cut_extension}")
        logger.warning("This is a large value and may corrupt the quality of the object trace.")
        logger.warning("Consider adjusting the value in the config file.")

    # plot QA for 9 equally spaced columns
    fig, ax = plt.subplots(3, 3, figsize=figsize)
    # for plot indexing
    plot_nr = 0

    num_it = 0
    # TODO hacked way to count the number of iterations, same as in tilt fits in wavecalid
    # consider a utils method for this
    for i in range(cut_extension, data.shape[1], cut_extension + 1): num_it += 1

    # find 9 equally spaced indices to which plot QA fits, cut the edges a bit
    # handle the unlicely case where the number of columns is less than 100
    if data.shape[1] < 100:
        qa_indices = np.linspace(0, data.shape[1], 9).astype(int)
    else:
        qa_indices = np.linspace(50, data.shape[1]-50, 9).astype(int) 
    
    if developer_params["verbose_print"]: print(qa_indices)

    # loop through the columns and find obj in each
    logger.info(f"Finding object in {filename}...")

    # these are the detector pixel values on the spacial axis
    x_spat = np.arange(data.shape[0])

    # loop through the columns - the tqdm is a progress bar
    for i in tqdm(range(cut_extension, data.shape[1], cut_extension + 1), desc=f"Fitting object trace for {filename}"):
        
        if cut_extension == 0:
            val = data[:, i]
        # take a  mean of the slice if there is a user defined cut extension
        else:
            val = np.mean(data[:, i-cut_extension:i+cut_extension+1], axis=1)

        # get the area only around the object
        refined_center, sky_left, sky_right = estimate_sky_regions(
            val, spacial_center, params["fwhm_guess"]
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

        center = g_fit.mean_0.value
        FWHM = g_fit.stddev_0.value * gaussian_sigma_to_fwhm

        # extract the amplitude to estimate the signal to noise ratio
        amplitude = g_fit.amplitude_0.value

        # this is the beta parameter that decides the shape of the fitted distribution
        beta = g_fit.beta_0.value

            
        # estimate the signal to noise ratio for later interactive adjustment
        signal_to_noise = estimate_signal_to_noise(val, amplitude, sky_left, sky_right)

        # check if the column is one of the QA columns. The complicated if statement
        # checks if the column is within the cut_extension pixels of the QA index
        if any((i - qa_index) in np.arange(0, cut_extension+1) for qa_index in qa_indices) and plot_nr < 9:

            if developer_params["verbose_print"]: print(f"Plotting QA for column {i}...") 

            obj_x_linspace = np.linspace(obj_x[0], obj_x[-1], 1000)

            # plot the fit
            row = plot_nr // 3
            col = plot_nr % 3
            ax[row, col].plot(obj_x, obj_val, ".-", label=f"Data at column {i}")
            ax[row, col].plot(obj_x_linspace, g_fit(obj_x_linspace), label=f"Fit with R2:{R2:.2f}\n FWHM {FWHM:.2f}", c = "red" if not good_fit else "green")
            ax[row, col].axvline(center, color="green", linestyle="--", label=f"Fitted center {center:.2f}")
            ax[row, col].legend()
            ax[row, col].grid(True)
            plot_nr += 1

        # append the results to the containers
        spectral = np.append(spectral, i)
        centers = np.append(centers, center)
        FWHMs = np.append(FWHMs, FWHM)
        signal_to_noise_array = np.append(signal_to_noise_array, signal_to_noise)
        good_fit_array = np.append(good_fit_array, good_fit)
        betas = np.append(betas, beta)
    
    fig.suptitle(
        f"QA for a sample of object center fits in {filename}. Green - accepted, red - rejected.\n"
        f"Current rejection threshold is R2 > {params['fit_R2']},\n" 
        f"with allowed center deviation of {params['center_thresh']} pixels away from max data value,\n"  
        f"and {params['fwhm_thresh']} pixels from FWHM guess. Initial FWHM guess is {params['fwhm_guess']}. All of these can be changed in the configuration file."
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

    # get initial guess for SNR threshold
    SNR_initial_guess = params["SNR"]


    logger.info("Starting interactive user refinement of object limits...")
    logger.info("Follow the instructions in the plot.")
    # interactive user refinment of object limits
    obj_start_pixel, obj_end_pixel = interactive_adjust_obj_limits(
        data, spectral, centers, signal_to_noise_array, SNR_initial_guess, good_fit_array
    )

    # the indexes are not on the same grid as the detector array, 
    # so we have to convert to array indexes
    obj_start_index = np.argmin(np.abs(spectral - obj_start_pixel))
    obj_end_index = np.argmin(np.abs(spectral - obj_end_pixel))

    # now the object and fwhm fitting

    # get polynomial degree for fitting
    fit_deg_trace = params["fit_order_trace"]
    fit_deg_fwhm = params["fit_order_fwhm"]
    fit_order_beta = params["fit_order_beta"]
    

    # for centers and FWHMs, mask everything below obj_start_index and above obj_end_index,
    # and only keep the good center fits
    good_fits_cropped = good_fit_array[obj_start_index:obj_end_index]

    good_x = spectral[obj_start_index:obj_end_index][good_fits_cropped]
    good_centers = centers[obj_start_index:obj_end_index][good_fits_cropped]
    good_FWHMs = FWHMs[obj_start_index:obj_end_index][good_fits_cropped]
    good_betas = betas[obj_start_index:obj_end_index][good_fits_cropped]


    logger.info("Fitting object centers and FWHMs...")

    # if the uses chooses to use the bspline. This is not recommended for the object trace
    # and should only be used if the trace can not be fitted with a regular polynomial.
    if params["use_bspline"]:
        logger.warning("B-spline fitting is selected for object tracing.")
        logger.warning("This should only be used when the trace can not be fitted with a regular polynomial.")
        logger.warning("This won't work well if there are many unfitted object centers.")
        logger.warning("Watch out for overfitting.")

        n_knots = params["knots_bspline"]

        # Create the knots array
        t_trace = np.concatenate(
            (
                np.repeat(good_x[0], fit_deg_trace + 1),  # k+1 knots at the beginning
                np.linspace(
                    good_x[1], good_x[-2], n_knots
                ),  # interior knots
                np.repeat(good_x[-1], fit_deg_trace + 1),  # k+1 knots at the end
            )
        )

        t_fwhm = np.concatenate(
            (
                np.repeat(good_x[0], fit_deg_fwhm + 1),  # k+1 knots at the beginning
                np.linspace(
                    good_x[1], good_x[-2], n_knots
                ),  # interior knots
                np.repeat(good_x[-1], fit_deg_fwhm + 1),  # k+1 knots at the end
            )
        )

        t_beta = np.concatenate(
            (
                np.repeat(good_x[0], fit_order_beta + 1),  # k+1 knots at the beginning
                np.linspace(
                    good_x[1], good_x[-2], n_knots
                ),  # interior knots
                np.repeat(good_x[-1], fit_order_beta + 1),  # k+1 knots at the end
            )
        )   

        spl_trace = make_lsq_spline(good_x, good_centers, t=t_trace, k=fit_deg_trace)
        bspline_trace = BSpline(spl_trace.t, spl_trace.c, spl_trace.k)

        spl_fwhm = make_lsq_spline(good_x, good_FWHMs, t=t_fwhm, k=fit_deg_fwhm)    
        bspline_fwhm = BSpline(spl_fwhm.t, spl_fwhm.c, spl_fwhm.k)

        spl_beta = make_lsq_spline(good_x, good_betas, t=t_beta, k=fit_order_beta)
        bspline_beta = BSpline(spl_beta.t, spl_beta.c, spl_beta.k)

        show_1d_fit_QA(
            good_x[1:-1],
            good_centers[1:-1],
            x_fit_values=good_x[1:-1],
            y_fit_values=bspline_trace(good_x[1:-1]),
            residuals=good_centers[1:-1] - bspline_trace(good_x[1:-1]),
            x_label="Wavelength (Å)",
            y_label="Counts (ADU)",
            legend_label="Extracted flat-field lamp spectrum",
            title=
                f"Object trace B-spline fit with {n_knots} interior knots, degree {fit_deg_trace} (this is set in the configuration file).\n"
                "You should aim for very little to no large-scale structure in the residuals, "
                "with the lowest amount of knots possible.",
        )

        show_1d_fit_QA(
            good_x[1:-1],
            good_FWHMs[1:-1],
            x_fit_values=good_x[1:-1],
            y_fit_values=bspline_fwhm(good_x[1:-1]),
            residuals=good_FWHMs[1:-1] - bspline_fwhm(good_x[1:-1]),
            x_label="Wavelength (Å)",
            y_label="Counts (ADU)",
            legend_label="Extracted flat-field lamp spectrum",
            title=
                f"FWHM B-spline fit with {n_knots} interior knots, degree {fit_deg_fwhm} (this is set in the configuration file).\n"
                "You should aim for very little to no large-scale structure in the residuals, "
                "with the lowest amount of knots possible.",
        )

        show_1d_fit_QA(
            good_x[1:-1],
            good_betas[1:-1],
            x_fit_values=good_x[1:-1],
            y_fit_values=bspline_beta(good_x[1:-1]),
            residuals=good_betas[1:-1] - bspline_beta(good_x[1:-1]),
            x_label="Wavelength (Å)",
            y_label="Counts (ADU)",
            legend_label="Extracted flat-field lamp spectrum",
            title=
                f"Beta (shape parameter for the distribution) B-spline fit with {n_knots} interior knots, degree {fit_order_beta} (this is set in the configuration file).\n"
                "You should aim for very little to no large-scale structure in the residuals, "
                "with the lowest amount of knots possible.",
        )

        # now evaluate the trace through whole detector 

        # for the bspline fits we need to extrapolate beyond the fit, so the
        # containers are filled up in several steps
        spectral_pixels = np.arange(data.shape[1])
        centers_fit_pix = np.zeros_like(spectral_pixels, dtype=float)
        fwhm_fit_pix = np.zeros_like(spectral_pixels, dtype=float)
        betas_fit_pix = np.zeros_like(spectral_pixels, dtype=float)

        good_spline_start = int(good_x[1])+1
        good_spline_end = int(good_x[-2])-1
        
        centers_fit_pix[good_spline_start:good_spline_end] = bspline_trace(spectral_pixels[good_spline_start:good_spline_end])
        fwhm_fit_pix[good_spline_start:good_spline_end] = bspline_fwhm(spectral_pixels[good_spline_start:good_spline_end])
        betas_fit_pix[good_spline_start:good_spline_end] = bspline_beta(spectral_pixels[good_spline_start:good_spline_end])

        # we take the first and last 10 points of good data points and extrapolate the ends 
        # with a line since the B-spline is ususally diverging at the ends
        start_fit_trace = chebfit(good_x[:10], good_centers[:10], deg=1)
        end_fit_trace = chebfit(good_x[-11: -1], good_centers[-11:-1], deg=1)

        start_fit_fwhm = chebfit(good_x[:10], good_FWHMs[:10], deg=1)
        end_fit_fwhm = chebfit(good_x[-11: -1], good_FWHMs[-11:-1], deg=1)

        start_fit_beta = chebfit(good_x[:10], good_betas[:10], deg=1)
        end_fit_beta = chebfit(good_x[-11: -1], good_betas[-11:-1], deg=1)
        
        # insert the linear extrapolations
        centers_fit_pix[:good_spline_start] = chebval(spectral_pixels[:good_spline_start], start_fit_trace)
        centers_fit_pix[good_spline_end:] = chebval(spectral_pixels[good_spline_end:], end_fit_trace)

        fwhm_fit_pix[:good_spline_start] = chebval(spectral_pixels[:good_spline_start], start_fit_fwhm)
        fwhm_fit_pix[good_spline_end:] = chebval(spectral_pixels[good_spline_end:], end_fit_fwhm)

        betas_fit_pix[:good_spline_start] = chebval(spectral_pixels[:good_spline_start], start_fit_beta)
        betas_fit_pix[good_spline_end:] = chebval(spectral_pixels[good_spline_end:], end_fit_beta)

        if developer_params["debug_plots"]:
            plt.axvline(x=good_spline_start, color='blue', linestyle='--', label='Spline/Extrapolation')
            plt.axvline(x=good_spline_end, color='blue', linestyle='--', label='Spline/Extrapolation')
            plt.plot(spectral_pixels, centers_fit_pix)
            plt.show()

    # regular Chebyshev fitting - the default appraoch
    else:


        centers_fit = chebfit(good_x, good_centers, deg=fit_deg_trace)
        fwhm_fit = chebfit(good_x, good_FWHMs, deg=fit_deg_fwhm)
        betas_fit = chebfit(good_x, good_betas, deg=fit_order_beta)

        # dummy x array for plotting the fit
        x_fit = np.linspace(good_x[0], good_x[-1], 1000)

        centers_fit_val = chebval(x_fit, centers_fit)
        fwhm_fit_val = chebval(x_fit, fwhm_fit)
        betas_fit_val = chebval(x_fit, betas_fit)

        # evaluate the fit at used pixels for residuals
        centers_fit_pts = chebval(good_x, centers_fit)
        fwhm_fit_pts = chebval(good_x, fwhm_fit)
        betas_fit_pts = chebval(good_x, betas_fit)

        # residuals
        resid_centers = good_centers - centers_fit_pts
        resid_FWHMs = good_FWHMs - fwhm_fit_pts
        resdid_betas = good_betas - betas_fit_pts

        # now evaluate the trace through whole detector and return
        spectral_pixels = np.arange(data.shape[1], dtype=float)
        centers_fit_pix = chebval(spectral_pixels, centers_fit)
        fwhm_fit_pix = chebval(spectral_pixels, fwhm_fit)
        betas_fit_pix = chebval(spectral_pixels, betas_fit)


        show_1d_fit_QA(
            good_x,
            good_centers,
            x_fit_values=x_fit,
            y_fit_values=centers_fit_val,
            residuals=resid_centers,
            x_label="Spectral pixel",
            y_label="Spatial pixel",
            legend_label="Fitted centers",
            title=f"Object center fitting QA for {filename}.\n Ensure that the residuals are random and the fit is generally correct."
            f"\nIf not, adjust the fit parameters in the config file. Current fit order is {fit_deg_trace}."
        )

        # plot the result of FWHM finding for QA
        show_1d_fit_QA(
            good_x,
            good_FWHMs,
            x_fit_values=x_fit,
            y_fit_values=fwhm_fit_val,
            residuals=resid_FWHMs,
            x_label="Spectral pixel",
            y_label="Spatial pixels",
            legend_label="Fitted FWHMss",
            title=f"FWHM fitting QA for {filename}.\n Ensure that the residuals are random and the fit is generally correct."
            f"\nIf not, adjust the fit parameters in the config file. Current fit order is {fit_deg_fwhm}."
        )

        show_1d_fit_QA(
            good_x,
            good_betas,
            x_fit_values=x_fit,
            y_fit_values=betas_fit_val,
            residuals=resdid_betas,
            x_label="Spectral pixel",
            y_label="Spatial pixels",
            legend_label="Fitted Beta values",
            title=f"Beta (parameter deciding shape of the distribution) fitting QA for {filename}.\n Ensure that the residuals are random and the fit is generally correct."
            f"\nIf not, adjust the fit parameters in the config file. Current fit order is {fit_order_beta}."
        )


        

        if developer_params["debug_plots"]:
            plt.plot(spectral_pixels, centers_fit_pix, label='centers')
            plt.plot(spectral_pixels, fwhm_fit_pix, label='fwhm')
            plt.plot(spectral_pixels, betas_fit_pix, label='betas')
            plt.plot(spectral_pixels, betas_fit_pix)
            plt.legend()
            plt.show()


    # lastly, construct the object model for QA

    obj_model = np.zeros_like(data)
    # loop through the spectral pixels and construct the model
    # we set amplitude to 1 for simplicity
    for i in range(data.shape[1]):
        obj_model[:, i] = GeneralizedNormal1D().evaluate(
            x_spat, 1, centers_fit_pix[i], fwhm_fit_pix[i] * gaussian_fwhm_to_sigma, betas_fit_pix[i]
        )
        if developer_params["debug_plots"] and i % 100 == 0:
            print(f"Model at column {i} constructed.")
            print(f"Center: {centers_fit_pix[i]}, FWHM: {fwhm_fit_pix[i]}, Beta: {betas_fit_pix[i]}")
            plt.plot(x_spat, obj_model[:, i], label=f"Model at column {i}")
            plt.show()

    # plot the QA for the object model
    fig, ax = plt.subplots(3, 1, figsize=(18, 9))
    ax1, ax2, ax3 = ax
    
    ax1.imshow(data, cmap="cool", label="Detector image")

    ax2.imshow(obj_model, cmap="hot", label="Object model")


    ax3.imshow(obj_model, cmap="hot", label="Object model", alpha=0.5)
    ax3.imshow(data, cmap="cool", label="Detector image", alpha=0.5)
    plt.show()



    return spectral_pixels, centers_fit_pix, fwhm_fit_pix


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
    from pylongslit.parser import trace_params

    # this is the container for the results
    obj_dict = {}

    # loop through the files
    for filename, center in center_dict.items():
        # sanity check for the filename
        if "science" not in filename and "standard" not in filename:
            logger.error(f"Unrecognized file type for {filename}.")
            logger.error("Make sure not to manually rename any files.")
            logger.error("Restart from the reduction procedure. Contact the developers if the problem persists.")
            exit()

        # the parameters are different for standard and object frames, 
        # as the apertures usually differ in size
        if "standard" in filename:
            params = trace_params["standard"]
            logger.info("This is a standard star frame.")
        else: 
            params = trace_params["object"]
            logger.info("This is a science frame.")

        logger.info(f"Finding object in {filename}...")
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

        # change to output directory
        os.chdir(output_dir)

        # write to the file
        with open(filename, "w") as f:
            for x, center, fwhm in zip(good_x, centers_fit_val, fwhm_fit_val):
                f.write(f"{x}\t{center}\t{fwhm}\n")

        # close the file
        f.close()

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
    parser = argparse.ArgumentParser(description="Run the pylongslit object-tracing procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_obj_trace()


if __name__ == "__main__":
    main()
