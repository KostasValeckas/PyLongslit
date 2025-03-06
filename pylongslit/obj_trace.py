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
        "Press on the object on a bright point"
        "\nYou can try several times. Press 'q' or close plot when done."
    )

    titles = [plot_title(file) for file in file_list]

    return choose_obj_centrum(file_list, titles)


def estimate_signal_to_noise(data, fitted_amplitude, sky_left, sky_right):
    """
    A very simple and approximate signal to noise estimation.

    Takes the mean of the absolute values of the data sky as the noise.

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


def find_obj_one_column(x, val, spacial_center, fwhm_guess, R2_threshold):
    """
    Perform a Gaussian fit to a single column of the detector image to
    estimate the object center and FWHM.

    Parameters
    ----------
    x : array
        The x values.

    val : array
        The data values.

    spacial_center : float
        The user-guess for the object center.

    FWHM_AP : float
        The user-guess for the FWHM of the object.

    column_index : int
        The index of the column (this is the spectral pixel).

    Returns
    -------
    fit_center : float
        The fitted object center.

    fitted_FWHM : float
        The fitted FWHM of the object.

    signal_to_noise : float
        The signal to noise ratio.

    good_fit : bool
        Whether the fit was successful or not.
    """
    from pylongslit.utils import estimate_sky_regions
    from pylongslit.parser import developer_params
    from astropy.modeling.models import Const1D
    from pylongslit.wavecalib import GeneralizedNormal1D

    
    # we don't want negative values in the data, as they might come from the 
    # A-B subtraction. We invert the data if it is negative, to keep the noise
    # properties the same.

    val = np.abs(val)

    # get the area only around the object
    refined_center, sky_left, sky_right = estimate_sky_regions(
        val, spacial_center, fwhm_guess
    )

    obj_x = x[sky_left:sky_right]
    obj_val = val[sky_left:sky_right]

    # this is neeeded for the fitting process
    amplitude_guess = np.max(obj_val)

    # construct tuples of min_max values for the Gaussian fitter

    # The amplitude should not deviate from max value by much, and not be negative
    amplitude_interval = (0.1, 1.1 * amplitude_guess)
    # allow the mean to vary by FWHM
    mean_interval = (refined_center - fwhm_guess, refined_center + fwhm_guess) 
    # allow the stddev to start at 0.1 pixel and vary by 2 FWHM
    stddev_interval = (
        0.1 * gaussian_fwhm_to_sigma,
        2 * fwhm_guess * gaussian_fwhm_to_sigma,
    )

    beta_init = 1

    # build a Gaussian fitter
    g_init = GeneralizedNormal1D(
        amplitude=amplitude_guess,
        mean=refined_center,
        stddev=fwhm_guess * gaussian_fwhm_to_sigma,
        beta=beta_init,
        bounds={
            "amplitude": amplitude_interval,
            "mean": mean_interval,
            "stddev": stddev_interval,
            # beta > 2 flattens peak, beta > 20 is almost a step function
            "beta": (0.1, 20),
        },
    )

    # a constant model to add to the Gaussian - sometimes needed
    # if a continuum is present in the line spectrum
    const = Const1D(amplitude=0)
    g_model = g_init + const

    # perform the fit
    fitter = LevMarLSQFitter()
    g_fit = fitter(g_model, obj_x, obj_val)

    R2 = r2_score(obj_val, g_fit(obj_x))

    if R2_threshold > R2:
        good_fit = False
    else:
        good_fit = True

    # extract the amplitude to estimate the signal to noise ratio
    amplitude = g_fit.amplitude_0.value

    
    # estimate the signal to noise ratio for later QA
    signal_to_noise = estimate_signal_to_noise(val, amplitude, sky_left, sky_right)

    return fitter, g_fit, signal_to_noise, good_fit


def find_obj_position(
    signal_to_noise_array, snr_threshold, minimum_connected_pixels=10
):
    """
    Estimes the object start and end by searching for a connected region of pixels
    with a signal to noise ratio above the threshold.

    Parameters
    ----------
    signal_to_noise_array : array
        An array containing the signal to noise ratio for each pixel.

    snr_threshold : float
        The signal to noise threshold.

    minimum_connected_pixels : int
        The minimum number of connected pixels above the threshold.
        Default is 10.

    Returns
    -------
    start_index : int
        The index where the object starts.

    end_index : int
        The index where the object ends.
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

    end_index = None
    consecutive_count = 0

    # loop through the signal to noise array and find the end index
    # from where the previous 10 pixels have a signal to noise ratio above the threshold
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

    return start_index, end_index


def interactive_adjust_obj_limits(
    image,
    center_data,
    signal_to_noise_array,
    SNR_initial_guess,
    good_fit_array,
    figsize=(18, 12),
):
    """
    A interactive method that allows the user to adjust the object limits
    by adjusting the signal to noise threshold and visually inspecting the results.

    Parameters
    ----------
    image : array
        The detector image.

    center_data : array
        The estimated object centers.

    signal_to_noise_array : array
        The signal to noise ratio for each pixel.

    SNR_initial_guess : float
        The initial guess for the signal to noise threshold.

    good_fit_array : array
        An array containing boolean values for each pixel,
        indicating whether the fit was good or not.

    figsize : tuple
        The figure size. Default is (18, 12).

    Returns
    -------
    start_index : int
        The index where the object starts.

    end_index : int
        The index where the object ends.
    """

    from pylongslit.utils import hist_normalize

    SNR_threshold = SNR_initial_guess

    # estimate the start and end index of the object
    start_index, end_index = find_obj_position(signal_to_noise_array, SNR_threshold)

    # normalize detector image data for plotting
    image = hist_normalize(image)

    # spectral pixel array for plotting
    x = np.arange(len(center_data))

    # create masked arrays for good and bad center fits
    good_center_data = np.ma.array(center_data, mask=~good_fit_array)
    x_good = np.ma.array(x, mask=~good_fit_array)

    bad_center_data = np.ma.array(center_data, mask=good_fit_array)
    x_bad = np.ma.array(x, mask=good_fit_array)

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
            "Adjust SNR threshold with arrowkeys to exclude parts that are too noisy.\n"
            "Press up/down arrows for small changes (+/- 0.1), right/left for large changes (+/- 1).\n"
            f"Current SNR threshold: {SNR_threshold}. Press 'q' or close window when done."
        )
        ax1.legend()
        ax1.set_ylabel("Spacial pixel")

        # plot the object image for visual referrence
        ax2.clear()
        ax2.imshow(image, cmap="gray", label="Detector Image")
        ax2.scatter(
            np.arange(len(center_data)),
            center_data,
            marker=".",
            color="red",
            label="Estimated object center (faint red line)",
            s=0.7,
            alpha=0.2,
        )
        ax2.axvline(x=start_index, color="red", linestyle="--", label="Object start")
        ax2.axvline(x=end_index, color="red", linestyle="--", label="Object end")
        ax2.set_xlabel("Spectral pixel")
        ax2.set_ylabel("Spatial pixel")
        ax2.legend()
        ax2.invert_yaxis()

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
        start_index, end_index = find_obj_position(signal_to_noise_array, SNR_threshold)
        update_plot(start_index, end_index)

    # call the update_plot method to plot the initial state
    update_plot(start_index, end_index)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.legend()
    plt.show()

    print(f"Object start index: {start_index}, object end index: {end_index}")

    return start_index, end_index


def show_obj_trace_QA(
    good_x,
    x_fit,
    good_centers,
    center_fit_values,
    good_FWHMs,
    FWHM_fit_values,
    resid_centers,
    resid_FWHMs,
    filename,
):
    """
    A wrapper for `utils.show_1d_fit_QA` that is used in the object-finding routine.
    Plots QA for fitting object center and FWHM.

    Parameters
    ----------
    good_x : array
        The spectral pixel array.

    x_fit : array
        The spectral pixel array for the fit.

    good_centers : array
        The estimated object centers.

    center_fit_values : array
        The fitted object centers.

    good_FWHMs : array
        The estimated FWHMs.

    FWHM_fit_values : array
        The fitted FWHMs.

    resid_centers : array
        The residuals for the object centers.

    resid_FWHMs : array
        The residuals for the FWHMs.

    filename : str
        The filename of the observation.
    """
    from pylongslit.utils import show_1d_fit_QA

    # plot the result of center finding for QA
    show_1d_fit_QA(
        good_x,
        good_centers,
        x_fit_values=x_fit,
        y_fit_values=center_fit_values,
        residuals=resid_centers,
        x_label="Spectral pixel",
        y_label="Spatial pixel",
        legend_label="Fitted centers for every detector column",
        title=f"Center finding QA for {filename}.\n Ensure the fit is good and residuals are random."
        "\nIf not, adjust the fit parameters in the config file.",
    )

    # plot the result of FWHM finding for QA
    show_1d_fit_QA(
        good_x,
        good_FWHMs,
        x_fit_values=x_fit,
        y_fit_values=FWHM_fit_values,
        residuals=resid_FWHMs,
        x_label="Spectral pixel",
        y_label="Spatial pixels",
        legend_label="Fitted FWHMs for every detector column",
        title=f"FWHM finding QA for {filename}.\n Ensure the fit is good and residuals are random."
        "\nIf not, adjust the fit parameters in the config file.",
    )


def find_obj_frame(filename, spacial_center, params):
    """
    Driver method for finding an object in a single frame.

    First, uses `find_obj_one_column` to find the object in every
    column of the detector image.

    Then, uses `interactive_adjust_obj_limits` to interactively adjust the object limits.

    Finally, fits a Chebyshev polynomial to the object centers and FWHMs,
    and shows QA for the results.

    Parameters
    ----------
    filename : str
        The filename of the observation.

    spacial_center : float
        The user-guess for the object center.

    Returns
    -------
    good_x : array
        The spectral pixel array.

    centers_fit_pix : array
        The fitted object centers.

    fwhm_fit_pix : array
        The fitted FWHMs.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import developer_params
    from pylongslit.utils import PyLongslit_frame

    # get initial guess for SNR threshold
    SNR_initial_guess = params["SNR"]
    # get polynomial degree for fitting
    fit_deg = params["fit_order"]

    frame = PyLongslit_frame.read_from_disc(filename)
    data = frame.data

    header = frame.header
    # get the cropped y offset for global detector coordinates
    # this is needed for matching the wavesolution in extraction
    y_lower = header["CROPY1"]
    y_upper = header["CROPY2"]

    if developer_params["verbose_print"]:
        print("Got cropped values: ", y_lower, y_upper)

    # final containers for the results
    used_spectral = []
    centers = []
    FWHMs = []
    signal_to_noise_array = []
    # this is used for distinguishing good and bad fits in the plots
    good_fit_array = []

    # instead of looping through every spectral pixel, the user can take a mean
    # of a slice +/- the cut extension value to get more robust results.  
    cut_extension = params["spectral_pixel_extension"]
    #check that the cut is not some large value that would corrupt the quality
    if cut_extension > len(data[1])//20:
        logger.warning(f"The spectral pixel extension is set to {cut_extension}")
        logger.warning("This is a large value and may corrupt the quality of the object trace.")
        logger.warning("Consider adjusting the value in the config file.")
    """
    # find the indices for the spectral pixel range
    # TODO: a bit hacked but works - same as in wavecalib tilt tracing
    # consider a utils method for this
    index_dict = {}
    num_it = 0
    for i in range(cut_extension, data.shape[1], cut_extension + 1):
        index_dict[num_it] = i
        num_it += 1
    """

    fig, ax = plt.subplots(3, 3, figsize=(18, 12))
    # for plot indexing
    plot_nr = 0

    num_it = 0
    # TODO hacked way to count the number of iterations, same as in tilt fits in wavecalid
    # consider a utils method for this
    for i in range(cut_extension, data.shape[1], cut_extension + 1): num_it += 1

    # find 9 equally spaced indices to which plot QA fits, cut the edges a bit
    qa_indices = np.linspace(50, data.shape[1]-50, 9).astype(int)
    print(qa_indices)


    # loop through the columns and find obj in each
    logger.info(f"Finding object in {filename}...")
    x_spat = np.arange(data.shape[0])
    for i in tqdm(range(cut_extension, data.shape[1], cut_extension + 1), desc=f"Fitting object trace for {filename}"):
        
        if cut_extension == 0:
            val = data[:, i]
        # take a  mean of the slice if there is a user defined cut extension
        else:
            val = np.mean(data[:, i-cut_extension:i+cut_extension+1], axis=1)

   
        fitter, g_fit, signal_to_noise, good_fit = find_obj_one_column(
                x_spat, val, spacial_center, params["fwhm_guess"], params["fit_R2"]
            )
        
        # extract the fit values:

        center = g_fit.mean_0.value
        FWHM = g_fit.stddev_0.value * gaussian_sigma_to_fwhm


        if any(np.abs(i - qa_index) in np.arange(0, cut_extension) for qa_index in qa_indices) and plot_nr < 9:

            print(f"Plotting QA for column {i}...") 

            x_spat_linspace = np.linspace(x_spat[0], x_spat[-1], 1000)

            # plot the fit
            row = plot_nr // 3
            col = plot_nr % 3
            ax[row, col].plot(x_spat, val, ".-", label="Data")
            ax[row, col].plot(x_spat_linspace, g_fit(x_spat_linspace), label="Fit", c = "red" if not good_fit else "green") 
            ax[row, col].axvline(center, color="green", linestyle="--", label="Fitted center")
            ax[row, col].set_title(f"Column {i}, fitted FWHM: {FWHM:.2f}, FWHM guess: {params['fwhm_guess']:.2f}")
            ax[row, col].legend()

            plot_nr += 1





        """
        centers.append(center)
        FWHMs.append(FWHM)
        signal_to_noise_array.append(signal_to_noise)
        good_fit_array.append(good_fit)
        """

    
    plt.show()
    exit()

    # ensure good_fit_array is a python array, since it is used for masking
    good_fit_array = np.array(good_fit_array)

    logger.info("Starting interactive user refinement of object limits...")
    logger.info("Follow the instructions in the plot.")
    # interactive user refinment of object limits
    obj_start_index, obj_end_index = interactive_adjust_obj_limits(
        data, centers, signal_to_noise_array, SNR_initial_guess, good_fit_array
    )

    # make a dummy x_array for fitting
    good_x = np.arange(obj_start_index, obj_end_index)

    # for centers and FWHMs, mask everythong below obj_start_index and above obj_end_index

    good_centers = np.array(centers)[obj_start_index:obj_end_index]
    good_FWHMs = np.array(FWHMs)[obj_start_index:obj_end_index]

    # add the offset from the crop procedure
    # good_centers += y_lower

    logger.info("Fitting object centers and FWHMs...")

    centers_fit = chebfit(good_x, good_centers, deg=fit_deg)
    fwhm_fit = chebfit(good_x, good_FWHMs, deg=fit_deg)

    # dummy x array for plotting the fit
    x_fit = np.linspace(good_x[0], good_x[-1], 1000)

    centers_fit_val = chebval(x_fit, centers_fit)
    fwhm_fit_val = chebval(x_fit, fwhm_fit)

    # evaluate the fit at every pixel
    centers_fit_pix = chebval(good_x, centers_fit)
    fwhm_fit_pix = chebval(good_x, fwhm_fit)

    # residuals
    resid_centers = good_centers - centers_fit_pix
    resid_FWHMs = good_FWHMs - fwhm_fit_pix

    # now evaluate the trace through whole detector and return
    spectral_pixels = np.arange(data.shape[1])
    centers_fit_pix = chebval(spectral_pixels, centers_fit)
    fwhm_fit_pix = chebval(spectral_pixels, fwhm_fit)

    # show QA
    show_obj_trace_QA(
        good_x,
        x_fit,
        good_centers,
        centers_fit_val,
        good_FWHMs,
        fwhm_fit_val,
        resid_centers,
        resid_FWHMs,
        filename,
    )

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
        Format: {filename: (good_x, centers_fit_val, fwhm_fit_val)}
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

        if "standard" in filename:
            params = trace_params["standard"]
            logger.info("This is a standard star frame.")
        else: 
            params = trace_params["object"]
            logger.info("This is a science frame.")

        logger.info(f"Finding object in {filename}...")
        # we only need the spatial center
        spacial_center = center[1]
        good_x, centers_fit_val, fwhm_fit_val = find_obj_frame(
            filename, spacial_center, params
        )
        obj_dict[filename] = (good_x, centers_fit_val, fwhm_fit_val)

    return obj_dict


def write_obj_trace_results(obj_dict):
    """
    Writes the object trace results to a file.

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

        # change back to the working directory
        os.chdir("..")

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

    filenames = get_reduced_frames()

    # get the user-guess for the object center
    center_dict = choose_obj_centrum_obj_trace(filenames)

    obj_dict = find_obj(center_dict)

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
