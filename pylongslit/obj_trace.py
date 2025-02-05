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


def find_obj_one_column(x, val, spacial_center, FWHM_AP, column_index):
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

    # get the area only around the object
    refined_center, sky_left, sky_right = estimate_sky_regions(
        val, spacial_center, FWHM_AP
    )

    obj_x = x[sky_left:sky_right]
    obj_val = val[sky_left:sky_right]

    # this is neeeded for the fitting process
    amplitude_guess = np.max(obj_val)

    # construct tuples of min_max values for the Gaussian fitter

    # The amplitude should not deviate from max value by much
    amplitude_interval = (0, 1.1 * amplitude_guess)
    # allow the mean to vary by FWHM
    mean_interval = (refined_center - FWHM_AP, refined_center + FWHM_AP)
    # allow the stddev to start at 0.1 pixel and vary by 2 FWHM
    stddev_interval = (
        0.1 * gaussian_fwhm_to_sigma,
        2 * FWHM_AP * gaussian_fwhm_to_sigma,
    )

    # build a Gaussian fitter
    g_init = Gaussian1D(
        amplitude=amplitude_guess,
        mean=refined_center,
        stddev=FWHM_AP * gaussian_fwhm_to_sigma,
        bounds={
            "amplitude": amplitude_interval,
            "mean": mean_interval,
            "stddev": stddev_interval,
        },
    )

    # perform the fit
    fitter = LevMarLSQFitter()
    g_fit = fitter(g_init, obj_x, obj_val)

    # extract the fitted peak position and FWHM:
    amplitude = g_fit.amplitude.value
    fit_center = g_fit.mean.value
    fitted_stddev = g_fit.stddev.value
    fitted_FWHM = fitted_stddev * gaussian_sigma_to_fwhm

    # If the fit has reached one of the bounds, the fit is likely not good,
    # and we should not trust the results.

    good_fit = True

    if (
        amplitude <= amplitude_interval[0]
        or amplitude >= amplitude_interval[1]
        or fit_center <= mean_interval[0]
        or fit_center >= mean_interval[1]
        or fitted_stddev <= stddev_interval[0]
        or fitted_stddev >= stddev_interval[1]
    ):
        good_fit = False

    # estimate the signal to noise ratio for later QA

    signal_to_noise = estimate_signal_to_noise(val, amplitude, sky_left, sky_right)

    return fit_center, fitted_FWHM, signal_to_noise, good_fit


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


def find_obj_frame(filename, spacial_center, FWHM_AP):
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

    FWHM_AP : float
        The user-guess for the FWHM of the object.

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
    from pylongslit.parser import extract_params, output_dir
    from pylongslit.utils import open_fits

    # get initial guess for SNR threshold
    SNR_initial_guess = extract_params["SNR"]
    # get polynomial degree for fitting
    fit_deg = extract_params["OBJ_FIT_DEG"]

    # open the file
    hdul = open_fits(output_dir, filename)
    data = hdul[0].data

    header = hdul[0].header
    # get the cropped y offset for global detector coordinates
    y_lower = header["CROPY1"]
    y_upper = header["CROPY2"]

    print("Got cropped values: ", y_lower, y_upper)

    # final containers for the results
    centers = []
    FWHMs = []
    signal_to_noise_array = []
    # this is used for distinguishing good and bad fits in the plots
    good_fit_array = []

    # loop through the columns and find obj in each
    logger.info(f"Finding object in {filename}...")
    x_spat = np.arange(data.shape[0])
    for i in tqdm(range(data.shape[1]), desc=f"Fitting object trace for {filename}"):
        val = data[:, i]

        try:

            center, FWHM, signal_to_noise, good_fit = find_obj_one_column(
                x_spat, val, spacial_center, FWHM_AP, i
            )

        except ValueError:
            # if the fit fails, add NaNs to the results
            center = np.nan
            FWHM = np.nan
            signal_to_noise = np.nan
            good_fit = False

        centers.append(center)
        FWHMs.append(FWHM)
        signal_to_noise_array.append(signal_to_noise)
        good_fit_array.append(good_fit)

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
    from pylongslit.parser import extract_params

    # extract the user-guess for the FWHM of the object
    FWHM_AP = extract_params["FWHM_AP"]

    # this is the container for the results
    obj_dict = {}

    # loop through the files
    for filename, center in center_dict.items():
        logger.info(f"Finding object in {filename}...")
        # we only need the spatial center
        spacial_center = center[1]
        good_x, centers_fit_val, fwhm_fit_val = find_obj_frame(
            filename, spacial_center, FWHM_AP
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
        filename = filename.replace("skysub_", "obj_").replace(".fits", ".dat")

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
    from pylongslit.utils import get_skysub_files

    logger.info("Starting object tracing routine...")

    filenames = get_skysub_files()

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
