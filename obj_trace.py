from logger import logger
from astropy.io import fits
from astropy.modeling.models import Gaussian1D
from parser import extract_params, output_dir
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from utils import get_file_group, open_fits, write_to_fits, choose_obj_centrum
import matplotlib.pyplot as plt
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Const1D
from utils import refine_obj_center
import matplotlib.pyplot as plt
from utils import hist_normalize


class GeneralizedNormal1D(Fittable1DModel):
    """
    This is a generalized normal distribution model for
    fitting the lines in the arc spectrum - it works like a Gaussian
    but has a shape parameter beta that controls the flatness of the peak.
    """

    amplitude = Parameter(default=1)
    mean = Parameter(default=0)
    stddev = Parameter(default=1)
    beta = Parameter(default=5)  # Shape parameter

    @staticmethod
    def evaluate(x, amplitude, mean, stddev, beta):
        return amplitude * np.exp(-((np.abs(x - mean) / stddev) ** beta))


def get_skysub_files():
    """
    Wrapper for ´get_file_group´ that returns the filenames of the skysubtracted,
    and performs some sanity checks.

    Returns
    -------
    filenames : list
        A list of filenames of the skysubtracted files.
    """

    logger.info("Getting skysubtracted files...")

    filenames = get_file_group("skysub")

    if len(filenames) == 0:
        logger.error("No skysubtracted files found.")
        logger.error("Make sure you run the sky-subraction routine first.")
        exit()

    return filenames


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

    # used for more readable plotting code
    plot_title = (
        lambda file: f"Object position estimation for {file}.\n"
        "Press on the object on a bright point"
        "\nYou can try several times. Press 'q' or close plot when done."
    )

    titles = [plot_title(file) for file in file_list]

    return choose_obj_centrum(file_list, titles)


def estimate_signal_to_noise(data, fitted_amplitude):
    """ """

    noise = np.mean(np.abs(data))

    return fitted_amplitude / noise


def find_obj_one_column(x, val, spacial_center, FWHM_AP):

    amplitude_guess = np.max(val)

    # build a Generalized Normal fitter with an added constant
    g_init = GeneralizedNormal1D(
        amplitude=amplitude_guess,
        mean=spacial_center,
        stddev=FWHM_AP * gaussian_fwhm_to_sigma,
        beta=2,  # Initial guess for beta
        bounds={
            # allow the amplitude to vary by 2 times the guess
            "amplitude": (0, 1.1 * amplitude_guess),
            # allow the mean to vary by 3 FWHM
            "mean": (spacial_center - 3 * FWHM_AP, spacial_center + 3 * FWHM_AP),
            # allow the stddev to vary by 2 FWHM
            "stddev": (gaussian_fwhm_to_sigma, 4 * FWHM_AP * gaussian_fwhm_to_sigma),
            "beta": (2, 20),
        },
    )

    # const = Const1D(amplitude=np.mean(val))
    g_model = g_init  # + const

    # perform the fit
    fitter = LevMarLSQFitter()
    g_fit = fitter(g_model, x, val)

    # print(g_fit)

    # plot the results
    # plt.figure()
    # plt.plot(x, val, label="Data")
    # plt.plot(x, g_fit(x), label="Fit")
    # plt.axhline(y=g_fit.amplitude.value, color="red", linestyle="--", label="Fitted amplitude")
    # plt.show()

    # extract the fitted peak position and FWHM:
    fit_center = g_fit.mean.value
    fitted_FWHM = g_fit.stddev.value * gaussian_sigma_to_fwhm
    amplitude = g_fit.amplitude.value

    signal_to_noise = estimate_signal_to_noise(val, amplitude)

    # plot QA
    # plt.plot(x, val, label="Data")
    # plt.plot(x, g_fit(x), label="Fit")
    # plt.show()

    return fit_center, fitted_FWHM, signal_to_noise


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
    image, center_data, signal_to_noise_array, SNR_initial_guess, figsize=(18, 12)
):
    # TODO: this is laggy and slow, optimize

    SNR_threshold = SNR_initial_guess

    # find the start and end index of the object
    start_index, end_index = find_obj_position(signal_to_noise_array, SNR_threshold)

    # normalize image data for plotting
    image = hist_normalize(image)

    # plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    def update_plot(start_index, end_index):
        ax1.clear()
        ax1.plot(center_data, label="Estimated object center")
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

    def on_key(event):
        nonlocal SNR_threshold
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

    update_plot(start_index, end_index)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.legend()
    plt.show()

    return start_index, end_index


def find_obj_frame(filename, spacial_center, FWHM_AP):

    # get initial guess for SNR
    SNR_initial_guess = extract_params["SNR"]

    # open the file
    hdul = open_fits(output_dir, filename)
    data = hdul[0].data

    # final containers for the results
    centers = []
    FWHMs = []
    signal_to_noise_array = []

    # loop through the columns and find obj in each
    for i in range(data.shape[1]):
        x = np.arange(data.shape[0])
        val = data[:, i]

        center, FWHM, signal_to_noise = find_obj_one_column(
            x, val, spacial_center, FWHM_AP
        )

        centers.append(center)
        FWHMs.append(FWHM)
        signal_to_noise_array.append(signal_to_noise)

    # interactive user refinment of object limits
    obj_start_index, obj_end_index = interactive_adjust_obj_limits(
        data, centers, signal_to_noise_array, SNR_initial_guess
    )


def refine_obj_centers(center_dict, FWHM_AP):
    # TODO: move this to utils and use in sky subtraction
    """
    Refines the object centers.

    Driver for the `refine_obj_center` function.

    Parameters
    ----------
    center_dict : dict
        A dictionary containing the object centers.
        Format: {filename: (x, y)}

    Returns
    -------
    refined_centers : dict
        A dictionary containing the refined object centers.
        Format: {filename: (x, y)}
    """

    logger.info("Refining object centers...")

    refined_centers = {}

    for filename, center in center_dict.items():
        logger.info(f"Refining object center for {filename}...")

        # open the file
        hdul = open_fits(output_dir, filename)
        data = hdul[0].data

        # get slice at where user defined the point
        slice = data[:, center[0]]
        x_slice = np.arange(slice.shape[0])

        refined_spat_center = refine_obj_center(x_slice, slice, center[1], FWHM_AP)

        refined_centers[filename] = (center[0], refined_spat_center)

    logger.info("Refinement done.")
    print(refined_centers)
    print("------------------------------------")

    return refined_centers


def find_obj(center_dict):

    # extract the user-guess for the FWHM of the object
    FWHM_AP = extract_params["FWHM_AP"]

    # loop through the files
    for filename, center in center_dict.items():
        logger.info(f"Finding object in {filename}...")
        # we only need the spatial center
        spacial_center = center[1]
        find_obj_frame(filename, spacial_center, FWHM_AP)


def run_obj_trace():
    logger.info("Starting object tracing routine...")

    filenames = get_skysub_files()

    # get the user-guess for the object center
    center_dict = choose_obj_centrum_obj_trace(filenames)

    refined_centers = refine_obj_centers(center_dict, extract_params["FWHM_AP"])

    find_obj(refined_centers)


if __name__ == "__main__":
    run_obj_trace()
