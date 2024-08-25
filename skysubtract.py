from logger import logger
from parser import skip_science_or_standard_bool
from parser import output_dir, extract_params
from utils import list_files, hist_normalize, open_fits, write_to_fits
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from numpy.polynomial.chebyshev import chebfit, chebval


def get_reduced_group(*prefixes):
    """
    Helper method to retrieve the names of the
    reduced frames (science or standard) from the output directory.

    Parameters
    ----------
    prefixes : str
        Prefixes of the files to be retrieved.
        Example: "reduced_science", "reduced_std"

    Returns
    -------
    reduced_files : list
        A list of reduced files.
    """

    file_list = os.listdir(output_dir)

    reduced_files = [file for file in file_list if file.startswith(prefixes)]

    logger.info(f"Found {len(reduced_files)} frames:")
    list_files(reduced_files)

    return reduced_files


def get_reduced_frames():
    """
    Driver for `get_reduced_frames` that acounts for skip_science or/and
    skip_standard parameters.

    Returns
    -------
    reduced_files : list
        A list of the reduced files.
    """
    if skip_science_or_standard_bool == 0:
        logger.error(
            "Both skip_science and skip_standard parameters are set to true "
            "in the configuration file."
        )
        logger.error("No extraction can be performed. Exitting...")
        exit()

    elif skip_science_or_standard_bool == 1:

        logger.warning(
            "Standard star extraction is set to be skipped in the config file."
        )
        logger.warning("Will only extract science spectra.")

        reduced_files = get_reduced_group("reduced_science")

    elif skip_science_or_standard_bool == 2:

        logger.warning("Science extraction is set to be skipped in the config file.")
        logger.warning("Will only extract standard star spectra.")

        reduced_files = get_reduced_group("reduced_std")

    else:

        reduced_files = get_reduced_group("reduced_science", "reduced_std")

    return reduced_files


def choose_obj_centrum(file_list, figsize=(18, 12)):
    """
    An interactive method to choose the center of the object on the frame.

    Parameters
    ----------
    file_list : list
        A list of filenames to be reduced.

    figsize : tuple
        The size of the figure to be displayed.
        Default is (18, 12).

    Returns
    -------
    center_dict : dict
        A dictionary containing the chosen centers of the objects.
    """

    logger.info("Starting object-choosing GUI. Follow the instructions on the plots.")

    # used for more readable plotting code
    def plot_title(file):
        return (
            f"Object position estimation for {file}.\n"
            "Press on the object on a spectral point with no sky-lines "
            "(but away from detector edges.) \n"
            "You can try several times. Press 'q' or close plot when done."
        )

    center_dict = {}

    def onclick(event):
        x = int(event.xdata)
        y = int(event.ydata)

        # put the clicked point in the dictionary
        center_dict[file] = (x, y)

        # Remove any previously clicked points
        plt.cla()
        # the plotting code below is repeated twice, but this is more stable
        # for the event function (it uses non-local variables)
        plt.imshow(norm_data, cmap="gray")
        plt.title(plot_title(file))

        # Color the clicked point
        plt.scatter(x, y, marker="x", color="red", s=50, label="Selected point")
        plt.legend()
        plt.draw()  # Update the plot

    for file in file_list:
        plt.figure(figsize=figsize)

        frame = open_fits(output_dir, file)
        data = frame[0].data
        norm_data = hist_normalize(data)

        plt.imshow(norm_data, cmap="gray")
        plt.connect("button_press_event", onclick)
        plt.title(plot_title(file))
        plt.show()

    logger.info("Object centers chosen successfully:")
    print(center_dict, "\n------------------------------------")

    return center_dict


def refine_obj_center(x, slice, clicked_center, FWHM_AP):
    """
    Refine the object center based on the slice of the data.

    Try a simple numerical estimation of the object center, and check
    if it is within the expected region. If not, use the clicked point.

    Used it in the `trace_sky` method.

    Parameters
    ----------
    x : array
        The x-axis of the slice.

    slice : array
        The slice of the data.

    clicked_center : int
        The center of the object clicked by the user.

    FWHM_AP : int
        The FWHM of the object.

    Returns
    -------
    center : int
        The refined object center.
    """

    logger.info("Refining the object center...")

    center = x[np.argmax(slice)]

    if center < clicked_center - 2 * FWHM_AP or center > clicked_center + 2 * FWHM_AP:
        logger.warning("The estimated object center is outside the expected region.")
        logger.warning("Using the user-clicked point as the center.")
        center = clicked_center

    return center


def estimate_sky_regions(slice_spec, spatial_center_guess, FWHM_AP):

    x_spec = np.arange(len(slice_spec))

    center = refine_obj_center(x_spec, slice_spec, spatial_center_guess, FWHM_AP)

    # QA for sky region selection
    sky_left = center - 2 * FWHM_AP
    sky_right = center + 2 * FWHM_AP

    # create sky value arrays by excludint the object region
    sky_val = np.concatenate((slice_spec[:sky_left], slice_spec[sky_right:]))
    x_sky = np.concatenate((x_spec[:sky_left], x_spec[sky_right:]))

    return x_spec, x_sky, sky_val, sky_left, sky_right


def fit_sky_one_column(
    slice_spec,
    spatial_center_guess,
    FWHM_AP,
    SIGMA_APSKY,
    ITERS_APSKY,
    ORDER_APSKY,
):

    x_spec, x_sky, sky_val, _, _ = estimate_sky_regions(
        slice_spec, spatial_center_guess, FWHM_AP
    )

    # mask the outliers
    clip_mask = sigma_clip(sky_val, sigma=SIGMA_APSKY, maxiters=ITERS_APSKY).mask

    # fit the sky
    coeff_apsky, _ = chebfit(
        x_sky[~clip_mask], sky_val[~clip_mask], deg=ORDER_APSKY, full=True
    )

    # evaluate the fit
    sky_fit = chebval(x_spec, coeff_apsky)

    return sky_fit


def fit_sky_QA(
    slice_spec,
    spatial_center_guess,
    spectral_center,
    NSUM_AP,
    FWHM_AP,
    SIGMA_APSKY,
    ITERS_APSKY,
    ORDER_APSKY,
    figsize=(18, 12),
):
    x_spec, _, _, sky_left, sky_right = estimate_sky_regions(
        slice_spec, spatial_center_guess, FWHM_AP
    )

    sky_fit = fit_sky_one_column(
        slice_spec, spatial_center_guess, FWHM_AP, SIGMA_APSKY, ITERS_APSKY, ORDER_APSKY
    )

    plt.figure(figsize=figsize)
    plt.axvline(x=sky_left, color="r", linestyle="--", label="Object boundary")
    plt.axvline(x=sky_right, color="r", linestyle="--")
    plt.plot(
        x_spec,
        slice_spec,
        label=f"Detector slice around spectral pixel {spectral_center} +/- {NSUM_AP//2}",
    )
    plt.plot(x_spec, sky_fit, label="Sky fit")
    plt.xlabel("Pixels (spatial direction)")
    plt.ylabel("Detector counts (ADU)")
    plt.legend()
    plt.title(
        "Sky-background fitting QA. Ensure the fit is reasonable, and that the object "
        "is completely encapsulated by the red lines.\n"
        "If not, change relative parameters in the config file."
    )
    plt.show()


def make_sky_map(
    data, spatial_center_guess, FWHM_AP, SIGMA_APSKY, ITERS_APSKY, ORDER_APSKY
):
    """ """

    n_spacial = data.shape[0]
    n_spectal = data.shape[1]

    sky_map = np.zeros((n_spacial, n_spectal))

    for column in range(n_spectal):
        slice_spec = data[:, column]
        sky_fit = fit_sky_one_column(
            slice_spec,
            spatial_center_guess,
            FWHM_AP,
            SIGMA_APSKY,
            ITERS_APSKY,
            ORDER_APSKY,
        )
        sky_map[:, column] = sky_fit

    norm_sky_map = hist_normalize(sky_map)
    plt.imshow(norm_sky_map, cmap="gray")
    plt.title("Sky map")
    plt.show()

    return sky_map


def remove_sky_background(center_dict):
    """ 
    """

    # user-defined size of the aperture slicing in extraction
    NSUM_AP = extract_params["NSUM_AP"]
    FWHM_AP = extract_params["FWHM_AP"]
    SIGMA_APSKY = extract_params["SIGMA_APSKY"]
    ITERS_APSKY = extract_params["ITERS_APSKY"]
    ORDER_APSKY = extract_params["ORDER_APSKY"]

    subtracted_frames = {}

    for key in center_dict.keys():

        frame = open_fits(output_dir, key)
        data = frame[0].data

        clicked_point = center_dict[key]

        # start with a QA at user defined point
        spacial_center_guess = clicked_point[1]
        spectral_center_guess = clicked_point[0]

        slice_spec = data[:, spectral_center_guess]

        fit_sky_QA(
            slice_spec,
            spacial_center_guess,
            spectral_center_guess,
            NSUM_AP,
            FWHM_AP,
            SIGMA_APSKY,
            ITERS_APSKY,
            ORDER_APSKY,
        )

        sky_map = make_sky_map(
            data,
            spacial_center_guess,
            FWHM_AP,
            SIGMA_APSKY,
            ITERS_APSKY,
            ORDER_APSKY,
        )

        skysub_data = data - sky_map

        norm_data = hist_normalize(skysub_data)

        plt.imshow(norm_data, cmap="gray")
        plt.title("Sky-subtracted frame")
        plt.show()

        key = key.replace("reduced_", "skysub_")

        subtracted_frames[key] = skysub_data

    return subtracted_frames

def write_sky_subtracted_frames_to_disc(subtracted_frames):

    for key, data in subtracted_frames.items():
        # steal header from the original file
        read_key = key.replace("skysub_", "reduced_")
        hdul = open_fits(output_dir, read_key)
        header = hdul[0].header
        write_name = key
        write_to_fits(data, header, write_name, output_dir)
        logger.info(f"Frame written to directory {output_dir}, filename {write_name}") 


def run_sky_subtraction():
    """
    """
    logger.info("Starting the 1d extraction process...")

    reduced_files = get_reduced_frames()

    center_dict = choose_obj_centrum(reduced_files)

    subtracted_frames = remove_sky_background(center_dict)

    write_sky_subtracted_frames_to_disc(subtracted_frames)


if __name__ == "__main__":
    run_sky_subtraction()
