from logger import logger
from parser import skip_science_or_standard_bool
from parser import output_dir, extract_params
from utils import list_files, hist_normalize, open_fits, write_to_fits
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from numpy.polynomial.chebyshev import chebfit, chebval
from utils import show_frame, get_file_group, choose_obj_centrum, estimate_sky_regions
from utils import refine_obj_center, get_reduced_frames
from tqdm import tqdm


def choose_obj_centrum_sky(file_list):
    """
    A wrapper for `choose_obj_centrum` that is used in the sky-subtraction routine.

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
        "Press on the object on a spectral point with no bright sky-lines (but away from detector edges.)"
        "\nYou can try several times. Press 'q' or close plot when done."
    )

    titles = [plot_title(file) for file in file_list]

    return choose_obj_centrum(file_list, titles)


def fit_sky_one_column(
    slice_spec,
    spatial_center_guess,
    FWHM_AP,
    SIGMA_APSKY,
    ITERS_APSKY,
    ORDER_APSKY,
):
    """
    In a detector slice, evaluates the sky region using `estimate_sky_regions`,
    removes the outlies using sigma-clipping, and fits the sky using a Chebyshev polynomial.

    Parameters
    ----------
    slice_spec : array
        The slice of the data.

    spatial_center_guess : int
        The user clicked center of the object.

    FWHM_AP : int
        The FWHM of the object.

    SIGMA_APSKY : float
        The sigma value for sigma-clipping in the sky fitting.

    ITERS_APSKY : int
        The number of iterations for sigma-clipping in the sky fitting.

    ORDER_APSKY : int
        The order of the Chebyshev polynomial to fit the sky.

    Returns
    -------
    sky_fit : array
        The fitted sky background (evaluated fit).
    """

    # sky region for this column
    _, sky_left, sky_right = estimate_sky_regions(
        slice_spec, spatial_center_guess, FWHM_AP
    )

    # x array for the fit
    x_spec = np.arange(len(slice_spec))

    # select the sky region in x and sky arrays
    x_sky = np.concatenate((x_spec[:sky_left], x_spec[sky_right:]))
    sky_val = np.concatenate((slice_spec[:sky_left], slice_spec[sky_right:]))

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
    spectral_column,
    FWHM_AP,
    SIGMA_APSKY,
    ITERS_APSKY,
    ORDER_APSKY,
    figsize=(18, 12),
):
    """
    A QA method for the sky fitting. Performs the sky-fitting routine
    for one column of the detector, and plots the results.

    This is used for used insection, before looping through the whole detector
    in the `make_sky_map` method.

    Parameters
    ----------
    slice_spec : array
        The slice of the data.

    spatial_center_guess : int
        The user clicked spacial center of the object.

    spectral_column : int
        The spectral column to extract

    FWHM_AP : int
        The FWHM of the object.

    SIGMA_APSKY : float
        The sigma value for sigma-clipping in the sky fitting.

    ITERS_APSKY : int
        The number of iterations for sigma-clipping in the sky fitting.

    ORDER_APSKY : int
        The order of the Chebyshev polynomial to fit the sky.

    figsize : tuple
        The size of the figure to be displayed.
        Default is (18, 12).
    """

    refined_center, sky_left, sky_right = estimate_sky_regions(
        slice_spec, spatial_center_guess, FWHM_AP
    )

    # dummy x array for plotting
    x_spec = np.arange(len(slice_spec))

    sky_fit = fit_sky_one_column(
        slice_spec, refined_center, FWHM_AP, SIGMA_APSKY, ITERS_APSKY, ORDER_APSKY
    )

    plt.figure(figsize=figsize)
    plt.axvline(x=sky_left, color="r", linestyle="--", label="Object boundary")
    plt.axvline(x=sky_right, color="r", linestyle="--")
    plt.plot(
        x_spec,
        slice_spec,
        label=f"Detector slice around spectral pixel {spectral_column}",
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
    filename, data, spatial_center_guess, FWHM_AP, SIGMA_APSKY, ITERS_APSKY, ORDER_APSKY
):
    """
    Loops through the detector columns, and fits the sky background for each one.
    Each column is fitted using the `fit_sky_one_column` method. Constructs
    an image of the sky-background on the detector.

    Parameters
    ----------
    data : array
        The frame detector data.

    spatial_center_guess : int
        User-clicked spatial center of the object.

    FWHM_AP : int
        The FWHM of the object.

    SIGMA_APSKY : float
        The sigma value for sigma-clipping in the sky fitting.

    ITERS_APSKY : int
        The number of iterations for sigma-clipping in the sky fitting.

    ORDER_APSKY : int
        The order of the Chebyshev polynomial to fit the sky.

    Returns
    -------
    sky_map : array
        Sky-background fit evaluated at every pixel
    """

    # get detector shape
    n_spacial = data.shape[0]
    n_spectal = data.shape[1]

    # evaluate the sky column-wise and insert in this array
    sky_map = np.zeros((n_spacial, n_spectal))

    logger.info(f"Creating sky map for {filename}...")
    for column in tqdm(range(n_spectal), desc=f"Fitting sky background for {filename}"):
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

    # plot QA
    title = f"Evaluated sky-background for {filename}"

    show_frame(sky_map, title)

    return sky_map


def remove_sky_background(center_dict):
    """
    For all reduced files, takes user estimate spacial object center,
    performs sky fitting QA using `fit_sky_QA`, constructs a sky map
    for every frame using `m̀ake_sky_map` and subtracts it from the reduced
    frame.

    Parameters
    ----------
    center_dict : dict
        A dictionary containing the user clicked object centers.
        Format: {filename: (x, y)}

    Returns
    -------
    subtracted_frames : dict
        A dictionary containing the sky-subtracted frames.
        Format: {filename: data}
    """

    # user-defined paramteres relevant for sky-subtraction

    # user-guess of FWHM of the object
    FWHM_AP = extract_params["FWHM_AP"]
    # sigma clipping parameters
    SIGMA_APSKY = extract_params["SIGMA_APSKY"]
    ITERS_APSKY = extract_params["ITERS_APSKY"]
    # order of the sky-fit
    ORDER_APSKY = extract_params["ORDER_APSKY"]

    # final container - this keeps the results
    subtracted_frames = {}

    # every key in the dict is a filename
    for file in center_dict.keys():

        frame = open_fits(output_dir, file)
        data = frame[0].data

        clicked_point = center_dict[file]

        # start with a QA at user defined point
        spacial_center_guess = clicked_point[1]
        spectral_center_guess = clicked_point[0]

        slice_spec = data[:, spectral_center_guess]

        fit_sky_QA(
            slice_spec,
            spacial_center_guess,
            spectral_center_guess,
            FWHM_AP,
            SIGMA_APSKY,
            ITERS_APSKY,
            ORDER_APSKY,
        )

        # create the sky map and subtract
        sky_map = make_sky_map(
            file,
            data,
            spacial_center_guess,
            FWHM_AP,
            SIGMA_APSKY,
            ITERS_APSKY,
            ORDER_APSKY,
        )

        logger.info(f"Sky map created for {file}")
        logger.info("Subtracting the sky background...")

        skysub_data = data - sky_map

        # plot QA

        title = f"Sky-subtracted frame {file}"

        show_frame(skysub_data, title)

        # create new filename for later handling
        key = file.replace("reduced_", "skysub_")

        subtracted_frames[key] = skysub_data

    return subtracted_frames


def write_sky_subtracted_frames_to_disc(subtracted_frames):
    """
    Writes sky-subtracted frames to the output directory.

    Parameters
    ----------
    subtracted_frames : dict
        A dictionary containing the sky-subtracted frames.
        Format: {filename: data}
    """

    for filename, data in subtracted_frames.items():

        # steal header from the original file
        # switch back to reduced filename to steal header
        # TODO: this is a bit hacky, maybe refactor
        read_key = filename.replace("skysub_", "reduced_")
        hdul = open_fits(output_dir, read_key)
        header = hdul[0].header
        # write the frame to the output directory
        write_to_fits(data, header, filename, output_dir)
        logger.info(f"Frame written to directory {output_dir}, filename {filename}")


def run_sky_subtraction():
    """
    Driver for the sky-subtraction process.
    """

    logger.info("Starting the 1d extraction process...")

    reduced_files = get_reduced_frames()

    center_dict = choose_obj_centrum_sky(reduced_files)

    subtracted_frames = remove_sky_background(center_dict)

    write_sky_subtracted_frames_to_disc(subtracted_frames)

    logger.info("Sky subtraction complete.")
    print("\n------------------------------------")


if __name__ == "__main__":
    run_sky_subtraction()
