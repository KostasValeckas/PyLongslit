from logger import logger
from parser import skip_science_or_standard_bool
from parser import output_dir, extract_params
from utils import list_files, hist_normalize, open_fits, write_to_fits
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from numpy.polynomial.chebyshev import chebfit, chebval
from utils import show_frame, get_file_group, choose_obj_centrum
from utils import refine_obj_center

def get_reduced_frames():
    """
    Driver for `get_reduced_frames` that acounts for skip_science and/or
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

        reduced_files = get_file_group("reduced_science")

    elif skip_science_or_standard_bool == 2:

        logger.warning("Science extraction is set to be skipped in the config file.")
        logger.warning("Will only extract standard star spectra.")

        reduced_files = get_file_group("reduced_std")

    else:

        reduced_files = get_file_group("reduced_science", "reduced_std")

    return reduced_files

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
    plot_title = lambda file: f"Object position estimation for {file}.\n" \
    "Press on the object on a spectral point with no bright sky-lines (but away from detector edges.)" \
    "\nYou can try several times. Press 'q' or close plot when done."

    titles = [plot_title(file) for file in file_list]

    return choose_obj_centrum(file_list, titles)





def estimate_sky_regions(slice_spec, spatial_center_guess, FWHM_AP):
    """
    From a user inputted object center guess, tries to refine the object centrum,
    and then estimates the sky region around the object.

    Parameters
    ----------
    slice_spec : array
        The slice of the data.

    spatial_center_guess : int
        The user clicked center of the object.

    FWHM_AP : int
        The FWHM of the object.

    Returns
    -------
    x_spec : array
        The x-axis of the slice.

    x_sky : array
        The x-axis of the sky region.

    sky_val : array
        The values of the sky region.

    sky_left : int
        The left boundary of the sky region.

    sky_right : int
        The right boundary of the sky region.
    """

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

    for column in range(n_spectal):
        logger.info(f"Fitting sky for spectal pixel {column}...")
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
        logger.info(f"Sky fit for spectral pixel {column} complete.")
        print("------------------------------------")

    #plot QA
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

        skysub_data = data - sky_map

        #plot QA

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
