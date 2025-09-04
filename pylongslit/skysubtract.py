import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from numpy.polynomial.chebyshev import chebfit, chebval
from tqdm import tqdm
import argparse

"""
PyLongslit sky-subtraction module for polynomial sky-fitting.
"""


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

    from pylongslit.utils import choose_obj_centrum

    # used for more readable plotting code
    plot_title = (
        lambda file: f"Object position estimation for {file}.\n"
        "Press on the object away from detector edges."
        "\nYou can try several times. Press 'q' or close plot when done.\n"
        "If no center is clicked, the procedure will be skipped for this frame."
    )

    titles = [plot_title(file) for file in file_list]

    return choose_obj_centrum(file_list, titles)


def fit_sky_one_column(
    slice_spec,
    spatial_center_guess,
    fwhm_guess,
    fwhm_thresh,
    sigma_cut,
    sigma_iters,
    sky_order,
):
    """
    In a detector slice, evaluates the sky region using `estimate_sky_regions`,
    removes the outliers using sigma-clipping, and fits the sky using a Chebyshev polynomial.

    Parameters
    ----------
    slice_spec : array
        The slice of the data.

    spatial_center_guess : int
        The user clicked center of the object.

    fwhm_guess : int
        The FWHM guess of the object.

    fwhm_thresh : int
        The threshold for the object FWHM.

    sigma_cut : float
        The sigma value for sigma-clipping in the sky fitting.

    sigma_iters : int
        The number of iterations for sigma-clipping in the sky fitting.

    sky_order : int
        The order of the Chebyshev polynomial to fit the sky.

    Returns
    -------
    sky_fit : array
        The fitted sky background (evaluated fit).

    clip_mask : array
        A boolean mask of the outliers.

    x_sky : array
        The x (spacial pixels) values of the sky region.

    sky_val : array
        The y (counts) values of the sky region.
    """

    from pylongslit.utils import estimate_sky_regions

    # sky region for this column
    _, sky_left, sky_right = estimate_sky_regions(
        slice_spec, spatial_center_guess, fwhm_guess, fwhm_thresh
    )

    # x array for the fit
    x_spec = np.arange(len(slice_spec))

    # select the sky region in x and sky arrays
    x_sky = np.concatenate((x_spec[:sky_left], x_spec[sky_right:]))
    sky_val = np.concatenate((slice_spec[:sky_left], slice_spec[sky_right:]))

    # mask the outliers
    clip_mask = sigma_clip(sky_val, sigma=sigma_cut, maxiters=sigma_iters).mask

    # fit the sky
    coeff_apsky, _ = chebfit(
        x_sky[~clip_mask], sky_val[~clip_mask], deg=sky_order, full=True
    )

    # evaluate the fit
    sky_fit = chebval(x_spec, coeff_apsky)

    residuals = sky_val[~clip_mask] - chebval(x_sky[~clip_mask], coeff_apsky)

    return sky_fit, clip_mask, x_sky, sky_val, residuals


def fit_sky_QA(
    slice_spec,
    spatial_center_guess,
    spectral_column,
    fwhm_guess,
    fwhm_thresh,
    sigma_cut,
    sigma_iters,
    sky_order,
    figsize=(10, 6),
):
    """
    A QA method for the sky fitting. Performs the sky-fitting routine
    for one column of the detector, and plots the results.

    This is used for user insection, before looping through the whole detector
    in the `make_sky_map` method.

    Parameters
    ----------
    slice_spec : array
        The slice of the data.

    spatial_center_guess : int
        The user clicked spacial center of the object.

    spectral_column : int
        The spectral column to extract

    fwhm_guess : int
        The FWHM of the object.

    fwhm_thresh : int
        The threshold for the object FWHM.

    sigma_cut : float
        The sigma value for sigma-clipping in the sky fitting.

    sigma_iters : int
        The number of iterations for sigma-clipping in the sky fitting.

    sky_order : int
        The order of the Chebyshev polynomial to fit the sky.

    figsize : tuple
        The size of the figure to be displayed.
        Default is (10, 6).
    """

    from pylongslit.utils import estimate_sky_regions

    refined_center, sky_left, sky_right = estimate_sky_regions(
        slice_spec, spatial_center_guess, fwhm_guess, fwhm_thresh
    )

    # dummy x array for plotting
    x_spec = np.arange(len(slice_spec))

    # fit the sky
    sky_fit, clip_mask, x_sky, sky_val, reasiduals = fit_sky_one_column(
        slice_spec,
        refined_center,
        fwhm_guess,
        fwhm_thresh,
        sigma_cut,
        sigma_iters,
        sky_order,
    )

    # plot the results
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot the sky fit
    ax1.axvline(x=sky_left, color="r", linestyle="--", label="Object boundary")
    ax1.axvline(x=sky_right, color="r", linestyle="--")
    ax1.plot(
        x_spec,
        slice_spec,
        label=f"Detector slice around spectral pixel {spectral_column}",
    )
    ax1.plot(x_sky[clip_mask], sky_val[clip_mask], "rx", label="Rejected Outliers")
    ax1.plot(x_spec, sky_fit, label="Sky fit")
    ax1.set_ylabel("Detector counts (ADU)")
    ax1.legend()
    ax1.set_title(
        "Sky-background fitting QA. Ensure the fit is reasonable, and that the object "
        "is completely encapsulated by the red lines.\n"
        "If not, change relative parameters (polynomial order and object FWHM) in the configuration file."
    )

    # Plot the residuals
    ax2.plot(x_sky[~clip_mask], reasiduals, "o", label="Residuals")
    ax2.axhline(y=0, color="k", linestyle="--")
    ax2.set_xlabel("Pixels (spatial direction)")
    ax2.set_ylabel("Residuals (ADU)")
    ax2.legend()

    plt.show()


def make_sky_map(
    filename,
    data,
    spatial_center_guess,
    fwhm_guess,
    fwhm_thresh,
    sigma_cut,
    sigma_iters,
    sky_order,
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

    fwhm_guess : int
        The FWHM of the object.

    fwhm_thresh : int
        The threshold for the object FWHM.

    sigma_cut : float
        The sigma value for sigma-clipping in the sky fitting.

    sigma_iters : int
        The number of iterations for sigma-clipping in the sky fitting.

    sky_order : int
        The order of the Chebyshev polynomial to fit the sky.

    Returns
    -------
    sky_map : array
        Sky-background fit evaluated at every pixel
    """

    from pylongslit.logger import logger
    from pylongslit.utils import PyLongslit_frame

    # get detector shape
    n_spacial = data.shape[0]
    n_spectal = data.shape[1]

    # evaluate the sky column-wise and insert in this array, together with the error
    sky_map = np.zeros((n_spacial, n_spectal))
    sky_error = np.zeros((n_spacial, n_spectal))

    logger.info(f"Creating sky map for {filename}...")
    for column in tqdm(range(n_spectal), desc=f"Fitting sky background for {filename}"):
        slice_spec = data[:, column]
        sky_fit, _, _, _, residuals = fit_sky_one_column(
            slice_spec,
            spatial_center_guess,
            fwhm_guess,
            fwhm_thresh,
            sigma_cut,
            sigma_iters,
            sky_order,
        )
        # insert the results
        sky_map[:, column] = sky_fit
        RMS_residuals = np.sqrt(np.mean(residuals**2))
        sky_error[:, column] = np.full(n_spacial, RMS_residuals)

    sky_frame = PyLongslit_frame(sky_map, sky_error, None, "skymap_" + filename)
    sky_frame.show_frame()
    sky_frame.write_to_disc()

    return sky_frame


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
    from pylongslit.logger import logger
    from pylongslit.parser import trace_params, sky_params
    from pylongslit.utils import PyLongslit_frame, hist_normalize

    # user-defined paramteres relevant for sky-subtraction

    # sigma clipping parameters
    sigma_cut = sky_params["sigma_cut"]
    sigma_iters = sky_params["sigma_clip_iters"]
    # order of the sky-fit
    sky_order = sky_params["fit_order"]

    # every key in the dict is a filename
    for file in center_dict.keys():

        # depending if it is a science or standard frame, the FWHM is different
        if "science" in file:
            fwhm_guess = trace_params["object"]["fwhm_guess"]
            fwhm_thresh = trace_params["object"]["fwhm_thresh"]
        else:
            fwhm_guess = trace_params["standard"]["fwhm_guess"]
            fwhm_thresh = trace_params["standard"]["fwhm_thresh"]

        logger.info(f"Starting sky subtraction for {file}...")
        try:
            frame = PyLongslit_frame.read_from_disc(file)
        except FileNotFoundError:
            logger.error(f"File {file} not found. Try running the reduction first.")
            exit()

        if frame.header["SKYSUBBED"] == True:
            logger.warning(f"Sky subtraction already performed for {file}. Skipping...")
            continue

        if frame.header["BCGSUBBED"] == True:
            logger.warning(
                f"Sky-subtraction was already performed by A-B image subtraction for {file}."
            )
            logger.warning(f"Using this routine might not be neccesery.")
            logger.warning(f"Inspect whether further sky-subtraction is needed.")
            logger.warning(
                f"This routine introduces noise - and should not be used if not neeeded."
            )

        # get the data and the center
        data = frame.data.copy()
        error = frame.sigma.copy()
        clicked_point = center_dict[file]

        # start with a QA at user defined point
        spacial_center_guess = clicked_point[1]
        spectral_center_guess = clicked_point[0]

        slice_spec = data[:, spectral_center_guess]

        fit_sky_QA(
            slice_spec,
            spacial_center_guess,
            spectral_center_guess,
            fwhm_guess,
            fwhm_thresh,
            sigma_cut,
            sigma_iters,
            sky_order,
        )

        # create the sky map and subtract
        sky_map = make_sky_map(
            file,
            data,
            spacial_center_guess,
            fwhm_guess,
            fwhm_thresh,
            sigma_cut,
            sigma_iters,
            sky_order,
        )

        logger.info(f"Sky map created for {file}")
        logger.info("Subtracting the sky background...")

        skysub_data = data - sky_map.data
        skysub_error = np.sqrt(error**2 + sky_map.sigma**2)

        # plot the difference for QA
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].imshow(hist_normalize(data), cmap="gray")
        ax[0].set_title("Original data (histogram normalized)")
        ax[1].imshow(hist_normalize(skysub_data), cmap="gray")
        ax[1].set_title("Sky-subtracted data (histogram normalized)")
        fig.suptitle(
            f"Sky subtraction QA for {file}.\n"
            f"Ensure that only random noise is left in the background.\n"
            f"If not, revise the sky subtraction parameters in the configuration file."
        )
        plt.show()

        # Swap the data and sigma in the frame, show and write to disc
        frame.data = skysub_data
        frame.sigma = skysub_error
        frame.header["SKYSUBBED"] = True

        frame.show_frame()
        frame.write_to_disc()


def write_sky_subtracted_frames_to_disc(subtracted_frames):
    """
    NOT USED

    Writes sky-subtracted frames to the output directory.

    Parameters
    ----------
    subtracted_frames : dict
        A dictionary containing the sky-subtracted frames.
        Format: {filename: data}
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir
    from pylongslit.utils import open_fits, write_to_fits

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

    from pylongslit.logger import logger
    from pylongslit.utils import get_reduced_frames

    logger.info("Starting the sky-subtraction process...")

    logger.info("Fetching the reduced frames...")
    reduced_files = get_reduced_frames()
    if len(reduced_files) == 0:
        logger.error(
            "No reduced frames found. Please run the reduction procedure first."
        )
        exit()

    # get the center estimations
    center_dict = choose_obj_centrum_sky(reduced_files)

    # this is the main driver method.
    remove_sky_background(center_dict)

    logger.info("Sky subtraction complete.")
    print("\n------------------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit sky-subtraction procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_sky_subtraction()


if __name__ == "__main__":
    main()
