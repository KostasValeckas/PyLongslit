import numpy as np
import glob as glob
import argparse

"""
PyLongslit module for reducing frames (bias subtraction and flat-fielding). 
"""


def estimate_initial_error(data, master_bias, dark_frame):
    """
    Estimates the initial error in the data frame, assuming that the
    only sources of error are the read noise, the dark current, the bias
    and the Poisson noise.

    From:
    Richard Berry, James Burnell - The Handbook of Astronomical Image Processing
    -Willmann-Bell (2005), p. 45 - 46

    Parameters
    ----------
    data : numpy.ndarray
        The data frame.

    master_bias : numpy.ndarray
        The master bias frame.

    dark_frame : PyLongslit_frame
        The dark frame.
    """

    from pylongslit.parser import detector_params
    from pylongslit.overscan import estimate_frame_overscan_bias


    gain = detector_params["gain"]  # e/ADU
    read_noise = detector_params["read_out_noise"]  # e


    use_overscan = detector_params["overscan"]["use_overscan"]

    if use_overscan:
        overscan_frame = estimate_frame_overscan_bias(data)

    read_noise_error = read_noise / gain

    # Poisson noise
    if use_overscan:
        poisson_noise = np.sqrt(
            data - dark_frame.data - overscan_frame.data - master_bias.data
        )

    else:
        poisson_noise = np.sqrt(data - dark_frame.data - master_bias.data)

    # total error (see docs for calculations)
    if use_overscan:
        error = np.sqrt(
            poisson_noise**2
            + dark_frame.sigma**2
            + overscan_frame.sigma**2
            + master_bias.sigma**2
            + read_noise_error**2
        )

    else:
        error = np.sqrt(
            poisson_noise**2
            + dark_frame.sigma**2
            + master_bias.sigma**2
            + read_noise_error**2
        )

    return error


def read_raw_object_files():
    """
    Reads the raw object (science and standard star) files from the input
    directories and returns the science and standard star file lists.
    Does some initial checks also.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import skip_science_or_standard_bool
    from pylongslit.parser import science_params, standard_params
    from pylongslit.utils import FileList

    # initiate user parameters

    if skip_science_or_standard_bool == 0:
        logger.critical(
            'Both skip_science and skip_standard are set to "true" in the '
            "config file. There is nothing to perform the reduction on."
        )
        logger.error('Set at least one of them "false" and try again.')

        exit()

    elif skip_science_or_standard_bool == 1:
        star_file_list = None
        logger.warning(
            "Skipping standard star reduction as requested... (check config. file)"
        )
        science_file_list = FileList(science_params["science_dir"])

    elif skip_science_or_standard_bool == 2:
        logger.warning(
            "Skipping science reduction as requested... (check config. file)"
        )
        star_file_list = FileList(standard_params["standard_dir"])
        science_file_list = None

    else:
        star_file_list = FileList(standard_params["standard_dir"])
        science_file_list = FileList(science_params["science_dir"])

    if star_file_list is not None:
        if star_file_list.num_files == 0:
            logger.critical(
                "No standard star frames found in the standard star directory."
            )
            logger.error("Please check the config file and the directory.")

            exit()

        logger.info(f"Found {star_file_list.num_files} " "standard star frames:")

        star_file_list.print_files()

    if science_file_list is not None:

        if science_file_list.num_files == 0:
            logger.critical("No science frames found in the science directory.")
            logger.error("Please check the config file and the directory.")

            exit()

        logger.info(f"Found {science_file_list.num_files} " "science frames:")

        science_file_list.print_files()

    return science_file_list, star_file_list


def reduce_frame(frame, master_bias, master_flat, use_overscan, type):
    """
    Performs bias subtraction
    and flat fielding for a single frame, and handles the error propagation.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be reduced.

    master_bias : pylongslit.utils.PyLongslit_frame
        The master bias frame.

    master_flat : pylongslit.utils.PyLongslit_frame
        The master flat frame.

    use_overscan : bool
        Whether to use the overscan subtraction or not.

    type : str
        The type of frame to be reduced. Can be either 'science' or 'standard'.

    Returns
    -------
    frame : numpy.ndarray
        The reduced frame.

    error : numpy.ndarray
        The error in the reduced frame.
    """

    from pylongslit.logger import logger
    from pylongslit.overscan import estimate_frame_overscan_bias
    from pylongslit.dark import estimate_dark, check_dark_directory
    from pylongslit.parser import science_params, standard_params
    from pylongslit.utils import PyLongslit_frame

    initial_frame = frame.copy()

    # estimate dark current, if dark frames are provided
    # get directory to look for darks:
    directory =  science_params["science_dir"] if type == "science" else standard_params["standard_dir"]

    use_darks = check_dark_directory(directory)

    if use_darks:
        dark_frame = estimate_dark(directory, group = type)
    else :
        # a dummy dark frame that wont do anything. Helps keep the code cleaner for if/else
        dark_frame = PyLongslit_frame(np.zeros_like(frame), np.zeros_like(frame), None, None)

    initial_error = estimate_initial_error(frame, master_bias, dark_frame)


    frame = frame - dark_frame.data

    # subtract the overscan if requested
    if use_overscan:

        overscan = estimate_frame_overscan_bias(frame)
        frame = frame - overscan.data

    logger.info("Subtracting the bias...")

    frame = frame - master_bias.data

    logger.info("Dividing by the master flat frame...")

    frame = frame / master_flat.data

    # error calculations by error propagation. See docs for calculations.

    if use_overscan:

        error = (1 / master_flat.data) * np.sqrt(
            initial_error**2
            + dark_frame.sigma**2
            + overscan.sigma**2
            + master_bias.sigma**2
            + (
                (
                    (initial_frame - dark_frame.data - overscan.data - master_bias.data)
                    * master_flat.sigma
                    / master_flat.data
                )
                ** 2
            )
        )

    else:

        error = (1 / master_flat.data) * np.sqrt(
            initial_error**2
            + dark_frame.sigma**2
            + master_bias.sigma**2
            + (
                (
                    (initial_frame - dark_frame.sigma - master_bias.data)
                    * master_flat.sigma
                    / master_flat.data
                )
                ** 2
            )
        )

    # Handle NaNs and Infs
    if np.isnan(frame).any() or np.isinf(frame).any():
        logger.warning("NaNs or Infs detected in the frame. Replacing with zero.")
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

    # for error, nans and infs are replaced with the mean of the error frame
    if np.isnan(error).any() or np.isinf(error).any():
        logger.warning(
            "NaNs or Infs detected in the error-frame. Replacing with mean error."
        )
        error = np.nan_to_num(
            error,
            nan=np.nanmean(error),
            posinf=np.nanmean(error),
            neginf=np.nanmean(error),
        )

    return frame, error


def reduce_group(file_list, BIAS, FLAT, use_overscan, exptime, type):
    """
    Driver for 'reduce_frame' function. Reduces a list of frames.

    Creates and writes the reduced frames to disc.

    Parameters
    ----------
    file_list : pylongslit.utils.FileList
        A list of filenames to be reduced.

    BIAS : pylongslit.utils.PyLongslit_frame
        The master bias frame.

    FLAT : pylongslit.utils.PyLongslit_frame
        The master flat frame.

    use_overscan : bool
        Whether to use the overscan subtraction or not.

    exptime : float
        The exposure time of the frame (in seconds).

    type : str
        The type of frame to be reduced. Can be either 'science' or 'standard'.
    """

    from pylongslit.parser import science_params, standard_params, data_params
    from pylongslit.utils import open_fits, PyLongslit_frame
    from pylongslit.utils import check_rotation, flip_and_rotate
    from pylongslit.logger import logger

    if type != "science" and type != "standard":
        logger.critical("Reduction type must be either 'science' or 'standard'.")
        logger.critical("Contact the developers about this error.")
        exit()

    for file in file_list:

        print("---------------------------------")
        logger.info(f"Reducing frame {file} ...")

        hdu = (
            open_fits(science_params["science_dir"], file)
            if type == "science"
            else open_fits(standard_params["standard_dir"], file)
        )

        data = hdu[data_params["raw_data_hdu_index"]].data

        data, error = reduce_frame(data, BIAS, FLAT, use_overscan, type)

        # check if the frame needs to be rotated or flipped -
        # later steps rely on x being the dispersion axis
        # with wavelength increasing with pixel number

        transpose, flip = check_rotation()

        # transpose and/or flip the frame if needed
        if transpose or flip:
            logger.info("Transposing and/or flipping the data array...")
            data = flip_and_rotate(data, transpose, flip)
            logger.info("Transposing and/or flipping the error array...")
            error = flip_and_rotate(error, transpose, flip)

        logger.info("Frame reduced, writing to disc...")

        write_name = (
            "reduced_science_" + file
            if type == "science"
            else "reduced_standard_" + file
        )
        # the .fits extension is added in the write_to_disc method, so we remove it here
        write_name = write_name.replace(".fits", "")

        header = hdu[0].header

        # if no cropping is to be done in the next pipeline step, these parameters
        # allow the full frame to be used
        header["CROPY1"] = 0
        header["CROPY2"] = data.shape[0]

        frame = PyLongslit_frame(data, error, header, write_name)
        # set some fits headers that will be used later in the pipeline
        frame.header["CRRREMOVD"] = False
        frame.header["BCGSUBBED"] = False
        frame.header["SKYSUBBED"] = False
        frame.show_frame()
        frame.write_to_disc()

        print("---------------------------------")


def reduce_all():
    """
    Driver for the reduction of all observations (standard star, science, arc lamp)
    in the output directory.
    """

    from pylongslit.parser import (
        detector_params,
        output_dir,
        skip_science_or_standard_bool,
    )
    from pylongslit.parser import science_params, standard_params
    from pylongslit.utils import PyLongslit_frame
    from pylongslit.logger import logger

    # prepare some parameters and load master frames

    use_overscan = detector_params["overscan"]["use_overscan"]
    BIAS = PyLongslit_frame.read_from_disc("master_bias.fits")
    FLAT = PyLongslit_frame.read_from_disc("master_flat.fits")

    # get the files to be reduced
    science_files, standard_files = read_raw_object_files()

    # Standard star reduction

    if skip_science_or_standard_bool == 1:
        logger.warning("Skipping standard star reduction as requested...")
    else:
        logger.info("Starting reduction of following standard star frames:")

        standard_files.print_files()

        exptime = standard_params["exptime"]

        reduce_group(standard_files, BIAS, FLAT, use_overscan, exptime, "standard")

    # Science reduction

    if skip_science_or_standard_bool == 2:
        logger.warning("Skipping science reduction as requested...")

    else:
        logger.info("Starting reduction of following science frames:")

        science_files.print_files()

        exptime = science_params["exptime"]

        reduce_group(science_files, BIAS, FLAT, use_overscan, exptime, "science")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit cosmic-ray removal procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    reduce_all()


if __name__ == "__main__":
    main()
