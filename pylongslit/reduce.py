import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import argparse

"""
Module for reducing (bias subtraction, flat division) and combining 
exposures (science, standard star and arc lamps).
"""


def estimate_initial_error(data, exptime, master_bias):
    """
    From:
    Richard Berry, James Burnell - The Handbook of Astronomical Image Processing
    -Willmann-Bell (2005), p. 45 - 46
    """

    from pylongslit.dark import estimate_dark
    from pylongslit.parser import detector_params
    from pylongslit.overscan import estimate_frame_overscan_bias

    gain = detector_params["gain"]  # e/ADU
    read_noise = detector_params["read_out_noise"]  # e

    dark = detector_params["dark_current"]  # e/s/pixel

    dark_current, dark_noise_error = estimate_dark(dark, exptime)

    use_overscan = detector_params["overscan"]["use_overscan"]
    
    if use_overscan:
        overscan_frame = estimate_frame_overscan_bias(data)


    read_noise_error = np.sqrt(read_noise / gain)

    # Poisson noise
    if use_overscan:
        poisson_noise = np.sqrt(data - dark_current - overscan_frame.data - master_bias.data)  
    
    else:
        poisson_noise = np.sqrt(data - dark_current - master_bias.data)

    # total error
    if use_overscan:
        error = np.sqrt(poisson_noise**2 + dark_noise_error**2 + overscan_frame.sigma**2 + master_bias.sigma**2 +  read_noise_error**2)

    else:
        error = np.sqrt(poisson_noise**2 + dark_noise_error**2 + master_bias.sigma**2 +  read_noise_error**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(data, origin="lower", cmap="gray")
    ax1.set_title("Data")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    im = ax2.imshow(error, origin="lower", cmap="viridis")
    ax2.set_title("Error")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    fig.colorbar(im, ax=ax2, orientation="vertical")
    plt.show()

    return error


def read_raw_object_files():

    from pylongslit.logger import logger
    from pylongslit.parser import skip_science_or_standard_bool
    from pylongslit.parser import science_params, standard_params
    from pylongslit.utils import FileList, list_files

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
        science_file_list = FileList(science_params["science_dir"])

    elif skip_science_or_standard_bool == 2:
        star_file_list = FileList(standard_params["standard_dir"])
        science_file_list = None

    else:
        star_file_list = FileList(standard_params["standard_dir"])
        science_file_list = FileList(science_params["science_dir"])

    if star_file_list is not None:
        logger.info(
            f"Reducing {star_file_list.num_files} "
            "standard star frames:"
        )

        list_files(star_file_list)

    if science_file_list is not None:
        logger.info(
            f"Reducing {science_file_list.num_files} "
            "science frames:"
        )

        list_files(science_file_list)


    logger.info("Cosmic-ray removal procedure finished.")

    return science_file_list, star_file_list


def reduce_frame(frame, master_bias, master_flat, use_overscan, overscan_dir, exptime):
    """
    Performs overscan subtraction, bias subtraction
    and flat fielding of a single frame.
    """

    from pylongslit.utils import hist_normalize
    from pylongslit.logger import logger
    from pylongslit.overscan import estimate_frame_overscan_bias
    from pylongslit.dark import estimate_dark

    initial_frame = frame.copy()
    initial_error = estimate_initial_error(frame, exptime, master_bias)

    # subtract the dark current

    dark_current, dark_error = estimate_dark(frame, exptime)

    if use_overscan:

        overscan = estimate_frame_overscan_bias(frame)
        frame = frame - overscan.data

    logger.info("Subtracting the bias...")

    frame = frame - master_bias.data

    logger.info("Dividing by the master flat frame...")

    frame = frame / master_flat.data

    if use_overscan:

        error = (1 / master_flat.data) * np.sqrt(
            initial_error**2
            + dark_error**2
            + overscan.sigma**2
            + master_bias.sigma**2
            + ((
                (initial_frame - dark_current - overscan.data - master_bias.data)
                * master_flat.sigma
                / master_flat.data
            )
            ** 2)
        )

    else:
        
        error = (1 / master_flat.data) * np.sqrt(
            initial_error**2
            + dark_error**2
            + master_bias.sigma**2
            + ((
                (initial_frame - dark_current - master_bias.data)
                * master_flat.sigma
                / master_flat.data
            )
            ** 2)
        )

    # Handle NaNs and Infs
    if np.isnan(frame).any() or np.isinf(frame).any():
        logger.warning("NaNs or Infs detected in the frame. Replacing with zero.")
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(error).any() or np.isinf(error).any():
        logger.warning("NaNs or Infs detected in the error-frame. Replacing with zero.")
        error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)


    return frame, error


def reduce_group(file_list, BIAS, FLAT, use_overscan, overscan_dir, exptime, type):
    """
    Driver for 'reduce_frame' function. Reduces a list of frames.

    Parameters
    ----------
    file_list : list
        A list of filenames to be reduced.

    BIAS : numpy.ndarray
        The master bias frame.

    FLAT : numpy.ndarray
        The master flat frame.

    use_overscan : bool
        Whether to use the overscan subtraction or not.
    """

    from pylongslit.parser import output_dir, science_params, standard_params, data_params
    from pylongslit.utils import open_fits, PyLongslit_frame
    from pylongslit.utils import check_rotation, flip_and_rotate
    from pylongslit.logger import logger

    if type != "science" and type != "standard":
        logger.critical("Reduction type must be either 'science' or 'standard'.")
        logger.critical("Contact the developers about this error.")

    for file in file_list:

        logger.info(f"Reducing frame {file} ...")
        

        hdu = open_fits(science_params["science_dir"], file) if type == "science" else open_fits(standard_params["standard_dir"], file)

        data = hdu[data_params["raw_data_hdu_index"]].data

        data, error = reduce_frame(
            data, BIAS, FLAT, use_overscan, overscan_dir, exptime
        )

        # check if the frame needs to be rotated or flipped -
        # later steps rely on x being the dispersion axis
        # with wavelength increasing with pixel number

        transpose, flip = check_rotation()

        # transpose and/or flip the frame if needed
        if transpose or flip:
            data = flip_and_rotate(data, transpose, flip)
            error = flip_and_rotate(error, transpose, flip)

        logger.info("Frame reduced, writing to disc...")

        write_name = "reduced_science_" + file if type == "science" else "reduced_standard_" + file
        write_name = write_name.replace(".fits", "")

        header = hdu[0].header

        # if no cropping is to be done in the next step, these parameters
        # allow the full frame to be used
        header["CROPY1"] = 0
        header["CROPY2"] = data.shape[0]

        frame = PyLongslit_frame(data, error, header, write_name)
        frame.show_frame(normalize=False)
        frame.show_frame()
        frame.write_to_disc()


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
    from pylongslit.utils import list_files, PyLongslit_frame
    from pylongslit.utils import get_bias_and_flats
    from pylongslit.logger import logger
    from pylongslit.overscan import detect_overscan_direction

    use_overscan = detector_params["overscan"]["use_overscan"]

    overscan_dir = detect_overscan_direction() if use_overscan else None
    

    BIAS = PyLongslit_frame.read_from_disc("master_bias.fits")
    FLAT = PyLongslit_frame.read_from_disc("master_flat.fits")

    BIAS.show_frame(normalize=False)
    FLAT.show_frame(normalize=False)

    logger.info(f"Fetching cosmic-ray removed files from {output_dir} ...")

    science_files, standard_files = read_raw_object_files()

    # Standard star reduction

    if skip_science_or_standard_bool == 1:
        logger.warning("Skipping standard star reduction as requested...")
    else:
        logger.info("Reducing following standard star frames:")

        list_files(standard_files)

        exptime = standard_params["exptime"]

        reduce_group(standard_files, BIAS, FLAT, use_overscan, overscan_dir, exptime, "standard")

    # Science reduction

    if skip_science_or_standard_bool == 2:
        logger.warning("Skipping science reduction as requested...")

    else:
        logger.info("Reducing following science frames:")

        list_files(science_files)

        exptime = science_params["exptime"]

        reduce_group(science_files, BIAS, FLAT, use_overscan, overscan_dir, exptime, "science")


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
