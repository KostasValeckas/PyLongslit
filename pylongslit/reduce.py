import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import argparse

"""
Module for reducing (bias subtraction, flat division) and combining 
exposures (science, standard star and arc lamps).
"""


def estimate_initial_error(data, exptime):
    """
    From:
    Richard Berry, James Burnell - The Handbook of Astronomical Image Processing
    -Willmann-Bell (2005), p. 45 - 46
    """

    from pylongslit.dark import estimate_dark

    from pylongslit.parser import detector_params

    gain = detector_params["gain"]  # e/ADU
    read_noise = detector_params["read_out_noise"]  # e

    dark = detector_params["dark_current"]  # e/s/pixel

    _, dark_noise_error = estimate_dark(dark, exptime)

    read_noise_error = np.sqrt(read_noise / gain)

    # Poisson noise
    poisson_noise = np.sqrt(data)

    plt.imshow(data, origin="lower", cmap="gray")
    plt.colorbar()
    plt.title("Data")
    plt.show()

    plt.imshow(poisson_noise, origin="lower", cmap="gray")
    plt.colorbar()
    plt.title("Poisson noise")
    plt.show()

    # total error
    error = np.sqrt(poisson_noise**2 + dark_noise_error**2 + read_noise_error**2)

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


def read_crr_files():
    """
    Read the cosmic-ray removed files from the output directory and
    perform some checks.

    Returns
    -------
    science_files : list
        A list of cosmic-ray removed science files.

    standard_files : list
        A list of cosmic-ray removed standard star files.
    """

    from pylongslit.utils import get_filenames
    from pylongslit.logger import logger

    science_files = get_filenames(starts_with="crr_science")
    standard_files = get_filenames(starts_with="crr_std")

    logger.info(f"Found {len(science_files)} cosmic-ray removed science files.")
    logger.info(f"Found {len(standard_files)} cosmic-ray removed standard star files.")

    # sort alphabetically to correctly match the centers

    science_files.sort()
    standard_files.sort()

    return science_files, standard_files


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
    initial_error = estimate_initial_error(frame, exptime)

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

    return frame, error


def reduce_group(file_list, BIAS, FLAT, use_overscan, overscan_dir, exptime):
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

    from pylongslit.parser import output_dir
    from pylongslit.utils import open_fits, write_to_fits, PyLongslit_frame
    from pylongslit.utils import check_rotation, flip_and_rotate
    from pylongslit.logger import logger

    for file in file_list:

        logger.info(f"Reducing frame {file} ...")

        hdu = open_fits(output_dir, file)

        data = hdu[0].data

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

        write_name = file.replace("crr_", "reduced_")

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

    science_files, standard_files = read_crr_files()

    # Standard star reduction

    if skip_science_or_standard_bool == 1:
        logger.warning("Skipping standard star reduction as requested...")
    else:
        logger.info("Reducing following standard star frames:")

        list_files(standard_files)

        exptime = standard_params["exptime"]

        reduce_group(standard_files, BIAS, FLAT, use_overscan, overscan_dir, exptime)

    # Science reduction

    if skip_science_or_standard_bool == 2:
        logger.warning("Skipping science reduction as requested...")

    else:
        logger.info("Reducing following science frames:")

        list_files(science_files)

        exptime = science_params["exptime"]

        reduce_group(science_files, BIAS, FLAT, use_overscan, overscan_dir, exptime)


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
