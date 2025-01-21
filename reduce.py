import numpy as np
import glob as glob
from astropy.io import fits
from parser import detector_params, output_dir, skip_science_or_standard_bool
from parser import science_params, standard_params, arc_params
from utils import FileList, open_fits, write_to_fits, list_files, hist_normalize
from logger import logger
import matplotlib.pyplot as plt
from overscan import subtract_overscan_from_frame, detect_overscan_direction
import os
from matplotlib.patches import Rectangle
from utils import check_rotation, flip_and_rotate, get_filenames, get_bias_and_flats

"""
Module for reducing (bias subtraction, flat division) and combining 
exposures (science, standard star and arc lamps).
"""


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

    science_files = get_filenames(starts_with="crr_science")
    standard_files = get_filenames(starts_with="crr_std")

    logger.info(f"Found {len(science_files)} cosmic-ray removed science files.")
    logger.info(f"Found {len(standard_files)} cosmic-ray removed standard star files.")

    # sort alphabetically to correctly match the centers

    science_files.sort()
    standard_files.sort()

    return science_files, standard_files


def reduce_frame(frame, master_bias, master_flat, use_overscan, overscan_dir):
    """
    Performs overscan subtraction, bias subtraction
    and flat fielding of a single frame.
    """

    if use_overscan:

        frame = subtract_overscan_from_frame(frame, overscan_dir)
        normalized_frame = hist_normalize(frame)
        plt.imshow(normalized_frame, cmap="gray")
        plt.show()
    else:
        logger.info("Subtracting the bias...")
        frame = frame - master_bias

    logger.info("Dividing by the master flat frame...")

    frame = frame / master_flat

    # Handle NaNs and Infs
    if np.isnan(frame).any() or np.isinf(frame).any():
        logger.warning("NaNs or Infs detected in the frame. Replacing with zero.")
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

    return frame


def reduce_group(file_list, BIAS, FLAT, use_overscan, overscan_dir):
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

    for file in file_list:

        logger.info(f"Reducing frame {file} ...")

        hdu = open_fits(output_dir, file)

        data = hdu[0].data

        data = reduce_frame(data, BIAS, FLAT, use_overscan, overscan_dir)

        # check if the frame needs to be rotated or flipped -
        # later steps rely on x being the dispersion axis
        # with wavelength increasing with pixel number

        transpose, flip = check_rotation()

        # transpose and/or flip the frame if needed
        if transpose or flip:
            data = flip_and_rotate(data, transpose, flip)

        logger.info("Frame reduced, writing to disc...")

        write_name = file.replace("crr_", "reduced_")

        header = hdu[0].header

        # if no cropping is to be done in the next step, these parameters
        # allow the full frame to be used
        header["CROPY1"] = 0
        header["CROPY2"] = data.shape[0]

        write_to_fits(data, hdu[0].header, write_name, output_dir)

        logger.info(f"Frame written to directory {output_dir}, filename {write_name}")


def reduce_all():
    """
    Driver for the reduction of all observations (standard star, science, arc lamp)
    in the output directory.
    """

    use_overscan = detector_params["overscan"]["use_overscan"]

    if use_overscan:
        logger.warning("Using overscan subtraction instead of master bias.")
        logger.warning("If this is not intended, check the config file.")

        # get the overscan direction
        overscan_dir = detect_overscan_direction()

        BIAS, FLAT = get_bias_and_flats(skip_bias=True)

        print("BIAS", BIAS)
        print("FLAT", FLAT)

    else:
        overscan_dir = None
        BIAS, FLAT = get_bias_and_flats()

    logger.info(f"Fetching cosmic-ray removed files from {output_dir} ...")

    science_files, standard_files = read_crr_files()

    # Standard star reduction

    if skip_science_or_standard_bool == 1:
        logger.warning("Skipping standard star reduction as requested...")
    else:
        logger.info("Reducing following standard star frames:")

        list_files(standard_files)

        reduce_group(standard_files, BIAS, FLAT, use_overscan, overscan_dir)

    # Science reduction

    if skip_science_or_standard_bool == 2:
        logger.warning("Skipping science reduction as requested...")

    else:
        logger.info("Reducing following science frames:")

        list_files(science_files)

        reduce_group(science_files, BIAS, FLAT, use_overscan, overscan_dir)


if __name__ == "__main__":
    reduce_all()
