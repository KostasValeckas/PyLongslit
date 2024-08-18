import numpy as np
import glob as glob
from astropy.io import fits
from parser import detector_params, output_dir, skip_science_or_standard_bool
from parser import science_params, standard_params, arc_params
from utils import FileList, open_fits, write_to_fits, list_files, hist_normalize
from logger import logger
import matplotlib.pyplot as plt
from overscan import subtract_overscan_from_frame
import os
from matplotlib.patches import Rectangle

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

    arc_files : list
        A list of cosmic-ray removed arc files.
    """

    science_files = []
    standard_files = []
    arc_files = []

    for file in os.listdir(output_dir):
        if file.startswith("crr") or file.startswith("/crr"):
            if "science" in file:
                science_files.append(file)
            elif "std" in file:
                standard_files.append(file)
            elif "arc" in file:
                arc_files.append(file)

    logger.info(f"Found {len(science_files)} cosmic-ray removed science files.")
    logger.info(f"Found {len(standard_files)} cosmic-ray removed standard star files.")
    logger.info(f"Found {len(arc_files)} cosmic-ray removed arc files.")

    # sort alphabetically to correctly match the centers

    science_files.sort()
    standard_files.sort()

    return science_files, standard_files, arc_files


def reduce_frame(frame, master_bias, master_flat, use_overscan):
    """
    Performs overscan subtraction, bias subtraction
    and flat fielding of a single frame.
    """

    if use_overscan:
        frame = subtract_overscan_from_frame(frame)

    logger.info("Subtracting the master bias frame...")

    frame = frame - master_bias

    logger.info("Dividing by the master flat frame...")

    frame = frame / master_flat

    return frame

def reduce_group(file_list, BIAS, FLAT, use_overscan):

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
        
        data = reduce_frame(data, BIAS, FLAT, use_overscan)
        
        logger.info("Frame reduced, writing to disc...")
        
        write_name = file.replace("crr_", "reduced_")
        
        write_to_fits(data, hdu[0].header, write_name, output_dir)
        
        logger.info(
            f"Frame written to directory {output_dir}, filename {write_name}"
        )


def reduce_all():
    """
    Driver for the reduction of all observations (standard star, science, arc lamp)
    in the output directory.
    """

    use_overscan = detector_params["overscan"]["use_overscan"]

    logger.info("Fetching the master bias frame...")

    try:
        BIAS_HDU = open_fits(output_dir, "master_bias.fits")
    except FileNotFoundError:
        logger.critical(f"Master bias frame not found in {output_dir}.")
        logger.error("Make sure you have excecuted the bias procdure first.")
        exit()

    BIAS = BIAS_HDU[0].data

    logger.info("Master bias frame found and loaded.")

    logger.info("Fetching the master flat frame...")

    try:
        FLAT_HDU = open_fits(output_dir, "master_flat.fits")
    except FileNotFoundError:
        logger.critical(f"Master flat frame not found in {output_dir}.")
        logger.error("Make sure you have excecuted the flat procdure first.")
        exit()

    FLAT = FLAT_HDU[0].data

    logger.info("Master flat frame found and loaded.")

    logger.info(f"Fetching cosmic-ray removed files from {output_dir} ...")

    science_files, standard_files, arc_files = read_crr_files()

    # Standard star reduction

    if skip_science_or_standard_bool == 1:
        logger.warning("Skipping standard star reduction as requested...")
    else:
        logger.info("Reducing following standard star frames:")

        list_files(standard_files)

        reduce_group(standard_files, BIAS, FLAT, use_overscan)

    # Science reduction

    if skip_science_or_standard_bool == 2:
        logger.warning("Skipping science reduction as requested...")

    else:
        logger.info("Reducing following science frames:")

        list_files(science_files)

        reduce_group(science_files, BIAS, FLAT, use_overscan)

    # Arc reduction

    logger.info("Reducing following arc frames:")

    list_files(arc_files)

    reduce_group(arc_files, BIAS, FLAT, use_overscan)

    logger.info("All frames reduced and written to disc.")


if __name__ == "__main__":
    reduce_all()
