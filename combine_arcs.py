"""
Module to combine arc frames into a single master arc frame.
"""

from logger import logger
from parser import output_dir, arc_params, data_params, detector_params
from utils import FileList, open_fits, write_to_fits, list_files, get_bias_and_flats
from utils import check_rotation, flip_and_rotate, load_bias
from overscan import subtract_overscan_from_frame, check_overscan
import os
import numpy as np
import matplotlib.pyplot as plt


def combine_arcs():

    logger.info("Fetching arc frames...")

    arc_files = FileList(arc_params["arc_dir"])

    if arc_files.num_files == 0:
        logger.critical("No arc files found.")
        logger.critical("Check the arc directory path in the config file.")

        exit()

    logger.info(f"Found {arc_files.num_files} raw arc files:")
    list_files(arc_files)

    use_overscan = check_overscan()

    logger.info("Subtracting bias...")

    BIASframe = load_bias()
    BIAS = np.array(BIASframe[0].data)

    # container to hold the reduced arc frames
    arc_data = []

    for arc_file in arc_files:

        hdu = open_fits(arc_files.path, arc_file)

        data = hdu[data_params["raw_data_hdu_index"]].data.astype(np.float32)

        if use_overscan:
            data = subtract_overscan_from_frame(data)

        data = data - BIAS

        arc_data.append(data)

    logger.info("Combining arc frames...")

    master_arc = np.sum(arc_data, axis=0)

    # Handle NaNs and Infs
    if np.isnan(master_arc).any() or np.isinf(master_arc).any():
        logger.warning("NaNs or Infs detected in the frame. Replacing with zero.")
        master_arc = np.nan_to_num(master_arc, nan=0.0, posinf=0.0, neginf=0.0)

    # check if the frame needs to be rotated or flipped -
    # later steps rely on x being the dispersion axis
    # with wavelength increasing with pixel number
    transpose, flip = check_rotation()

    # transpose and/or flip the frame if needed
    if transpose or flip:
        master_arc = flip_and_rotate(master_arc, transpose, flip)

    logger.info("Master arc created successfully, writing to disc...")

    # Write the master arc to a FITS file
    write_to_fits(master_arc, hdu[0].header, "master_arc.fits", output_dir)

    logger.info(
        f"Master arc written to disc in directory {output_dir}, filename 'master_arc.fits'."
    )


if __name__ == "__main__":
    combine_arcs()
