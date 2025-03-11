"""
PyLongslit module to combine arc frames into a single master arc frame.
"""

import numpy as np
import argparse


def combine_arcs():
    """
    Simple procedure to combine arc frames into a single master arc frame.
    Only preprocessing steps are bias subtraction and overscan correction,
    if requested in the config file, and also alligning the frames to have
    the dispersion axis as the x-axis with wavelength increasing with pixel
    number. The frames are simply summed to create the master arc frame.

    The master arc frame is saved to disc as 'master_arc.fits'.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import arc_params, data_params, combine_arc_params
    from pylongslit.utils import FileList, open_fits
    from pylongslit.utils import check_rotation, flip_and_rotate, PyLongslit_frame
    from pylongslit.overscan import estimate_frame_overscan_bias, check_overscan

    logger.info("Fetching arc frames...")

    arc_files = FileList(arc_params["arc_dir"])

    if arc_files.num_files == 0:
        logger.critical("No arc files found.")
        logger.critical("Check the arc directory path in the config file.")

        exit()

    logger.info(f"Found {arc_files.num_files} raw arc files:")
    arc_files.print_files()

    use_overscan = check_overscan()

    skip_bias = combine_arc_params["skip_bias"]

    if not skip_bias:

        logger.info("Fetching bias...")

        try:
            BIAS_frame = PyLongslit_frame.read_from_disc("master_bias.fits")
            BIAS = BIAS_frame.data
        except FileNotFoundError:
            logger.critical("Master bias frame not found.")
            logger.critical("Bias subtraction is requested in the config file.")
            logger.critical("Please create the master bias frame first.")
            exit()

        logger.info("Bias frame loaded successfully.")

    else:
        logger.warning("Skipping bias subtraction in arc combination.")
        logger.warning("This is requested in the config file.")

    # container to hold the reduced arc frames
    arc_data = []

    for arc_file in arc_files:

        print("--------------------------------------------------")

        logger.info(f"Processing arc frame: {arc_file}...")

        hdu = open_fits(arc_files.path, arc_file)

        data = hdu[data_params["raw_data_hdu_index"]].data.astype(np.float32)

        if use_overscan:
            overscan = estimate_frame_overscan_bias(data, plot=False)
            data = data - overscan.data
            logger.info("Overscan subtracted.")

        if not skip_bias:
            data = data - BIAS
            logger.info("Master bias subtracted.")

        arc_data.append(data)

        print("--------------------------------------------------")

    logger.info("Combining arc frames...")

    # we simply sum, as we only are interested in the line positions
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
        logger.info("Alligning the frame to default direction...")
        master_arc = flip_and_rotate(master_arc, transpose, flip)

    master_arc = PyLongslit_frame(master_arc, None, hdu[0].header, "master_arc")

    master_arc.show_frame(skip_sigma=True)

    logger.info("Master arc created successfully, writing to disc...")

    master_arc.write_to_disc()


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit combine-arc procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    combine_arcs()


if __name__ == "__main__":
    main()
