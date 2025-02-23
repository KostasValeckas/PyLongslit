"""
Module to combine arc frames into a single master arc frame.
"""

import numpy as np
import argparse



def combine_arcs():

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir, arc_params, data_params, combine_arc_params
    from pylongslit.utils import FileList, open_fits, write_to_fits, list_files
    from pylongslit.utils import check_rotation, flip_and_rotate, load_bias, PyLongslit_frame
    from pylongslit.overscan import estimate_frame_overscan_bias, check_overscan

    logger.info("Fetching arc frames...")

    arc_files = FileList(arc_params["arc_dir"])

    if arc_files.num_files == 0:
        logger.critical("No arc files found.")
        logger.critical("Check the arc directory path in the config file.")

        exit()

    logger.info(f"Found {arc_files.num_files} raw arc files:")
    list_files(arc_files)

    use_overscan = check_overscan()

    skip_bias = combine_arc_params["skip_bias"]

    if not skip_bias:

        logger.info("Fetching bias...")

        BIAS_frame = PyLongslit_frame.read_from_disc("master_bias.fits")
        BIAS = BIAS_frame.data

    else:
        logger.warning("Skipping bias subtraction in arc combination.")
        logger.warning("This is requested in the config file.")

    # container to hold the reduced arc frames
    arc_data = []

    for arc_file in arc_files:

        hdu = open_fits(arc_files.path, arc_file)

        data = hdu[data_params["raw_data_hdu_index"]].data.astype(np.float32)

        if use_overscan:
            overscan = estimate_frame_overscan_bias(data, plot = False)
            data = data - overscan.data

        if not skip_bias: data = data - BIAS

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

    master_arc = PyLongslit_frame(master_arc, None, hdu[0].header, "master_arc")

    master_arc.show_frame(skip_sigma=True, normalize=True)

    logger.info("Master arc created successfully, writing to disc...")

    master_arc.write_to_disc()


def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit combine-arc procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    combine_arcs()


if __name__ == "__main__":
    main()
