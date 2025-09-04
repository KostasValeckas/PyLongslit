import numpy
import argparse

"""
Module for creating a master bias frame from raw bias frames.
"""


def run_bias():
    """
    Driver for the bias procedure.

    The function reads the raw bias frames from the directory specified in the
    'bias_dir' parameter in the 'config.json' file. It then stacks the frames and calculates the median value at each
    pixel. The final master bias frame is written to disc in the output directory.
    """

    from pylongslit.parser import detector_params, bias_params, data_params
    from pylongslit.logger import logger
    from pylongslit.utils import FileList, check_dimensions, open_fits
    from pylongslit.utils import PyLongslit_frame
    from pylongslit.overscan import check_overscan, subtract_overscan
    from pylongslit.stats import bootstrap_median_errors_framestack

    # Extract the detector parameters
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    logger.info("Bias procedure running...")
    logger.info("Using the following parameters:")
    logger.info(f"xsize = {xsize}")
    logger.info(f"ysize = {ysize}")

    # read the names of the bias files from the directory
    file_list = FileList(bias_params["bias_dir"])

    logger.info(f"Found {file_list.num_files} bias frames.")
    logger.info(f"Files used for bias processing:")

    file_list.print_files()

    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # initialize a big array to hold all the bias frames for stacking
    bigbias = numpy.zeros((file_list.num_files, ysize, xsize), float)

    use_overscan = check_overscan()
    # loop over all the bias files and stack them in the bigbias array
    for i, file in enumerate(file_list):

        rawbias = open_fits(bias_params["bias_dir"], file)

        logger.info(f"Processing file: {file}")

        # this might result in underflows if unsigned integers are used,
        # dtype should always be set explicitly
        data = numpy.array(rawbias[data_params["raw_data_hdu_index"]].data, dtype=int)

        if use_overscan:
            data = subtract_overscan(data)

        bigbias[i] = data

        # close the file handler
        rawbias.close()

        logger.info(f"File {file} processed.\n")

    # Calculate bias as median at each pixel
    medianbias = numpy.median(bigbias, axis=0)
    if file_list.num_files < 30 and (not bias_params["bootstrap_errors"]):
        logger.warning(
            f"Number of bias frames ({file_list.num_files}) is less than 30. Error estimation might not be accurate."
        )
        logger.warning(
            "Please consider taking more bias frames or activating error bootstrapping in the config file."
        )

    if not bias_params["bootstrap_errors"]:
        medianbias_error = (
            1.2533 * numpy.std(bigbias, axis=0) / numpy.sqrt(file_list.num_files)
        )

    else:
        medianbias_error = bootstrap_median_errors_framestack(bigbias)

    logger.info("Bias frames processed.")

    # Write out result to fitsfile
    hdr = rawbias[0].header

    # create a PyLongslit_frame object to hold the data and header
    master_bias = PyLongslit_frame(medianbias, medianbias_error, hdr, "master_bias")

    (
        master_bias.show_frame()
        if not use_overscan
        else master_bias.show_frame(
            title_addition="Overscan subtracted - therefore negative values are okay."
        )
    )
    master_bias.write_to_disc()
    logger.info("Bias procedure completed.")


def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit bias procedure.")
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_bias()


if __name__ == "__main__":
    main()
