"""
Prelimenary check of the configuration file.
Won't crash - but will warn the user if some parameters 
are set in a way that might not be intended.
"""

import os
import argparse


def check_directory(directory, dir_type, error_message, empty_message, any_errors):
    """
    A method for checking if a directory exists and is not empty.

    Used for checking the raw data directories.

    Parameters
    ----------
    directory : str
        The path to the directory to be checked.

    dir_type : str
        A string specifying the type of the directory.

    error_message : str
        A message to be printed if the directory does not exist.

    empty_message : str
        A message to be printed if the directory is empty.

    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.

    Returns
    -------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.
    """

    from pylongslit.logger import logger

    if not os.path.exists(directory):
        logger.warning(f"Raw {dir_type} directory {directory} does not exist.")
        logger.warning(error_message)
        any_errors = True
    else:
        logger.info(f"Raw {dir_type} directory: {directory} was found.")
        if os.listdir(directory):
            logger.info(f"{dir_type.capitalize()} directory is not empty.")
        else:
            logger.info(f"{dir_type.capitalize()} directory is empty.")
            logger.info(empty_message)
            any_errors = True
    return any_errors


def check_dirs(any_errors):
    """
    A driver loop for `check_directory` method.

    Loops over all the raw input directories in the configuration file.

    Parameters
    ----------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.

    Returns
    -------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.
    """

    from pylongslit.parser import bias_params, flat_params
    from pylongslit.parser import science_params, standard_params, arc_params
    from pylongslit.logger import logger


    any_errors = check_directory(
        bias_params["bias_dir"],
        "bias",
        "This will crash the pipeline in first step.",
        "This will crash the pipeline.",
        any_errors,
    )

    any_errors = check_directory(
        flat_params["flat_dir"],
        "flat",
        "You can only do the bias step.",
        "This will crash the pipeline.",
        any_errors,
    )

    any_errors = (
        check_directory(
            science_params["science_dir"],
            "science",
            "If this is an only standard star reduction - set skip_science parameter to 'true' in the config file.",
            "This will crash the pipeline after bias and flats.",
            any_errors,
        )
        if not science_params["skip_science"]
        else any_errors
    )

    any_errors = (
        check_directory(
            standard_params["standard_dir"],
            "standard star",
            "If this is an only science reduction - set skip_standard parameter to 'true' in the config file.",
            "This will crash the pipeline after bias and flats.",
            any_errors,
        )
        if not standard_params["skip_standard"]
        else any_errors
    )

    any_errors = check_directory(
        arc_params["arc_dir"],
        "arc",
        "This will crash the pipeline in wavelength calibration.",
        "This will crash the pipeline.",
        any_errors,
    )

    if not any_errors:
        print("\n")
        logger.info("All input directories are found and not empty.")

    return any_errors


def check_regions(any_errors):
    """
    Check that rectangles defined by the user in the configuration file
    for overscan are valid.

    Parameters
    ----------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.

    Returns
    -------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.
    """

    from pylongslit.parser import detector_params
    from pylongslit.logger import logger

    # check overscan region
    overscan = detector_params["overscan"]
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    if overscan["use_overscan"]:
        if (
            overscan["overscan_x_start"] < 0
            or overscan["overscan_x_end"] > xsize
            or overscan["overscan_y_start"] < 0
            or overscan["overscan_y_end"] > ysize
        ):
            logger.warning("Overscan region is not within the detector area.")
            any_errors = True

        if (
            overscan["overscan_x_end"] < overscan["overscan_x_start"]
            or overscan["overscan_y_end"] < overscan["overscan_y_start"]
        ):
            logger.warning("Overscan region is not defined correctly.")
            logger.warning("End coordinates must be larger than start coordinates.")
            any_errors = True

    else:
        logger.info("Overscan is set to be skipped in configuration file.")

    return any_errors


def check_detector(any_errors):
    """
    Visualize the overscan on a raw flat frame.

    Parameters
    ----------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.

    Returns
    -------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.
    """
    from pylongslit.overscan import show_overscan
    from pylongslit.parser import detector_params
    from pylongslit.logger import logger

    try:

        # show overscan
        if detector_params["overscan"]["use_overscan"]:
            logger.info(
                "Showing the overscan region on a raw flat frame "
                "for user inspection..."
            )
            show_overscan()

        else:
            logger.info("Overscan is set to be skipped in configuration file.")

    # except "catch all" are usually not allowed in the pipeline, but since
    # this is a configuration check, we can allow it
    except:
        logger.warning("Could not show detector regions.")
        logger.warning("Check for any previous warnings.")
        logger.warning(
            "This is most likely caused by missing flat files, "
            "or by bad overscan or flat normalization area definitions "
            "in the configuration file."
        )

        any_errors = True

    return any_errors


def check_for_negative_params(any_errors):
    """
    Checks for negative physical parameters in the configuration file.

    Does not check overscan , but it checkedwavecalib in `check_regions.

    Parameters
    ----------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.

    Returns
    -------
    any_errors : bool
        A boolean used by `run_config_checks` to track if any errors were found.
    """

    from pylongslit.parser import detector_params, bias_params, flat_params
    from pylongslit.parser import science_params, standard_params, arc_params
    from pylongslit.parser import data_params, crr_params, wavecalib_params, extract_params
    from pylongslit.parser import obj_trace_clone_params, sens_params
    from pylongslit.logger import logger

    # TODO re-use this list globally
    params = [
        detector_params,
        bias_params,
        flat_params,
        science_params,
        standard_params,
        arc_params,
        data_params,
        crr_params,
        wavecalib_params,
        extract_params,
        obj_trace_clone_params,
        sens_params,
    ]
    for param in params:
        for key, value in param.items():
            if isinstance(value, (int, float)) and value < 0:
                logger.warning(
                    f"Negative value found for {key} in the configuration file."
                )
                logger.warning("All physical parameters should be positive.")
                any_errors = True
    return any_errors


def run_config_checks():
    """
    A driver function for running all the configuration checks.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import check_science_and_standard

    any_errors = False

    print("\n------------------------------------")

    logger.info("Checking directories...")

    any_errors = check_dirs(any_errors)

    print("\n------------------------------------")

    logger.info("Checking user-defined overscan region...")

    any_errors = check_regions(any_errors)
    print("\n------------------------------------\n")

    logger.info("Showing user-defined overscan region...")

    any_errors = check_detector(any_errors)

    print("\n------------------------------------\n")


    logger.info(
        "Checking reduction parameters..."
    )
    return_code = check_science_and_standard()

    if return_code == 0:
        any_errors = True

    else:
        logger.info("Check complete.")

    print("\n------------------------------------\n")

    logger.info("Checking for negative physical parameters...")

    any_errors = check_for_negative_params(any_errors)

    if any_errors:
        logger.warning("Negative physical parameters found.")
        
    else:
        logger.info("All physical parameters are positive.")

    print("\n------------------------------------\n")

    if any_errors:
        logger.warning("Errors found in the configuration file.")
        logger.warning("Check the warnings.")

    else:
        logger.info("NO ERRORS FOUND.")
        logger.info("CONFIGURATION FILE IS READY FOR PIPELINE EXECUTION.")

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit config-file checker.")
    parser.add_argument('config', type=str, help='Configuration file path')

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_config_checks()

if __name__ == "__main__":

    main()
