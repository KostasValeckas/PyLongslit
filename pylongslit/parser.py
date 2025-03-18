"""
This script reads the config file and loads the user parameters.
It is executed every time a pipeline routine is run. 
The path to the config file is set in the __init__.py file.
"""

import json
from pylongslit.logger import logger
import os
from pylongslit import CONFIG_FILE_PATH


# Open the config file
try:
    file = open(CONFIG_FILE_PATH, "r")
except FileNotFoundError:

    logger.error("Config file not found.")
    logger.error(
        "Make sure a config file exists. \n"
        "See the docs at:\n"
        "https://kostasvaleckas.github.io/PyLongslit/"
    )

    exit()

logger.info("Config file found. Loading user parameters...")

data = json.load(file)

file.close()

# Define parameter groups for easier access
try:
    detector_params = data["detector"]
    data_params = data["data"]
    bias_params = data["bias"]
    flat_params = data["flat"]
    output_dir = data["output"]["out_dir"]
    instrument_params = data["instrument"]
    crr_params = data["crr_removal"]
    background_params = data["background_sub"]
    science_params = data["science"]
    standard_params = data["standard"]
    arc_params = data["arc"]
    wavecalib_params = data["wavecalib"]
    sky_params = data["sky"]
    trace_params = data["trace"]
    obj_trace_clone_params = data["obj_trace_clone"]
    sens_params = data["sensfunc"]
    flux_params = data["flux_calib"]
    combine_arc_params = data["combine_arcs"]
    combine_params = data["combine"]
    developer_params = data["developer"]

except KeyError:
    logger.error(
        "Config file is not formatted correctly. "
        "Check the example config files at: \n"
        "https://kostasvaleckas.github.io/PyLongslit/"
    )
    exit()

logger.info("User parameters loaded successfully.")

if not os.path.exists(output_dir):
    logger.info(f"Output directory {output_dir} not found. Creating...")
    try:
        os.makedirs(output_dir)
    except OSError:
        logger.error(f"Creation of the directory {output_dir} failed")
        logger.error(
            "Check if you have the necessary permissions, and if the path in the config file is correct."
        )
        exit()
else:
    logger.info(f"Output directory {output_dir} found.")

# Check if the user wants to skip the science or standard star reduction
from pylongslit.check_config import check_science_and_standard

skip_science_or_standard_bool = check_science_and_standard()
