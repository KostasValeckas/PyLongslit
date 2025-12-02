"""
This script reads the config file and loads the user parameters.
It is executed every time a pipeline routine is run. 
The path to the config file is set in the __init__.py file.
"""

import json
from pylongslit.logger import logger
import os
from pathlib import Path
from pylongslit import CONFIG_FILE_PATH

def read_config_dir(config_path):
    """
    Reads the config files from the specified directory.

    Parameters
    ----------
    config_path : Path
        Path to the config files directory.

    Returns
    -------
    dict
        Dictionary containing the user parameters.
    """
    merged_data = {}
    for config_file in config_path.glob("*.json"):
        logger.info(f"Found config file: {config_file.name}")
        with open(config_file, "r") as file:
            data = json.load(file)

            for key, value in data.items():
                if key in merged_data:
                    logger.error(f"Key '{key}' found in multiple config files. Please ensure all keys are unique.")
                    raise KeyError(f"Duplicate key '{key}' in config files.")

            merged_data.update(data)

            logger.info(f"Loaded config file: {config_file.name}")
    return merged_data

# Open the config file
try:
    config_file_path = Path(CONFIG_FILE_PATH)
    if not config_file_path.exists():
        logger.error("Config file not found.")
        logger.error(
            "Make sure a config file exists. \n"
            "See the docs at:\n"
            "https://kostasvaleckas.github.io/PyLongslit/"
        )
        raise FileNotFoundError(f"Config file {CONFIG_FILE_PATH} does not exist")
    elif config_file_path.is_file():
        file = open(CONFIG_FILE_PATH, "r")
        logger.info("Config file found. Loading user parameters...")
        data = json.load(file)
        file.close()
    elif config_file_path.is_dir():
        data = read_config_dir(config_file_path)
except Exception as e:

    logger.error(f"Config file Error.{e}")
    logger.error(
        "Make sure a config file exists and everything is correct. \n"
        "See the docs at:\n"
        "https://kostasvaleckas.github.io/PyLongslit/"
    )

    exit()

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
