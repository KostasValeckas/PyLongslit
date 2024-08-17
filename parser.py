import json
from logger import logger
import os


# Open the config file
try:
    file = open("config.json")
except FileNotFoundError:

    logger.error("Config file not found.")
    logger.error("Make sure a \"config.json\" file exists. \n"
                "See the docs at:\n"
                "https://kostasvaleckas.github.io/PyLongslit/")
    
    exit()
    
logger.info("Config file found. Loading user parameters...")

data = json.load(file)

# Define parameter groups for easier access

detector_params = data["detector"]
bias_params = data["bias"]
flat_params = data["flat"]
output_dir = data["output"]["out_dir"]

if not os.path.exists(output_dir):
    logger.info(f"Output directory {output_dir} not found. Creating...")
    os.makedirs(output_dir)

logger.info("User parameters loaded successfully.") 

crr_params = data["crr_removal"]
science_params = data["science"]
standard_params = data["standard"]
arc_params = data["arc"]



def check_science_and_standard():
    """
    Sanity checks for whether the user wants to use science frames, 
    standard star frames, or both for the given run.

    Warns the user if one or the other is missing.

    Terminates the program if both are missing.

    Returns
    -------
    A return code : int
        0 - skip standard star and science reductions 
        (only bias and flats are possible).

        1 - skip standard star reduction.

        2 - skip science reduction.

        3 - skip none.    
    """

    skip_science = science_params["skip_science"]
    skip_standard = standard_params["skip_standard"]

    # check for unreasonable user input

    if skip_science and skip_standard:
        logger.warning(
            "Both skip_science and skip_standard are set to \"true\" "
            "in the config file. Only bias and flat operations can be performed"
        )
        logger.warning("Pipeline will crash if you proceed beyond flats.")
        return 0

    if skip_standard:
        logger.warning(
            "Standard star reduction is set to be skipped in the config file. "
            "This is okay if this is your intention - only the science frames "
            "will be reduced."

        )

        return 1
    
    if skip_science:
        logger.warning(
            "Science reduction is set to be skipped in the config file. "
            "This is okay if this is your intention - only standard star "
            "will be reduced."
        )

        return 2
    
    else:

        return 3
    


skip_science_or_standard_bool = check_science_and_standard()



# Create a dictionary to hold the file information
obs_file_dict = {}

if skip_science_or_standard_bool == 0: pass

else:    
    # Get the list of files in the science directory
    if skip_science_or_standard_bool == 2:
        logger.info("Skipping science frames.")
        science_files = []
    else:
        science_dir = science_params["science_dir"]
        science_files = os.listdir(science_dir)
    
    obs_file_dict["science"] = science_files

    # Get the list of files in the standard directory
    if skip_science_or_standard_bool == 1:
        logger.info("Skipping standard star frames.")
        standard_files = []
    else:
        standard_dir = standard_params["standard_dir"]
        standard_files = os.listdir(standard_dir)

    obs_file_dict["standard_star"] = standard_files

    # Get the list of files in the arc directory

    arc_dir = arc_params["arc_dir"]
    arc_files = os.listdir(arc_dir)

    obs_file_dict["arc_lamp"] = arc_files

