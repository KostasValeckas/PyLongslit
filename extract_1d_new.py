from logger import logger
from parser import skip_science_or_standard_bool
from parser import output_dir
from utils import list_files
import os


def get_reduced_group(*prefixes):
    """
    Helper method to retrieve the names of the 
    reduced frames (science or standard) from the output directory.

    Parameters
    ----------
    prefixes : str
        Prefixes of the files to be retrieved.
        Example: "reduced_science", "reduced_std"

    Returns
    -------
    reduced_files : list
        A list of reduced files.
    """

    file_list = os.listdir(output_dir)
    
    reduced_files = [
        file for file in file_list if file.startswith(prefixes)
    ]
    
    logger.info(f"Found {len(reduced_files)} frames:")
    list_files(reduced_files)
    
    return reduced_files

def get_reduced_frames():
    """ 
    Driver for `get_reduced_frames` that acounts for skip_science or/and 
    skip_standard parameters.

    Returns
    -------
    reduced_files : list
        A list of the reduced files.
    """
    if skip_science_or_standard_bool == 0:
        logger.error(
            "Both skip_science and skip_standard parameters are set to true "
            "in the configuration file."
        )
        logger.error("No extraction can be performed. Exitting...")
        exit()

    elif skip_science_or_standard_bool == 1:

        logger.warning(
            "Standard star extraction is set to be skipped in the config file."
        )
        logger.warning("Will only extract science spectra.")

        reduced_files = get_reduced_group("reduced_science")


    elif skip_science_or_standard_bool == 2:
            
        logger.warning(
            "Science extraction is set to be skipped in the config file."
        )
        logger.warning("Will only extract standard star spectra.")

        reduced_files = get_reduced_group("reduced_std")

    else:

        reduced_files = get_reduced_group("reduced_science", "reduced_std")

    return reduced_files



def run_extract_1d():
    reduced_files = get_reduced_frames()


if __name__ == "__main__":
    run_extract_1d()
