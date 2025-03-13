"""
PyLongslit module for dark current estimation.
"""
import numpy as np
from astropy.io import fits
import os

def check_dark_directory(directory, throw_warning=False):
    """
    Check if the directory contains dark frames.

    If user want dark subtracted from flats, science or stadard star frames, 
    they need to provide dark frames with correct exposure times in the 
    input directories in a directory called "dark".

    Parameters
    ----------
    directory : str
        Directory to check for dark frames.

    throw_warning : bool, optional
        If True, a warning will be thrown if dark frames are not found.
        Defaults to False.

    Returns
    -------
    bool
        True if dark frames are present, False otherwise.
    """

    from pylongslit.logger import logger

    dark_directory_path = os.path.join(directory, "dark")

    logger.info(f"Checking for dark frames in {dark_directory_path}")

    if not os.path.exists(dark_directory_path):
        if throw_warning:
            logger.warning(f"No dark directory found in {directory} - no dark current will be subtracted.")
            logger.warning("If dark current is negligible for your instrument, this is okay.")
            logger.warning("If you wish to subtract dark current, see the documentation on how this is set-up.")
        else:
            logger.info(f"No dark directory found in {directory} - no dark current will be subtracted.")
            logger.info("If dark current is negligible for your instrument, this is okay.")
            logger.info("If you wish to subtract dark current, see the documentation on how this is set-up.")
        return False

    
    return True


def estimate_dark(directory, group):
    """
    Estimate the dark current of the detector for a file group.

    If user wants dark subtracted from flats, science or stadard star frames, 
    they need to provide dark frames with correct exposure times in the 
    input directories in a directory called "dark".

    Parameters
    ----------
    directory : str
        The directory where the dark frame directory is located.

    group : str
        The group of files to estimate the dark current for. Only
        for printing purposes.

    Returns
    -------
    dark_frame : PyLongslit_frame
        The dark frame with the estimated dark current.
    """

    from pylongslit.utils import PyLongslit_frame, FileList, check_dimensions, open_fits
    from pylongslit.parser import detector_params, data_params, developer_params
    from pylongslit.logger import logger
    from pylongslit.overscan import subtract_overscan
    from pylongslit.stats import safe_mean

    # extract the dimensions
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]
    use_overscan = detector_params["overscan"]["use_overscan"]

    logger.info(f"Estimating dark current for {group}...")
    logger.info("Using the following parameters:")
    logger.info(f"xsize = {xsize}")
    logger.info(f"ysize = {ysize}")

    dark_directory = os.path.join(directory, "dark")

    # read the names of the dark files from the directory
    file_list = FileList(dark_directory)

    logger.info(f"Found {file_list.num_files} dark frames for {group}.")
    if developer_params["verbose_print"]:
        logger.info(f"Files used for dark current estimation:")
        file_list.print_files()
    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # initialize a big array to hold all the dark frames for stacking
    bigdark = np.zeros((file_list.num_files, ysize, xsize), float)

    logger.info("Fetching the master bias frame...")
    BIASframe = PyLongslit_frame.read_from_disc("master_bias.fits")
    BIAS = BIASframe.data
    logger.info("Master bias frame found and loaded.")

    print("\n------------------------------------------------------------\n")
    # loop over all the darkfiles, subtract bias and stack them in the bigflat array
    
    for i, file in enumerate(file_list):
        rawdark = open_fits(dark_directory, file)
        if developer_params["verbose_print"]: logger.info(f"Processing file: {file}")
        data = np.array(
            rawdark[data_params["raw_data_hdu_index"]].data, dtype=np.float64
        )
        # Subtract the bias
        if use_overscan:
            data = subtract_overscan(data, print=False)
        data = data - BIAS
        
        if developer_params["verbose_print"]: logger.info("Subtracted the bias.")
        
        bigdark[i] = data
        # close the file handler
        rawdark.close()
        if developer_params["verbose_print"]: logger.info(f"File {file} processed.\n")

    # Calculate dark as a mean at each pixel
    mediandark = np.median(bigdark, axis=0)

    # set nan and inf values to mean value
    mediandark[np.isnan(mediandark)] = safe_mean(mediandark)
    mediandark[np.isinf(mediandark)] = safe_mean(mediandark)

    # set negative values to zero
    mediandark[mediandark < 0] = 0

    # dark has a Poisson noise
    dark_error = np.sqrt(mediandark)

    # create a dummy header
    header = fits.Header()

    # Create a PyLongslit_frame object
    dark_frame = PyLongslit_frame(mediandark, dark_error, header, "dark")

    # show for QA 
    dark_frame.show_frame()
    return dark_frame
