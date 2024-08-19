"""
Utility functions for PyLongslit.

For code that is useful in multiple modules.
"""

from logger import logger
import os
from astropy.io import fits
import numpy as np
from parser import detector_params, flat_params, science_params 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import exposure


class FileList:
    def __init__(self, path):

        """
        A class that reads all filenames from a directory
        and counts them. Made iterable so files can be looped over.

        Parameters
        ----------
        path : str
            The path to the directory containing the files.

        Attributes
        ----------
        path : str
            The path to the directory containing the files.

        files : list
            A list of all filenames in the directory.

        num_files : int
            The number of files in the directory.
        """

        self.path = path

        if not os.path.exists(self.path):
            logger.error(f"Directory {self.path} not found.")
            logger.error(
                "Make sure the directory is provided correctly "
                'in the "config.json" file. '
                "See the docs at:\n"
                "https://kostasvaleckas.github.io/PyLongslit/"
            )
            exit()

        self.files = os.listdir(self.path)

        # sort alphabetically for consistency in naming
        self.files.sort()

        self.num_files = len(self.files)

    def __iter__(self):
        return iter(self.files)


def open_fits(dir_path, file_name):
    """
    A more robust wrapper for 'astropy.io.fits.open'.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing the file.

    file_name : str
        The name of the file to open.

    Returns
    -------
    hdul : HDUList
        An HDUList object containing the data from the file.
    """

    try:
        hdul = fits.open(dir_path + file_name)
    # acount for the user forgetting to add a slash at the end of the path
    except FileNotFoundError:
        hdul = fits.open(dir_path + "/" + file_name)

    return hdul

def write_to_fits(data, header, file_name, path):
    """
    A more robust wrapper for 'astropy.io.fits.writeto'.

    Parameters
    ----------
    data : numpy.ndarray
        The data to write to the file.

    header : Header
        The header to write to the file.

    file_name : str
        The name of the file to write to.
    
    path : str
        The path to the directory to write the file to.
    """

    try:
        fits.writeto(path + "/" + file_name, data, header, overwrite=True)
    # acount for missing slashes in the path
    except FileNotFoundError:
        fits.writeto(path + file_name, data, header, overwrite=True)
        


def check_dimensions(FileList: FileList, x, y):
    """
    Check that dimensions of all files in a FileList match the wanted dimensions.

    Parameters
    ----------
    FileList : FileList
        A FileList object containing filenames.

    x : int
        The wanted x dimension.

    y : int
        The wanted y dimension.

    Returns
    -------
    Prints a message to the logger if the dimensions do not match,
    and exits the program.
    """

    for file in FileList:

        hdul = open_fits(FileList.path, file)

        data = hdul[1].data

        if data.shape != (y, x):
            logger.error(
                f"Dimensions of file {file} do not match the user "
                'dimensions set in the "config.json" file.'
            )
            logger.error(
                f"Expected ({y}, {x}), got {data.shape}."
                f"\nCheck all files in {FileList.path} and try again."
            )
            exit()

        hdul.close()

    logger.info("All files have the correct dimensions.")
    return None

def hist_normalize(data, z_thresh=3):
    """
    Aggresive normalization of used for showing detail in raw frames.

    First performs outlier rejection based on Z-scores and then
    applies histogram equalization.

    Parameters
    ----------
    data : numpy.ndarray
        The data to normalize.

    z_thresh : float
        The Z-score threshold for outlier rejection.

    Returns
    -------
    data_equalized : numpy.ndarray
        The normalized data.
    """


    # Calculate the Z-scores
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std

    # Remove outliers by setting them to the mean or a capped value
    data_no_outliers = np.where(np.abs(z_scores) > z_thresh, mean, data)


    # Now apply histogram equalization
    data_equalized = exposure.equalize_hist(data_no_outliers)

    return data_equalized

def show_flat():
    """
    Shows the first flat-frame in the user defined flat-directory.

    This is used together with ´overscan.show_overscan()´ and
    ´mkspecflat.show_flat_norm_region()´
    for sanity checks of the user defined regions.
    """

    logger.info("Opening the first file in the flat directory...")
    # read the names of the flat files from the directory
    file_list = FileList(flat_params["flat_dir"])

    # open the first file in the directory
    raw_flat = open_fits(flat_params["flat_dir"], file_list.files[0])
    logger.info("File opened successfully.")

    data = np.array(raw_flat[1].data)

    norm_data = hist_normalize(data)

    # show the overscan region overlayed on a raw flat frame
    plt.imshow(norm_data, cmap="gray")



def list_files(file_list: FileList):
    """
    List all files in a FileList object.

    Parameters
    ----------
    file_list : FileList
        A FileList object containing filenames.
    """

    print("------------------------------------")
    for file in file_list:
        print(file)
    print("------------------------------------")
    return None

def check_rotation():
    """
    Check if the raw frames need to be rotated.

    Returns
    -------
    transpose : bool
        If True, the raw frames need to be transposed.

    flip : bool
        If True, the raw frames need to be flipped.
    """

    disp_ax = detector_params["dispersion"]["spectral_dir"]
    disp_dir = detector_params["dispersion"]["wavelength_grows_with_pixel"]

    if disp_ax == "x":
        pass

    elif disp_ax == "y":
        transpose = True

    else:
        logger.error(
            'The "dispersion" key in the "detector" section of the '
            'config.json file must be either "x" or "y".'
        )
        exit()

    if disp_dir == True:
        flip = False

    elif disp_dir == False:
        flip = True

    else:
        logger.error(
            'The "wavelength_grows_with_pixel" key in the "dispersion" '
            'section of the config.json file must be either "true" or "false".'
        )
        exit()

    return transpose, flip

    


def flip_and_rotate(frame_data, transpose, flip):
    """
    The PyLongslit default orientation is dispersion in the x-direction,
    with wavelength increasing from left to right.

    If the raw frames are not oriented this way, this function will
    flip and rotate the frames so they are.

    Parameters
    ----------
    frame_data : numpy.ndarray
        The data to flip and rotate.

    transpose : bool
        If True, transpose the data.

    flip : bool
        If True, flip the data.

    Returns
    -------
    frame_data : numpy.ndarray
        The flipped and rotated data.
    """

    if transpose:
        logger.info("Rotating image to make x the spectral direction...")
        frame_data = np.rot90(frame_data)

    if flip:
        logger.info("Flipping the image to make wavelengths increase with x-pixels...")
        frame_data = np.flip(frame_data, axis=1)

    return frame_data
