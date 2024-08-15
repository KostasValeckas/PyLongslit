"""
Utility functions for PyLongslit.
"""

from logger import logger
import os
from astropy.io import fits
import numpy as np
from parser import detector_params, flat_params
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
        fits.writeto(path + file_name, data, header, overwrite=True)
    # acount for missing slashes in the path
    except FileNotFoundError:
        fits.writeto(path + "/" + file_name, data, header, overwrite=True)


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


def show_overscan():
    """
    Show the user defined ovsercan region.

    Fetches a raw flat frame from the user defined directory
    and displays the overscan region overlayed on it.
    """

    logger.info("Showing the overscan region on a raw flat frame for user inspection...")

    logger.info("Opening the first file in the flat directory...")
    # read the names of the flat files from the directory
    file_list = FileList(flat_params["flat_dir"])

    # open the first file in the directory
    raw_flat = open_fits(flat_params["flat_dir"], file_list.files[0])
    logger.info("File opened successfully.")

    data = np.array(raw_flat[1].data)

    log_data = np.log10(data)

    # show the overscan region overlayed on a raw flat frame
    plt.imshow(log_data, cmap="gray")

    # Add rectangular box to show the overscan region
    width = detector_params["overscan_x_end"] - detector_params["overscan_x_start"]
    height = detector_params["overscan_y_end"] - detector_params["overscan_y_start"]

    rect = Rectangle(
        (detector_params["overscan_x_start"], detector_params["overscan_y_start"]),
        width,
        height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
        linestyle="--",
        label="Overscan Region Limit",
    )
    plt.gca().add_patch(rect)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("Pixels in x-direction")
    plt.ylabel("Pixels in y-direction")
    plt.title(
        "Overscan region overlayed on a raw flat frame with logaritghmic normalization.\n"
        "The overscan region should be dark compared to the rest of the frame.\n"
        "If it is not, check the overscan region definition in the config file."
    )
    plt.show()
