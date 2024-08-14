"""
Utility functions for PyLongslit.
"""

from logger import logger
import os
from astropy.io import fits

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
                "in the \"config.json\" file. "
                "See the docs at:\n"
                "https://kostasvaleckas.github.io/PyLongslit/"
                )
            exit()

        self.files = os.listdir(self.path)

        self.num_files = len(self.files)
        
    def __iter__(self):
        return iter(self.files)


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
        try:
            hdul = fits.open(FileList.path + file)
        except FileNotFoundError:
            hdul = fits.open(FileList.path + "/" + file)

        data = hdul[1].data

        if data.shape != (y, x):
            logger.error(f"Dimensions of file {file} do not match the user "
                         "dimensions set in the \"config.json\" file."
            )
            logger.error(
                f"Expected ({y}, {x}), got {data.shape}."
                f"\nCheck all files in {FileList.path} and try again."
            )
            exit()

        hdul.close()
        
    logger.info("All files have the correct dimensions.")
    return None

    
