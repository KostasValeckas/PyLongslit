from logger import logger
from parser import skip_science_or_standard_bool
from parser import output_dir, extract_params
from utils import list_files, hist_normalize, open_fits
import os
import matplotlib.pyplot as plt
import numpy as np


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


def choose_obj_centrum(file_list, figsize=(18, 12)):
    """
    An interactive method to choose the center of the object on the frame.

    Parameters
    ----------
    file_list : list
        A list of filenames to be reduced.

    figsize : tuple
        The size of the figure to be displayed.
        Default is (18, 12).

    Returns
    -------
    centrum_dict : dict
        A dictionary containing the chosen centers of the objects.
    """

    logger.info("Starting object-choosing GUI. Follow the instructions on the plots.")

    # used for more readable plotting code
    plot_title = "Press on the object on a spectral point with no sky-lines " \
        "(but away from detector edges.) \n" \
        "You can try several times. Press 'q' or close plot when done." 
   
    centrum_dict = {}

    def onclick(event):
        x = int(event.xdata)
        y = int(event.ydata)

        #put the clicked point in the dictionary
        centrum_dict[file] = (x, y)
        
        # Remove any previously clicked points
        plt.cla()
        # the plotting code below is repeated twice, but this is more stable
        # for the event function (it uses non-local variables)
        plt.imshow(norm_data, cmap="gray")
        plt.title(plot_title)
        
        # Color the clicked point
        plt.scatter(x, y, marker = "x", color='red', s=50, label="Selected point")
        plt.legend()
        plt.draw()  # Update the plot

    for file in file_list:
        plt.figure(figsize=figsize)

        frame = open_fits(output_dir, file)
        data = frame[0].data
        norm_data = hist_normalize(data)

        plt.imshow(norm_data, cmap="gray")
        plt.connect('button_press_event', onclick)
        plt.title(plot_title)
        plt.show()

    
    logger.info("Object centers chosen successfully:")
    print(centrum_dict, "\n------------------------------------")

    return centrum_dict

def trace_sky():
    
    NSUM_AP = extract_params["NSUM_AP"]

    print(NSUM_AP)

    pass



def run_extract_1d():
    logger.info("Starting the 1d extraction process...")
    
    reduced_files = get_reduced_frames()
    
    choose_obj_centrum(reduced_files)


if __name__ == "__main__":
    run_extract_1d()
