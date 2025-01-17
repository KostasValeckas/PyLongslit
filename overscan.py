import numpy
from astropy.io import fits
from logger import logger
from parser import detector_params, bias_params, output_dir
from utils import FileList, check_dimensions, open_fits, write_to_fits
from utils import show_flat
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

"""
Module for subtracting overscan from a frame.
"""


def show_overscan():
    """
    Show the user defined ovsercan region.

    Fetches a raw flat frame from the user defined directory
    and displays the overscan region overlayed on it.
    """

    logger.info(
        "Showing the overscan region on a raw flat frame for user inspection..."
    )

    show_flat()

    # Add rectangular box to show the overscan region
    width = (
        detector_params["overscan"]["overscan_x_end"]
        - detector_params["overscan"]["overscan_x_start"]
    )
    height = (
        detector_params["overscan"]["overscan_y_end"]
        - detector_params["overscan"]["overscan_y_start"]
    )

    rect = Rectangle(
        (
            detector_params["overscan"]["overscan_x_start"],
            detector_params["overscan"]["overscan_y_start"],
        ),
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

def detect_overscan_direction():
    """
    Detect the direction of the overscan region.

    Returns
    -------
    direction : str
        The direction of the overscan region.
        Possible values are "horizontal" or "vertical".
    """

    logger.info("Detecting the direction of the overscan region...")

    # Extract the overscan region
    overscan_x_start = detector_params["overscan"]["overscan_x_start"]
    overscan_x_end = detector_params["overscan"]["overscan_x_end"]
    overscan_y_start = detector_params["overscan"]["overscan_y_start"]
    overscan_y_end = detector_params["overscan"]["overscan_y_end"]

    # Extract detector size in order to calculate whether this is
    # a horizontal or vertical overscan

    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    # Check if the overscan is horizontal or vertical
    # Current implementation only supports horizontal or vertical overscan,
    # not both at the same time. 
    if (overscan_x_end - overscan_x_start) + 1 == xsize:
        logger.info("Horizontal overscan detected.")
        direction = "horizontal"
    elif (overscan_y_end - overscan_y_start) + 1 == ysize:
        logger.info("Vertical overscan detected.")
        direction = "vertical"
    else:
        logger.critical("Overscan region does not match detector size.")
        logger.critical("Check the config file.")
        exit(1)

    return direction

def check_overscan():
    """
    A simple bool return to checck whether the user wants to use the overscan subtraction or not.
    """

    use_overscan = detector_params["overscan"]["use_overscan"]

    if not use_overscan:
        logger.info("Overscan subtraction is disabled.")
        logger.info("Skipping overscan subtraction...")
        return False
    
    return True


def subtract_overscan_from_frame(image_data):
    """
    Subtract the overscan region from a single frame.

    Parameters
    ----------
    input_dir : str
        The directory where the frame is located.

    file : str
        The name of the frame.

    Returns
    -------
    image_data : numpy.ndarray
        The frame with the overscan region subtracted.
    """

    logger.info(f"Subtracting overscan...")

    # Extract the overscan region
    overscan_x_start = detector_params["overscan"]["overscan_x_start"]
    overscan_x_end = detector_params["overscan"]["overscan_x_end"]
    overscan_y_start = detector_params["overscan"]["overscan_y_start"]
    overscan_y_end = detector_params["overscan"]["overscan_y_end"]

    overscan_direction = detect_overscan_direction()

    if overscan_direction == "horizontal":
        # loop through the columns and subtract the mean value of the overscan region
        for i in range(overscan_x_start, overscan_x_end + 1):
            mean = numpy.mean(image_data[overscan_y_start:overscan_y_end, i])
            image_data[:, i] = image_data[:, i] - mean

    elif overscan_direction == "vertical":
        # loop through the rows and subtract the mean value of the overscan region
        for i in range(overscan_y_start, overscan_y_end + 1):
            mean = numpy.mean(image_data[i, overscan_x_start:overscan_x_end])
            image_data[i, :] = image_data[i, :] - mean

    logger.info("Overscan subtracted successfully.")

    return image_data
