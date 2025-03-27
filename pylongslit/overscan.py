import numpy
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

"""
PyLongslit module for handling overscan bias.
"""


def show_overscan(figsize=(10, 6)):
    """
    Show the user defined ovsercan region.

    Fetches a raw flat frame from the user defined directory
    and displays the overscan region overlayed on it.

    Does not crash if no flats are present, but simply returns without showing
    anything.

    Parameters
    ----------
    figsize : tuple, optional
        The size of the figure to display. Default is (10, 6).
    """
    from pylongslit.logger import logger
    from pylongslit.parser import data_params, flat_params, detector_params
    from pylongslit.utils import FileList, open_fits, hist_normalize

    logger.info(
        "Showing the overscan region on a raw flat frame for user inspection..."
    )

    logger.info("Trying to open the first file in the flat directory...")
    # read the names of the flat files from the directory
    file_list = FileList(flat_params["flat_dir"])

    if len(file_list.files) == 0:
        logger.error("No files found in the flat directory.")
        logger.error("Please check the configuration file.")
        logger.error("Can't show overscan region without a flat frame. Skipping...")
        return

    # open the first file in the directory
    raw_flat = open_fits(flat_params["flat_dir"], file_list.files[0])
    logger.info("File opened successfully.")

    # get the data and normalize
    data = np.array(raw_flat[data_params["raw_data_hdu_index"]].data)
    norm_data = hist_normalize(data)

    plt.figure(figsize=figsize)

    # show the overscan region overlayed on a raw flat frame
    plt.imshow(norm_data, cmap="gray")

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
        "Overscan region overlayed on a raw (normalized) flat frame.\n"
        "The overscan region should be dark compared to the rest of the frame.\n"
        "If it is not, check the overscan region definition in the config file.\n"
        "Remember that the overscan subtraction is optional, and can be disabled in the configuration file."
    )
    plt.show()

    return


def detect_overscan_direction():
    """
    NOT USED

    Detect the direction of the overscan region.

    Returns
    -------
    direction : str
        The direction of the overscan region.
        Possible values are "horizontal" or "vertical".
    """

    from pylongslit.logger import logger
    from pylongslit.parser import detector_params

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
    NOT USED

    A simple bool return to checck whether the user wants to use the overscan subtraction or not.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import detector_params

    use_overscan = detector_params["overscan"]["use_overscan"]

    if not use_overscan:
        logger.info("Overscan subtraction is disabled.")
        logger.info("Skipping overscan subtraction...")
        return False

    return True


def estimate_frame_overscan_bias(image_data, plot=False, print=True):
    """
    Estimate the overscan bias of a frame.

    Parameters
    ----------
    input_dir : pylongslit.utils.PyLongslit_frame
        The directory where the frame is located.

    plot : bool, optional
        Whether to plot the estimated overscan bias. Default is False.

    print : bool, optional
        Whether to print logger messages. Default is True.

    Returns
    -------
    overscan_frame : pylongslit.utils.PyLongslit_frame
        The overscan bias frame
    """

    from pylongslit.logger import logger
    from pylongslit.parser import detector_params
    from pylongslit.utils import PyLongslit_frame

    if print: logger.info("Estimating the overscan bias...")

    # Extract the overscan region
    overscan_x_start = detector_params["overscan"]["overscan_x_start"]
    overscan_x_end = detector_params["overscan"]["overscan_x_end"]
    overscan_y_start = detector_params["overscan"]["overscan_y_start"]
    overscan_y_end = detector_params["overscan"]["overscan_y_end"]

    # create a frame with same shape as the input frame, and fill up the
    # mean of the overscan region. As error, take the error of the mean
    # for a Gaussian distribution.
    overscan_image = numpy.zeros_like(image_data)
    error_image = numpy.zeros_like(image_data)

    overscan = image_data[
        overscan_y_start:overscan_y_end, overscan_x_start:overscan_x_end
    ]
    mean = numpy.mean(overscan)
    overscan_image = numpy.full(image_data.shape, mean)
    error = numpy.std(overscan) / numpy.sqrt(overscan.size)
    error_image = numpy.full(image_data.shape, error)

    # construct the overscan frame and return it
    overscan_frame = PyLongslit_frame(
        overscan_image, error_image, None, "overscan_bias"
    )

    if plot:
        overscan_frame.show_frame()

    return overscan_frame


def subtract_overscan(data, print = True):
    """
    Subtract the overscan bias from the frame.

    Uses 'estimate_frame_overscan_bias' to estimate the overscan bias
    and subtracts it from the frame.

    Parameters
    ----------
    data : numpy array
        The data to subtract the overscan bias from.

    print : bool, optional
        Whether to print logger messages. Default is True. 

    Returns
    -------
    data : numpy array
        The data with the overscan bias subtracted.
    """

    from pylongslit.logger import logger

    overscan_bias = estimate_frame_overscan_bias(data, plot=False, print=print)
    if print: logger.info("Subtracting overscan bias from the frame.")
    data = data - overscan_bias.data

    return data
