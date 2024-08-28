"""
Utility functions for PyLongslit.

For code that is useful in multiple modules.
"""

from logger import logger
import os
from astropy.io import fits
import numpy as np
from parser import detector_params, flat_params, science_params, output_dir
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


def show_frame(
    data, title, figsize=(18, 12), normalize=True, new_figure=True, show=True
):
    """
    This method is used to plot any frames passed the `reduce`
    procedure. It is used for visual inspection of the data.
    It assumes all data passed to it as aligned in a certain
    direction (this is done in the `reduce` procedure). Data is
    normalized before plotting.

    Parameters
    ----------
    data : numpy.ndarray
        The data to plot.

    title : str
        The title of the plot.

    figsize : tuple
        The size of the figure.

    new_figure : bool
        If True, create a new figure

    show : bool
        If True, show the plot.
    """

    # normalize to show detail
    if normalize:
        data = hist_normalize(data)

    # start the figure

    if new_figure:
        plt.figure(figsize=figsize)

    plt.imshow(data, cmap="gray")
    plt.title(title)
    plt.xlabel("Pixels in spectral direction")
    plt.ylabel("Pixels in spatial direction")
    if show:
        plt.show()


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


def get_file_group(*prefixes):
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

    files = [file for file in file_list if file.startswith(prefixes)]

    if len(files) == 0:
        logger.warning(f"No files found with prefixes {prefixes}.")

    logger.info(f"Found {len(files)} frames:")
    list_files(files)

    return files

def get_skysub_files():
    """
    Wrapper for ´get_file_group´ that returns the filenames of the skysubtracted,
    and performs some sanity checks.

    Returns
    -------
    filenames : list
        A list of filenames of the skysubtracted files.
    """

    logger.info("Getting skysubtracted files...")

    filenames = get_file_group("skysub")

    if len(filenames) == 0:
        logger.error("No skysubtracted files found.")
        logger.error("Make sure you run the sky-subraction routine first.")
        exit()

    # sort as this is needed when cross referencing with traces
    filenames.sort()

    return filenames



def choose_obj_centrum(file_list, titles, figsize=(18, 12)):
    # TODO: titles list is a bit hacky, should be refactored
    """
    An interactive method to choose the center of the object on the frame.

    Parameters
    ----------
    file_list : list
        A list of filenames to be reduced.

    titles : list
        A list of titles for the plots, matching the file_list.

    figsize : tuple
        The size of the figure to be displayed.
        Default is (18, 12).

    Returns
    -------
    center_dict : dict
        A dictionary containing the chosen centers of the objects.
        Format: {filename: (x, y)}
    """

    logger.info("Starting object-choosing GUI. Follow the instructions on the plots.")

    # cointainer ti store the clicked points - this will be returned
    center_dict = {}

    # this is the event we connect to the interactive plot
    def onclick(event):
        x = int(event.xdata)
        y = int(event.ydata)

        # put the clicked point in the dictionary
        center_dict[file] = (x, y)

        # Remove any previously clicked points
        plt.cla()

        show_frame(data, titles[i], new_figure=False, show=False)

        # Color the clicked point
        plt.scatter(x, y, marker="x", color="red", s=50, label="Selected point")
        plt.legend()
        plt.draw()  # Update the plot

    # loop over the files and display the interactive plot
    for i, file in enumerate(file_list):

        frame = open_fits(output_dir, file)
        data = frame[0].data

        plt.figure(figsize=figsize)
        plt.connect("button_press_event", onclick)
        show_frame(data, titles[i], new_figure=False)

    logger.info("Object centers chosen successfully:")
    print(center_dict, "\n------------------------------------")

    return center_dict


def refine_obj_center(x, slice, clicked_center, FWHM_AP):
    """
    Refine the object center based on the slice of the data.

    Try a simple numerical estimation of the object center, and check
    if it is within the expected region. If not, use the clicked point.

    Used it in the `trace_sky` method.

    Parameters
    ----------
    x : array
        The x-axis of the slice.

    slice : array
        The slice of the data.

    clicked_center : int
        The center of the object clicked by the user.

    FWHM_AP : int
        The FWHM of the object.

    Returns
    -------
    center : int
        The refined object center.
    """

    logger.info("Refining the object center...")

    # assume center is at the maximum of the slice
    center = x[np.argmax(slice)]

    # check if the center is within the expected region (2FWHM from the clicked point)
    if center < clicked_center - 2 * FWHM_AP or center > clicked_center + 2 * FWHM_AP:
        logger.warning("The estimated object center is outside the expected region.")
        logger.warning("Using the user-clicked point as the center.")
        logger.warning(
            "This is okay if this is on detector edge or a singular occurence."
        )
        center = clicked_center

    return center


def estimate_sky_regions(slice_spec, spatial_center_guess, FWHM_AP):
    # TODO - modify returns - choose between return obj or return sky
    # or maybe return just the sky_left, sky_righrt and let other modules
    # take care of the rest
    """
    From a user inputted object center guess, tries to refine the object centrum,
    and then estimates the sky region around the object.

    Parameters
    ----------
    slice_spec : array
        The slice of the data.

    spatial_center_guess : int
        The user clicked center of the object.

    FWHM_AP : int
        The FWHM of the object.

    Returns
    -------
    x_spec : array
        The x-axis of the slice.

    x_sky : array
        The x-axis of the sky region.

    sky_val : array
        The values of the sky region.

    sky_left : int
        The left boundary of the sky region.

    sky_right : int
        The right boundary of the sky region.
    """

    x_spec = np.arange(len(slice_spec))

    center = refine_obj_center(x_spec, slice_spec, spatial_center_guess, FWHM_AP)

    # QA for sky region selection
    sky_left = center - 2 * FWHM_AP
    sky_right = center + 2 * FWHM_AP

    return center, sky_left, sky_right


def show_1d_fit_QA(
    x_data,
    y_data,
    x_fit_values=None,
    y_fit_values=None,
    residuals=None,
    x_label=None,
    y_label=None,
    legend_label=None,
    title=None,
    figsize=(18, 12),
):
    """
    A method to plot the 1D fit and residuals for QA purposes.

    Parameters
    ----------
    x_data : array
        The x-axis data.
    
    y_data : array
        The y-axis data.

    x_fit_values : array, optional
        The x-axis values of the evaluated fit.

    y_fit_values : array, optional
        The y-axis values of the evaluated fit.

    residuals : array, optional
        The residuals of the fit.

    x_label : str, optional
        The x-axis label.

    y_label : str, optional
        The y-axis label.

    legend_label : str, optional
        The label for the data.

    title : str, optional
        The title of the plot.

    figsize : tuple, optional
        The size of the figure.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(
        x_data,
        y_data,
        "x",
        color="green",
        label=legend_label,
    )

    ax1.plot(x_fit_values, y_fit_values, label="Fit")
    ax1.set_ylabel(y_label)
    ax1.legend()

    ax2.plot(x_data, residuals, "x", color="red", label = "Residuals")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.axhline(0, color="black", linestyle="--")
    ax2.legend()

    # setting the x-axis to be shared between the two plots
    ax1.set_xlim(ax2.get_xlim())
    ax1.set_xticks([])

    fig.suptitle(title)

    plt.show()
