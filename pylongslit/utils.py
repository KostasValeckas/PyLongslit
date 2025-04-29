"""
Utility functions for PyLongslit.

For code that is useful in multiple modules.
"""

import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from numpy.polynomial.chebyshev import chebval
from matplotlib.widgets import Slider


class PyLongslit_frame:
    """
    A class to store the data, sigma, header, and name of a frame.
    Allows for easier loading/writing of frames to/from disc.
    """

    def __init__(self, data, sigma, header, name):
        """
        Parameters
        ----------
        data : numpy.ndarray
            The data of the frame.

        sigma : numpy.ndarray
            The 1-sigma error of the frame.

        header : Header
            The header of the frame.

        name : str
            The name of the frame, without the file extension.
        """

        self.data = data
        self.sigma = sigma
        self.header = header
        self.name = name

    # context manager methods - needed when the class is used in a "with" statement
    # does not do anything, just a python-techincality
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def path(self):
        """
        Get the path to the frame, without the file extension.
        """
        from pylongslit.parser import output_dir

        return os.path.join(output_dir, self.name)

    def write_to_disc(self):
        """
        Write the frame data and sigma to a FITS file.
        Writes to the output directory with the name of the frame.
        """

        from pylongslit.logger import logger

        # Create a PrimaryHDU object to store the data
        hdu_data = fits.PrimaryHDU(self.data, header=self.header)

        # Create an ImageHDU object to store the sigma
        hdu_sigma = fits.ImageHDU(self.sigma, name="1-SIGMA ERROR")

        # Create an HDUList to combine both HDUs
        hdulist = fits.HDUList([hdu_data, hdu_sigma])

        # Write the HDUList to a FITS file

        hdulist.writeto(self.path() + ".fits", output_verify="silentfix+ignore", overwrite=True)

        logger.info(f"File written to {self.path()}.fits")

    def show_frame(self, save=False, skip_sigma=False, title_addition=""):
        """
        Show the frame data and sigma as two subfigures (or one
        if the error is to be skipped).

        Parameters
        ----------
        save : bool, optional
            If True, save the plot to a file. Default is False.

        skip_sigma : bool, optional
            If True, skip the error plot. Default is False.

        title_addition : str, optional
            Additional text to add to the title of the plot.
            The default title is the name of the file.
        """

        from pylongslit.logger import logger

        # TODO: a few "hacked" solutions in this method in order for the same
        # method to be able to handle both single and double plots.
        # Consider a refactor if this gives any issues down the road.

        # copy the data and sigma so the original is not modified
        data = self.data.copy()
        if not skip_sigma:
            sigma = self.sigma.copy()

        # these are needed for the interactive plot to avoid repeating colorbars
        self.colorbar1 = None
        self.colorbar2 = None

        def update_plot(normalize):
            """
            The method that actually updates the plot.

            Parameters
            ----------
            normalize : bool
                If True, normalize the data and sigma before plotting.
            """

            # 2 subplots
            if not skip_sigma:

                if (not self.colorbar1 == None) and (not self.colorbar2 == None):
                    self.colorbar1.remove()
                    self.colorbar2.remove()

                if normalize:
                    data_to_plot = hist_normalize(data)
                    sigma_to_plot = hist_normalize(sigma)

                else:
                    data_to_plot = data
                    sigma_to_plot = sigma

                im1 = ax1.imshow(data_to_plot, cmap="gray")
                ax1.set_xlabel("Pixels")
                ax1.set_ylabel("Pixels")
                self.colorbar1 = fig.colorbar(im1, ax=ax1, orientation="vertical")
                ax1.set_title(f"Data" + (" (normalized)" if normalize else ""))

                im2 = ax2.imshow(sigma_to_plot, cmap="gray")
                ax2.set_xlabel("Pixels")
                ax2.set_ylabel("Pixels")
                self.colorbar2 = fig.colorbar(im2, ax=ax2, orientation="vertical")
                ax2.set_title(f"Error" + (" (normalized)" if normalize else ""))

            # 1 subplot
            else:

                # this is a technicality to remove the previous plot if needed
                try:
                    if not ax1 == None:
                        ax1.remove()
                    if not ax2 == None:
                        ax2.remove()
                except KeyError:
                    pass

                if not self.colorbar1 == None:
                    self.colorbar1.remove()

                if normalize:
                    data_to_plot = hist_normalize(data)
                else:
                    data_to_plot = data

                plt.imshow(data_to_plot, cmap="gray")
                self.colorbar1 = plt.colorbar()
                plt.title(f"{self.name}" + (" (normalized)" if normalize else ""))
                plt.xlabel("Pixels")
                plt.ylabel("Pixels")

            # this overwrites the previous plot until the user exits
            if save:
                plt.savefig(self.path() + ".png")

            plt.draw()

        # Orient the plot based on the shape of the data so most of the
        # figure is utilized
        if data.shape[0] > data.shape[1]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        fig.suptitle(f'{self.name} - Press "h" to normalize.\n {title_addition}')
        update_plot(normalize=False)

        # toggle normalization on key press
        def on_key(event):
            if event.key == "h":
                on_key.normalize = not on_key.normalize
                update_plot(normalize=on_key.normalize)

        # start with normalization off
        on_key.normalize = False
        fig.canvas.mpl_connect("key_press_event", on_key)

        plt.show()

        # actually it saves plot every time on update, but only print once on exit
        if save:
            logger.info(f"Saving plot to {self.path()}.png")

    @classmethod
    def read_from_disc(cls, filename):
        """
        Read the frame data and sigma from a FITS file.
        Terminates the program if the file is not found.

        Parameters
        ----------
        filename : str
            The name of the file to read from.

        Returns
        -------
        PyLongslit_frame
            An instance of PyLongslit_frame.
        """
        from pylongslit.logger import logger
        from pylongslit.parser import output_dir

        filepath = os.path.join(output_dir, filename)

        try:
            # Open the FITS file
            with fits.open(filepath) as hdulist:
                # Read the primary HDU (data)
                data = hdulist[0].data.copy()
                header = hdulist[0].header.copy()
                # Read the image HDU (sigma)
                if hdulist[1].data is not None:
                    sigma = hdulist[1].data.copy()
                else:
                    sigma = None
                # ensure that the file is released:
                hdulist.close()
        except FileNotFoundError:
            logger.error(f"File {filename} not found in {output_dir}.")
            logger.error(
                "Check the log prior to this message and the configuration file."
            )
            exit()

        # Remove the file extension, as the class expects it to be without it
        filename = filename.split(".")[0]



        return cls(data, sigma, header, filename)


class FileList:
    """
    A class that reads all filenames from a directory.
    Made iterable so files can be looped over.

    Used in fetching the raw data from designated directories.
    """

    def __init__(self, path):
        """
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

        from pylongslit.logger import logger

        self.path = path

        if not os.path.exists(self.path):
            logger.error(f"Directory {self.path} not found.")
            logger.error(
                "Make sure the directory is provided correctly "
                'in the "configuration" file. '
            )
            exit()

        self.files = os.listdir(self.path)

        # remove the "dark" directory from the list
        self.files = [file for file in self.files if file != "dark"]

        # sort alphabetically for consistency in naming
        self.files.sort()

        self.num_files = len(self.files)

    # allows to loop over the files directly
    def __iter__(self):
        return iter(self.files)

    def print_files(self):
        """
        A simple method for prettier printing of the filenames.
        """

        print("------------------------------------")
        for file in self:
            print(file)
        print("------------------------------------")


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
    from pylongslit.logger import logger

    try:
        try:
            hdul = fits.open(dir_path + file_name)
        # acount for the user forgetting to add a slash at the end of the path
        except FileNotFoundError:
            hdul = fits.open(dir_path + "/" + file_name)
    except FileNotFoundError:
        logger.error(f"File {file_name} not found in {dir_path}.")
        logger.error("Check the file path in the configuration file.")
        exit()

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


def get_filenames(starts_with=None, ends_with=None, contains=None):
    """
    Get a list of filenames from the output directory based on the given criteria.

    Parameters
    ----------
    starts_with : str, optional
        The filenames should start with this string.
        If None, not used.

    ends_with : str, optional
        The filenames should end with this string.
        If None, not used.

    contains : str, optional
        The filenames should contain this string.
        If None, not used.

    Returns
    -------
    filenames : list
        A list of filenames that match the criteria.
    """
    from pylongslit.parser import output_dir

    filenames = os.listdir(output_dir)

    # Initialize sets for each condition
    starts_with_set = (
        set(filenames)
        if starts_with is None
        else {filename for filename in filenames if filename.startswith(starts_with)}
    )
    ends_with_set = (
        set(filenames)
        if ends_with is None
        else {filename for filename in filenames if filename.endswith(ends_with)}
    )
    contains_set = (
        set(filenames)
        if contains is None
        else {filename for filename in filenames if contains in filename}
    )

    # Find the intersection of all sets
    filtered_filenames = starts_with_set & ends_with_set & contains_set

    return list(filtered_filenames)


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
    from pylongslit.logger import logger
    from pylongslit.parser import data_params

    for file in FileList:

        hdul = open_fits(FileList.path, file)

        data = hdul[data_params["raw_data_hdu_index"]].data

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
    print("------------------------------------")
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
    mean = np.nanmean(data)
    std = np.nanstd(data)
    z_scores = (data - mean) / std

    # Remove outliers by setting them to the mean or a capped value
    data_no_outliers = np.where(np.abs(z_scores) > z_thresh, mean, data)

    # Now apply histogram equalization
    data_equalized = exposure.equalize_hist(data_no_outliers)

    # Ensure the data is equalized from 0 to 1
    data_equalized = (data_equalized - np.min(data_equalized)) / (
        np.max(data_equalized) - np.min(data_equalized)
    )

    return data_equalized


def show_frame(
    inp_data, title=None, figsize=(10, 6), normalize=True, new_figure=True, show=True
):
    """
    A wrapper for plt.imshow to avoid repeating code.

    Parameters
    ----------
    data : numpy.ndarray
        The data to plot.

    title : str
        The title of the plot.

    figsize : tuple
        The size of the figure.

    normalize : bool
        If True, normalize the data before plotting.

    new_figure : bool
        If True, create a new figure

    show : bool
        If True, show the plot.
    """

    data = inp_data.copy()

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


def check_rotation():
    """
    Check if the raw frames need to be rotated. This is done
    from the "dispersion" and "detector" sections of the configuration file.

    The default orientation is dispersion in the x-direction,
    with wavelength increasing from left to right.

    Returns
    -------
    transpose : bool
        If True, the raw frames need to be transposed.

    flip : bool
        If True, the raw frames need to be flipped.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import detector_params

    disp_ax = detector_params["dispersion"]["spectral_dir"]
    disp_dir = detector_params["dispersion"]["wavelength_grows_with_pixel"]

    if disp_ax == "x":
        transpose = False

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


def flip_and_rotate(frame_data, transpose, flip, inverse=False):
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

    inverse: bool
        If True, the inverse operation is performed.

    Returns
    -------
    frame_data : numpy.ndarray
        The flipped and rotated data.
    """

    from pylongslit.logger import logger

    if transpose:
        logger.info("Rotating image to make x the spectral direction...")
        frame_data = np.rot90(frame_data) if not inverse else np.rot90(frame_data, k=-1)

    if flip:
        logger.info("Flipping the image to make wavelengths increase with x-pixels...")
        frame_data = np.flip(frame_data, axis=1)

    return frame_data


def get_file_group(*prefixes):
    # TODO: code would be more clean if this returned an isntance of the
    # class FileList, instead of a list of filenames
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
    reduced_files : list of str
        A list of reduced files.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    file_list = os.listdir(output_dir)

    files = [file for file in file_list if file.startswith(prefixes)]

    if len(files) == 0:
        logger.warning(f"No files found with prefixes {prefixes}.")

    logger.info(f"Found {len(files)} frames:")
    for file in files:
        print(file)
    print("------------------------------------")

    return files


def get_skysub_files(only_science=False):
    """
    Wrapper for ´get_file_group´ that returns the filenames of the skysubtracted,
    and performs some sanity checks.

    Parameters
    ----------
    only_science : bool
        If True, only return the science files.

    Returns
    -------
    filenames : list
        A list of filenames of the skysubtracted files.
    """
    from pylongslit.logger import logger

    logger.info("Getting skysubtracted files...")

    filenames = (
        get_file_group("skysub")
        if not only_science
        else get_file_group("skysub_science")
    )

    if len(filenames) == 0:
        logger.error("No skysubtracted files found.")
        logger.error("Make sure you run the sky-subraction routine first.")
        exit()

    # sort as this is needed when cross referencing with traces
    filenames.sort()

    return filenames


def choose_obj_centrum(file_list, titles, figsize=(10, 6)):
    # TODO: titles list is a bit hacky, should be refactored
    """
    An interactive method to choose the center of the object on the frame.

    Parameters
    ----------
    file_list : list
        A list of filenames.

    titles : list
        A list of titles for the plots, matching the file_list.

    figsize : tuple
        The size of the figure to be displayed.
        Default is (10, 6).

    Returns
    -------
    center_dict : dict
        A dictionary containing the chosen centers of the objects.
        Format: {filename: (x, y)}
    """
    from pylongslit.logger import logger
    from pylongslit.utils import PyLongslit_frame

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

        frame = PyLongslit_frame.read_from_disc(file)
        data = frame.data

        plt.figure(figsize=figsize)
        plt.connect("button_press_event", onclick)
        show_frame(data, titles[i], new_figure=False)

    logger.info("Object centers chosen successfully:")
    print(center_dict, "\n------------------------------------")

    return center_dict


def refine_obj_center(x, slice, clicked_center):
    """
    Refine the object center based on the slice of the data.

    Try a simple numerical estimation of the object center, and check
    if it is not a nan value. If it is, return the clicked center.

    Used it in the `trace_sky` method and object tracing procedure in general.

    Parameters
    ----------
    x : array
        The x-axis of the slice

    slice : array
        The slice of the data.

    clicked_center : int
        The center of the object clicked by the user.

    Returns
    -------
    center : int
        The refined object center.
    """

    # assume center is at the maximum of the slice
    center = x[np.argmax(slice)]

    # if the center is a nan value, return the clicked center
    if np.isnan(center):
        return clicked_center

    # if it is an inf value, return the clicked center
    if np.isinf(center):
        return

    return center


def estimate_sky_regions(slice_spec, spatial_center_guess, fwhm_guess, fwhm_thresh):
    """
    From a user inputted object center guess, tries to refine the object centrum,
    and then estimates the sky region around the object.

    The sky is estimated as everything 3 times the FWHM of the object away from the object
    +/- the fwthm threshold.

    Parameters
    ----------
    slice_spec : array
        The slice of the data.

    spatial_center_guess : int
        The user clicked center of the object.

    fwhm_guess : float
        The FWHM of the object guess.

    fwhm_thresh : float
        The threshold for how much the fwhm could deviate from the initial
        guess.

    Returns
    -------
    center : int
        The refined object center.

    sky_left : int
        The left edge of the sky region.

    sky_right : int
        The right edge of the sky region.
    """

    x_spec = np.arange(len(slice_spec))

    center = refine_obj_center(x_spec, slice_spec, spatial_center_guess)

    # QA for sky region selection
    sky_left = center - 3 * fwhm_guess - fwhm_thresh
    sky_right = center + 3 * fwhm_guess + fwhm_thresh

    return int(center), int(sky_left), int(sky_right)


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
    figsize=(10, 6),
):
    """
    A method to plot a 1D fit and residuals for QA purposes.

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

    # RMS of the residuals will be printed in the title
    RMS_residuals = np.sqrt(np.mean(residuals**2))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # fit plot
    ax1.plot(x_data, y_data, ".", color="black", label=legend_label, markersize=8)

    ax1.plot(x_fit_values, y_fit_values, label="Fit", color="red", markersize=10)
    ax1.set_ylabel(y_label, fontsize=10)
    ax1.legend(fontsize=10)

    # residuals plot
    ax2.plot(x_data, residuals, "x", color="red", label="Residuals")
    ax2.set_xlabel(x_label, fontsize=10)
    ax2.set_ylabel(y_label, fontsize=10)
    ax2.axhline(0, color="black", linestyle="--")
    ax2.legend(fontsize=10)

    # setting the x-axis to be shared between the two plots
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_xticks([])

    fig.suptitle(title + f"\n RMS of residuals: {RMS_residuals}", fontsize=12)

    # Enhance tick font size
    ax1.tick_params(axis="both", which="major", labelsize=8)
    ax2.tick_params(axis="both", which="major", labelsize=8)

    plt.show()


# TODO: check if the 2 below methods would be sensible to combine into 1
def load_spec_data(group="science"):
    """
    Loads the science or standard star (unfluxed) spectra from the output directory.

    A wrapper for the `get_filenames` method.

    Parameters
    ----------
    group : str
        The group of files to load.
        Options: "science", "standard".

    Returns
    -------
    spectra : dict
        A dictionary containing the spectra.
        Format: {filename: (wavelength, counts,variance)}
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    if group != "science" and group != "standard":
        logger.error('The "group" parameter must be either "science" or "standard".')
        exit()

    filenames = get_filenames(
        starts_with="1d_science" if group == "science" else "1d_standard",
    )

    if len(filenames) == 0:
        logger.error(f"No {group} spectra found.")
        logger.error("Run the extract 1d procedure first.")
        logger.error(
            f'If you have already run the procedure, check the "skip_{group}" parameter in the config file.'
        )
        exit()

    # container for the spectra
    spectra = {}

    # get current working directory
    cwd = os.getcwd()

    # make sure we are in the output directory
    os.chdir(output_dir)

    for filename in filenames:
        data = np.loadtxt(filename, skiprows=2)
        wavelength = data[:, 0]
        counts = data[:, 1]
        var = data[:, 2]

        spectra[filename] = (wavelength, counts, var)

    # change back to the original directory
    # this is useful when the user uses relative pathes in the configuration file
    os.chdir(cwd)

    return spectra


def load_fluxed_spec():
    """
    Loads the science or standard star (fluxed) spectra from the output directory.

    A wrapper for the `get_filenames` method.

    Returns
    -------
    spectra : dict
        A dictionary containing the spectra.
        Format: {filename: (wavelength, counts,variance)}
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Loading the fluxed spectra...")

    filenames = get_filenames(starts_with="1d_fluxed_science")

    if len(filenames) == 0:
        logger.error(f"No spectra found.")
        logger.error("Run the flux calibration 1d procedure first.")

        exit()

    # container for the spectra
    spectra = {}

    # get current working directory
    cwd = os.getcwd()

    # make sure we are in the output directory
    os.chdir(output_dir)

    for filename in filenames:
        data = np.loadtxt(filename, skiprows=2)
        wavelength = data[:, 0]
        counts = data[:, 1]
        var = data[:, 2]

        spectra[filename] = (wavelength, counts, var)

    # change back to the original directory
    # this is useful when the user uses relative pathes in the configuration file
    os.chdir(cwd)

    return spectra


def load_bias():
    """
    NOT USED
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    try:

        BIASframe = open_fits(output_dir, "master_bias.fits")

    except FileNotFoundError:

        logger.critical("Master bias frame not found.")
        logger.error(
            "Make sure a master bias frame exists before proceeding with flats."
        )
        logger.error("Run the mkspecbias.py script first.")
        exit()

    return BIASframe


def get_bias_and_flats(skip_bias=False):
    """
    NOT USED
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    if not skip_bias:

        logger.info("Fetching the master bias frame...")

        try:
            BIAS_HDU = open_fits(output_dir, "master_bias.fits")
        except FileNotFoundError:
            logger.critical(f"Master bias frame not found in {output_dir}.")
            logger.error("Make sure you have excecuted the bias procdure first.")
            exit()

        BIAS = BIAS_HDU[0].data

        logger.info("Master bias frame found and loaded.")

    else:
        BIAS = None

    logger.info("Fetching the master flat frame...")

    try:
        FLAT_HDU = open_fits(output_dir, "master_flat.fits")
    except FileNotFoundError:
        logger.critical(f"Master flat frame not found in {output_dir}.")
        logger.error("Make sure you have excecuted the flat procdure first.")
        exit()

    FLAT = FLAT_HDU[0].data

    logger.info("Master flat frame found and loaded.")

    return BIAS, FLAT


def get_reduced_frames(only_science=False):
    """
    A wrapper for the `get_file_group` method that acounts for skip_science and/or
    skip_standard parameters.

    Parameters
    ----------
    only_science : bool
        If True, only return the science files.

    Returns
    -------
    reduced_files : list
        A list of the reduced files.
    """
    from pylongslit.logger import logger
    from pylongslit.parser import skip_science_or_standard_bool

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

        reduced_files = get_file_group("reduced_science")

    # used only when standard is to be skipped for some step, but not the whole reduction
    elif only_science:
        reduced_files = get_file_group("reduced_science")

    elif skip_science_or_standard_bool == 2:

        logger.warning("Science extraction is set to be skipped in the config file.")
        logger.warning("Will only extract standard star spectra.")

        reduced_files = get_file_group("reduced_standard")

    else:

        reduced_files = get_file_group("reduced_science", "reduced_standard")

    if len(reduced_files) == 0:
        logger.warning("No reduced files found.")
        logger.error("Run the reduction procedure first.")
        exit()

    # sort as this is needed when cross referencing with traces
    reduced_files.sort()

    return reduced_files


def wavelength_sol(spectral_pix, spatial_pix, wavelen_fit, tilt_fit):
    """
    Method for evaluating the wavelength solution for a given pixel.

    Parameters
    ----------
    spectral_pix : float
        The spectral pixel.

    spatial_pix : float
        The spatial pixel.

    wavelen_fit : array
        The wavelength fit coefficients for the Chebyshev polynomial.

    tilt_fit : array
        The 2D tilt fit coefficients. (see the wavelength calibration procedure)

    Returns
    -------
    wavelength : float
        The evaluated wavelength in Angstroms.
    """

    # first evaluate the tilt value and append it to the spectral pixel
    tilt_value = tilt_fit(spectral_pix, spatial_pix)
    wavelength = chebval(spectral_pix + tilt_value, wavelen_fit)

    return wavelength


def interactively_crop_spec(
    x,
    y,
    x_label: str = "",
    y_label: str = "",
    label: str = "",
    title: str = "",
    figsize=(10, 6),
):
    """
    A method for interactively cropping spectrum edges for noise.

    Parameters
    ----------
    x : array-like
        The x data.

    y : array-like
        The y data.

    x_label : str, optional
        The label for the x-axis.

    y_label : str, optional
        The label for the y-axis.

    label : str, optional
        The label for the plot legend.

    title : str, optional
        The title of the plot.

    figsize : tuple, optional
        The size of the figure.

    Returns
    -------
    min_val : int
        The minimum x value.

    max_val : int
        The maximum x value.
    """
    # Initial plot
    fig, _ = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)
    (l,) = plt.plot(x, y, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if label not in [None, ""]:
        plt.legend()
    plt.title(title)

    # Add sliders for selecting the range
    axcolor = "lightgoldenrodyellow"
    axmin = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    smin = Slider(
        axmin,
        "Min " + x_label,
        np.min(x),
        np.max(x),
        valinit=np.min(x),
    )
    smax = Slider(
        axmax,
        "Max " + x_label,
        np.min(x),
        np.max(x),
        valinit=np.max(x),
    )

    # this function will be called when the sliders are moved
    def update(val):
        min = smin.val.copy()
        max = smax.val.copy()
        valid_indices = (x >= min) & (x <= max)
        l.set_xdata(x[valid_indices])
        l.set_ydata(y[valid_indices])
        fig.canvas.draw_idle()

    smin.on_changed(update)
    smax.on_changed(update)

    plt.show()

    return int(smin.val), int(smax.val)


def plot_1d_spec_interactive_limits(
    x,
    y,
    y_error=None,
    x_label: str = "",
    y_label: str = "",
    label: str = "",
    title: str = "",
    figsize=(10, 6),
):
    """
    Plot a 1D spectrum with interactive sliders to adjust the x and y-axis limits.

    Parameters
    ----------
    x : array-like
        The x data.
    y : array-like
        The y data.
    y_error : array-like, optional
        The error in the y data (1 sigma).
    x_label : str, optional
        The label for the x-axis.
    y_label : str, optional
        The label for the y-axis.
    label : str, optional
        The label for the plot legend.
    title : str, optional
        The title of the plot.
    figsize : tuple, optional
        The size of the figure.
    """

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)
    (l,) = ax.plot(x, y, label=label, color="black")
    if y_error is not None:
        (l_error,) = ax.plot(
            x, y_error, label=f"{label} Noise - 1 $\sigma$", color="red"
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)

    # Add sliders for selecting the range
    axcolor = "lightgoldenrodyellow"
    axmin = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    smin = Slider(
        axmin,
        "Min " + x_label,
        np.min(x),
        np.max(x),
        valinit=np.min(x),
    )
    smax = Slider(
        axmax,
        "Max " + x_label,
        np.min(x),
        np.max(x),
        valinit=np.max(x),
    )

    # this function will be called when the sliders are moved
    def update(val):
        min_val = smin.val
        max_val = smax.val

        # Ensure the sliders do not pass each other
        if min_val >= (max_val - 10):
            smin.set_val(np.min(x))
            smax.set_val(np.max(x))
            return

        valid_indices = (x >= min_val) & (x <= max_val)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([0, 1.1 * np.max(y[valid_indices])])
        l.set_xdata(x[valid_indices])
        l.set_ydata(y[valid_indices])
        if y_error is not None:
            l_error.set_xdata(x[valid_indices])
            l_error.set_ydata(y_error[valid_indices])
        fig.canvas.draw_idle()

    smin.on_changed(update)
    smax.on_changed(update)

    plt.show()


def check_crr_and_sky(header, filename):
    """
    Checks if the cosmic ray removal and sky subtraction have been performed
    on the reduced frame. Used in post-reduce procedures.

    Does not return anything, but logs a warning if the procedures have not been performed.

    Parameters
    ----------
    header : Header
        The header of the reduced frame.

    filename : str
        The name of the file.
    """
    from pylongslit.logger import logger

    if not header["CRRREMOVD"]:
        logger.warning(f"{filename} has not been cosmic ray removed.")
        logger.warning("This may affect the quality of the object trace.")
        logger.warning(
            "Consider running the cosmic ray removal routine - but continuing for now..."
        )

    if not header["SKYSUBBED"] and not header["BCGSUBBED"]:
        logger.warning(f"{filename} has not been sky subtracted.")
        logger.warning("This may affect the quality of the object trace.")
        logger.warning(
            "Consider running the sky subtraction or A-B background subtraction routines - but continuing for now..."
        )
