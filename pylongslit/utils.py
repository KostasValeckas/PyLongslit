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


class PyLongslit_frame:
    def __init__(self, data, sigma, header, name):
        """
        """

        self.data = data
        self.sigma = sigma
        self.header = header
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def path(self):
        from pylongslit.parser import output_dir
        return os.path.join(output_dir, self.name)


    def write_to_disc(self):
        """
        """
        
        from pylongslit.logger import logger

        # Create a PrimaryHDU object to store the data
        hdu_data = fits.PrimaryHDU(self.data, header=self.header)
        
        # Create an ImageHDU object to store the sigma
        hdu_sigma = fits.ImageHDU(self.sigma, name='1-SIGMA ERROR')
        
        # Create an HDUList to combine both HDUs
        hdulist = fits.HDUList([hdu_data, hdu_sigma])
        
        # Write the HDUList to a FITS file
        
        hdulist.writeto(self.path() + ".fits", overwrite=True)
        
        logger.info(f"File written to {self.path()}.fits")


    def show_frame(self, normalize=True, show=True, save=False, skip_sigma=False):
        """
        Show the frame data and sigma as two subfigures.

        Parameters
        ----------
        normalize : bool
            If True, normalize the data for better visualization.

        new_figure : bool
            If True, create a new figure.

        show : bool
            If True, display the plot.
        """

        # normalize to show detail

        if not skip_sigma:

            data = self.data.copy()
            sigma = self.sigma.copy()

            if normalize:
                data = hist_normalize(data)
                sigma = hist_normalize(sigma)

            # create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

            # plot data
            im = ax1.imshow(data, cmap="gray")
            ax1.set_title(f"{self.name} - Data" + (" (normalized)" if normalize else ""))
            ax1.set_xlabel("Pixels")
            ax1.set_ylabel("Pixels")
            fig.colorbar(im, ax=ax1, orientation='vertical')

            # plot sigma
            im = ax2.imshow(sigma, cmap="gray")
            ax2.set_title(f"{self.name} - Sigma" + (" (normalized)" if normalize else ""))
            ax2.set_xlabel("Pixels")
            ax2.set_ylabel("Pixels")
            fig.colorbar(im, ax=ax2, orientation='vertical')

        else:

            data = self.data.copy()

            if normalize:
                data = hist_normalize(data)
            
            plt.imshow(data, cmap="gray")
            plt.title(f"{self.name}" + (" (normalized)" if normalize else ""))
            plt.xlabel("Pixels")
            plt.ylabel("Pixels")



        if save: 
            plt.savefig(self.path() + ".png")

        if show:
            plt.show()

    @classmethod
    def read_from_disc(cls, filename):
        """
        Read the frame data and sigma from a FITS file.

        Parameters
        ----------
        filepath : str
            Path to the FITS file.

        Returns
        -------
        PyLongslit_frame
            An instance of PyLongslit_frame with the read data, sigma, and header.
        """
        from pylongslit.logger import logger
        from pylongslit.parser import output_dir


        filepath = os.path.join(output_dir, filename)

        # Open the FITS file
        with fits.open(filepath) as hdulist:
            # Read the primary HDU (data)
            data = hdulist[0].data
            header = hdulist[0].header

            # Read the image HDU (sigma)
            sigma = hdulist[1].data

        filename = filename.split(".")[0]

        return cls(data, sigma, header, filename)


       





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

        from pylongslit.logger import logger

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

    from pylongslit.logger import logger
    from pylongslit.parser import data_params, flat_params

    logger.info("Opening the first file in the flat directory...")
    # read the names of the flat files from the directory
    file_list = FileList(flat_params["flat_dir"])

    # open the first file in the directory
    raw_flat = open_fits(flat_params["flat_dir"], file_list.files[0])
    logger.info("File opened successfully.")

    data = np.array(raw_flat[data_params["raw_data_hdu_index"]].data)

    norm_data = hist_normalize(data)

    # show the overscan region overlayed on a raw flat frame
    plt.imshow(norm_data, cmap="gray")


def show_frame(
    inp_data, title=None, figsize=(18, 12), normalize=True, new_figure=True, show=True
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

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    file_list = os.listdir(output_dir)

    files = [file for file in file_list if file.startswith(prefixes)]

    if len(files) == 0:
        logger.warning(f"No files found with prefixes {prefixes}.")

    logger.info(f"Found {len(files)} frames:")
    list_files(files)

    return files


def get_skysub_files(only_science=False):
    """
    Wrapper for ´get_file_group´ that returns the filenames of the skysubtracted,
    and performs some sanity checks.

    Returns
    -------
    filenames : list
        A list of filenames of the skysubtracted files.
    """
    from pylongslit.logger import logger

    logger.info("Getting skysubtracted files...")

    filenames = get_file_group("skysub") if not only_science else get_file_group("skysub_science")

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
    from pylongslit.logger import logger
    from pylongslit.parser import output_dir
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

    # assume center is at the maximum of the slice
    center = x[np.argmax(slice)]

    # check if the center is within the expected region (2FWHM from the clicked point)
    if center < clicked_center - 3 * FWHM_AP or center > clicked_center + 3 * FWHM_AP:
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
    sky_left = center - 3 * FWHM_AP
    sky_right = center + 3 * FWHM_AP

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

    ax1.plot(x_data, y_data, "s", color="black", label=legend_label, markersize=14)

    ax1.plot(x_fit_values, y_fit_values, label="Fit", color="red", markersize=16)
    ax1.set_ylabel(y_label, fontsize=14)
    ax1.legend(fontsize=14)

    ax2.plot(x_data, residuals, "x", color="red", label="Residuals")
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel(y_label, fontsize=14)
    ax2.axhline(0, color="black", linestyle="--")
    ax2.legend(fontsize=14)

    # setting the x-axis to be shared between the two plots
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_xticks([])

    fig.suptitle(title, fontsize=18)

    # Enhance tick font size
    ax1.tick_params(axis="both", which="major", labelsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=14)

    plt.show()


def load_spec_data(group="science"):
    """
    Loads the science or standard star spectra from the output directory.

    Parameters
    ----------
    group : str
        The group of files to load.
        Options: "science", "standard".
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    if group != "science" and group != "standard":
        logger.error('The "group" parameter must be either "science" or "standard".')
        exit()

    filenames = get_filenames(
        starts_with="1d_science" if group == "science" else "1d_std",
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

    # make sure we are in the output directory
    os.chdir(output_dir)

    for filename in filenames:
        data = np.loadtxt(filename, skiprows=2)
        wavelength = data[:, 0]
        counts = data[:, 1]
        var = data[:, 2]

        spectra[filename] = (wavelength, counts, var)

    return spectra


def load_fluxed_spec():

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    filenames = get_filenames(starts_with="1d_fluxed_science")

    if len(filenames) == 0:
        logger.error(f"No pectra found.")
        logger.error("Run the flux calibration 1d procedure first.")

        exit()

    # container for the spectra
    spectra = {}

    # make sure we are in the output directory
    os.chdir(output_dir)

    for filename in filenames:
        data = np.loadtxt(filename, skiprows=2)
        wavelength = data[:, 0]
        counts = data[:, 1]
        var = data[:, 2]

        spectra[filename] = (wavelength, counts, var)

    return spectra


def load_bias():

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
    Driver for `get_reduced_frames` that acounts for skip_science and/or
    skip_standard parameters.

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

        reduced_files = get_file_group("reduced_std")

    else:

        reduced_files = get_file_group("reduced_science", "reduced_standard")

    return reduced_files


def wavelength_sol(spectral_pix, spatial_pix, wavelen_fit, tilt_fit):

    tilt_value = tilt_fit(spectral_pix, spatial_pix)
    wavelength = chebval(spectral_pix + tilt_value, wavelen_fit)

    return wavelength
