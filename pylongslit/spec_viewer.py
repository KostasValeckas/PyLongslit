"""
PyLongslit module for displaying the 2D spectrum.
"""

import matplotlib.pyplot as plt
import argparse


def make_2dspec(frame, wavelength_map, figsize=(10, 6)):
    """ "
    Display the 2D spectrum with interactive wavelength display and histogram normalization toggle.

    Parameters
    ----------
    frame : PyLongslit_frame
        The frame to be displayed.

    wavelength_map : numpy.ndarray
        The wavelength map constructed from the wavelength solution.

    figsize : tuple, optional
        The size of the figure. Default is (10, 6).
    """
    from pylongslit.utils import hist_normalize
    from pylongslit.parser import developer_params

    # copy the data and error to avoid altering the original frame
    image = frame.data.copy()
    header = frame.header.copy()

    # these are needed to know where to crop the wavelength map,
    # since the wavelength map covers the full CCD
    cropy1 = header["CROPY1"]
    cropy2 = header["CROPY2"]

    if developer_params["verbose_print"]:
        print(f"crop1: {cropy1}, crop2: {cropy2}")

    wavemap_cut = wavelength_map[cropy1:cropy2, :]

    # now, plot the 2D spectrum with interactive wavelength display and
    # histogram normalization toggle

    # set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img_display = ax.imshow(image, cmap="gray")
    ax.set_xlabel("Spectral pixel")
    # remove the y axis
    ax.get_yaxis().set_visible(False)

    # Function to display wavelength on hover
    def on_hover(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= y < wavemap_cut.shape[0] and 0 <= x < wavemap_cut.shape[1]:
                wavelength = wavemap_cut[y, x]
                ax.set_title(f"Wavelength: {wavelength:.2f} Å")
                fig.canvas.draw_idle()

    # Connect the hover event to the function
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    normalized = False

    # Function to toggle histogram normalization
    def toggle_histogram_normalization(event):
        nonlocal normalized
        if event.key == "h":
            normalized = not normalized
            nonlocal image
            if normalized:
                img_display.set_data(hist_normalize(image))
                img_display.set_clim(0, 1)
            else:
                img_display.set_data(image)
                img_display.set_clim(image.min(), image.max())

            fig.canvas.draw_idle()

    # Connect the key press event to the function
    fig.canvas.mpl_connect("key_press_event", toggle_histogram_normalization)

    fig.suptitle(
        f"2D Spectrum for {frame.name}.\n Hoover cursor over the image to see the wavelength.\n"
        "Press 'h' to toggle histogram normalization."
    )

    plt.show()


def run_2dspec():
    """
    Driver function to run the 2D spectrum viewer.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import get_reduced_frames, PyLongslit_frame
    from pylongslit.wavecalib import (
        get_tilt_fit_from_disc,
        get_wavelen_fit_from_disc,
        construct_wavelen_map,
    )

    # get the files and the wavelength solution
    logger.info("Starting the 2D-Spectrum viewer...")

    logger.info("Fetching the reduced frames...")
    reduced_files = get_reduced_frames()
    if len(reduced_files) == 0:
        logger.error(
            "No reduced frames found. Please run the reduction procedure first."
        )
        exit()

    logger.info("Loading the wavelength solution...")
    wavelen_fit = get_wavelen_fit_from_disc()
    tilt_fit = get_tilt_fit_from_disc()
    wavelength_map = construct_wavelen_map(wavelen_fit, tilt_fit)

    # loop through the reduced files and construct the 2D spectrum
    for file in reduced_files:

        logger.info(f"Constructing 2D spectrum for frame {file}...")

        frame = PyLongslit_frame.read_from_disc(file)

        make_2dspec(frame, wavelength_map)


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit 2 d spectrum construction."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_2dspec()


if __name__ == "__main__":
    main()
