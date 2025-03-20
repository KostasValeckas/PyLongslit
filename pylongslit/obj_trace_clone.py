import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.modeling.models import Gaussian1D
from astropy.stats import gaussian_fwhm_to_sigma


def read_obj_trace_results():
    """
    Method for loading the archived object trace from disk. The path
    to the file is fetched from the configuration file.

    Returns
    -------
    pixel : numpy.ndarray
        The pixel values of the archived object trace.

    center : numpy.ndarray
        The center values of the archived object trace.

    fwhm : numpy.ndarray
        The FWHM values of the archived object trace.
    """
    from pylongslit.parser import obj_trace_clone_params
    from pylongslit.logger import logger

    file_path = obj_trace_clone_params["archived_spec_root"]

    logger.info(f"Loading the archived object trace from {file_path}...")

    try:
        pixel, center, fwhm = np.loadtxt(file_path, unpack=True)
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        logger.error("Check the configuration file.")
        exit()

    if pixel is None or center is None or fwhm is None:
        logger.error("Error reading the archived object trace.")
        logger.error("Cheeck the file format.")
        logger.error("The file should contain three columns: pixel, center, fwhm.")
        exit()

    logger.info("Archived object trace loaded.")

    return pixel, center, fwhm


def load_frame():
    """
    Wrapper method for 'pylongslit.utils.PyLongslit_frame.read_from_disc'
    to load the user defined frame for object trace cloning. The
    path to the file is fetched from the configuration file.

    Returns
    -------
    frame : pylongslit.utils.PyLongslit_frame
        The frame for object trace cloning.
    """

    from pylongslit.utils import PyLongslit_frame, check_crr_and_sky
    from pylongslit.parser import obj_trace_clone_params
    from pylongslit.logger import logger

    logger.info(
        f"Loading the frame for object trace cloning from {obj_trace_clone_params['frame_root']}..."
    )

    file_path = obj_trace_clone_params["frame_root"]

    frame = PyLongslit_frame.read_from_disc(file_path)

    check_crr_and_sky(frame.header, frame.name)

    return frame


def construct_obj_model(frame_data, params, center, fwhm):
    """
    Method for constructing the object model based on the fitting model
    defined in the configuration file.

    Called by the `overlay_trace` method on every key press event, so
    no logging performed in this method.

    Parameters
    ----------
    frame_data : numpy.ndarray
        The frame data for which the object model is to be constructed.

    params : dict
        The configuration parameters.

    center : numpy.ndarray
        The center values of the object trace.

    fwhm : numpy.ndarray
        The FWHM values of the object trace.
    """

    from pylongslit.logger import logger
    from pylongslit.obj_trace import Cauchy1D

    obj_model = np.zeros_like(frame_data)

    fitting_model = params["model"]

    if fitting_model != "Gaussian" and fitting_model != "Cauchy":
        logger.error("Invalid fitting model defined in the configuration file.")
        logger.error("Valid models are: Gaussian, Cauchy.")
        exit()

    spatial_array = np.arange(frame_data.shape[0])

    # loop through the pixel values and build the model
    for i in range(len(center)):
        # set amplitude to 1 for simplicity
        if fitting_model == "Gaussian":
            obj_model[:, i] = Gaussian1D.evaluate(
                spatial_array, 1, center[i], fwhm[i] * gaussian_fwhm_to_sigma
            )
        elif fitting_model == "Cauchy":
            # Cauchy FWHM is defined as 2 * gamma
            obj_model[:, i] = Cauchy1D.evaluate(
                spatial_array, 1, center[i], fwhm[i] / 2.0
            )

    return obj_model


def overlay_trace(pixel, center, fwhm, frame, figsize=(10, 6)):
    """ "
    Method for overlaying the object trace on the frame data and adjusting
    the center and FWHM interactively."

    Parameters
    ----------
    pixel : numpy.ndarray
        The spectral pixel values of the object trace.

    center : numpy.ndarray
        The center values of the object trace.

    fwhm : numpy.ndarray
        The FWHM values of the object trace.

    frame : pylongslit.utils.PyLongslit_frame
        The frame for object trace cloning.

    figsize : tuple
        The size of the figure.

    Returns
    -------
    center : numpy.ndarray
        The adjusted center values of the object trace.

    fwhm : float
        The adjusted FWHM value of the object trace.
    """

    from pylongslit.logger import logger
    from pylongslit.obj_trace import get_params

    # extract the needed frame data
    filename = frame.name
    data = frame.data

    # get the parameters

    params = get_params(filename)

    # start with adjusting the center

    # create the interactive plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data, origin="lower")
    (line,) = ax.plot(pixel, center, label="Object Trace", color="red")

    plt.title(
        "Use arrow keys (up/down for 1 pixel, left/right for 0.1 pixel) to move the center trace.\n"
        'Close the plot "q" when done.'
    )
    plt.legend()

    # disable the default key bindings, so we can use the arrow keys
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    # Event handler function - reacts on every key press
    def on_key(event):
        nonlocal center
        if event.key == "up":
            center += 1  # Move the aptrace up by 1 pixel
        elif event.key == "down":
            center -= 1  # Move the aptrace down by 1 pixel
        elif event.key == "right":
            center += 0.1
        elif event.key == "left":
            center -= 0.1
        elif event.key == "q":
            plt.close(fig)
        line.set_ydata(center)  # Update the plot
        fig.canvas.draw()  # Redraw the figure

    # Connect the event handler to the figure
    fig.canvas.mpl_connect("key_press_event", on_key)

    logger.info("Starting the center trace adjustment...")

    plt.show()

    # now adjust the FWHM

    # create the object model
    obj_model = construct_obj_model(data, params, center, fwhm)

    # create the interactive plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

    ax1.imshow(data, cmap="cool", label="Detector image")
    ax1.set_title("Detector image.")

    image1 = ax2.imshow(obj_model, cmap="hot", label="Object model")
    ax2.set_title("Object model.")

    ax3.imshow(data, cmap="cool")
    image2 = ax3.imshow(obj_model, cmap="hot", alpha=0.3)
    ax3.set_title("Object model overlayed on detector image.")

    # disable the default key bindings, so we can use the arrow keys
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    fig.suptitle(
        "Use arrow keys (up/down for 1 pixel, left/right for 0.1 pixel) to adjust the FWHM of the object model.\n"
        'Close the plot "q" when done.'
    )

    fig.text(0.5, 0.04, "Spectral pixels", ha="center", va="center", fontsize=12)
    fig.text(
        0.04,
        0.5,
        "Spacial pixels",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    # Event handler function - reacts on every key press
    def on_key(event):
        nonlocal fwhm
        if event.key == "up":
            fwhm += 1
        elif event.key == "down":
            if 0 in (fwhm - 1):
                pass
            else:
                fwhm -= 1
        elif event.key == "right":
            fwhm += 0.1
        elif event.key == "left":
            if 0 in (fwhm - 0.1):
                pass
            else:
                fwhm -= 0.1
        elif event.key == "q":
            plt.close(fig)
        obj_model = construct_obj_model(data, params, center, fwhm)
        image1.set_data(obj_model)
        image2.set_data(obj_model)
        fig.canvas.draw()  # Redraw the figure

    # Connect the event handler to the figure
    fig.canvas.mpl_connect("key_press_event", on_key)

    logger.info("Starting the FWHM adjustment...")

    plt.show()

    logger.info("Cloned object model created")

    return center, fwhm


def write_cloned_trace_to_file(pixel, center, fwhm):

    from pylongslit.logger import logger
    from pylongslit.parser import obj_trace_clone_params

    filename = obj_trace_clone_params["frame_root"]

    output_file = filename.replace("reduced_", "obj_").replace(".fits", ".dat")

    logger.info("Writing the cloned object trace to disk...")
    logger.info(f"Output file: {output_file}")

    fwhm_array = np.full_like(center, fwhm)

    # write to the file
    with open(output_file, "w") as f:
        for x, center, fwhm in zip(pixel, center, fwhm_array):
            f.write(f"{x}\t{center}\t{fwhm}\n")

    logger.info("Cloned trace written to disk.")


def run_obj_trace_clone():
    """ "
    Driver method for the object trace cloning procedure. This allows using
    an archived object trace to clone the object trace in a new frame and
    adjust the center and FWHM interactively.
    """
    from pylongslit.logger import logger

    logger.info("Starting the object trace cloning procedure...")

    pixel, center, fwhm = read_obj_trace_results()

    frame = load_frame()

    # for initial fwhm guess, just use the mean of the fwhm archived trace
    corrected_centers, fwhm = overlay_trace(pixel, center, fwhm, frame)

    write_cloned_trace_to_file(pixel, corrected_centers, fwhm)

    logger.info("Object trace cloning procedure finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit cloned object trace procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_obj_trace_clone()


if __name__ == "__main__":
    main()
