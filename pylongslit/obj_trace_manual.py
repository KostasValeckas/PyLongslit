import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval
import os
import argparse


def find_obj_frame_manual(filename, params, figsize=(10, 6)):
    """
    Manual object finding routine for a single frame.

    Parameters
    ----------
    filename : str
        The filename of the frame to find the object in.

    params : dict
        A dictionary containing the fit parameters - this is the dictionary
        that is fetched directly from the configuration file (['trace']['object']
        or ['trace']['standard'].

    figsize : tuple, optional
        The size of the figure to display the image in. Default is (10, 6).
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir
    from pylongslit.utils import hist_normalize, PyLongslit_frame, check_crr_and_sky
    from pylongslit.obj_trace import fit_distribution_parameter, objmodel_QA

    # open the file
    logger.info(f"Opening file {filename}...")
    frame = PyLongslit_frame.read_from_disc(filename)
    check_crr_and_sky(frame.header, filename)
    data = frame.data
    logger.info(f"File {filename} opened.")

    # plot title here fore more readable code
    title = (
        "Hover over the object centers and press '+' to add, '-' to delete last point,\n"
        "'h' to toggle histogram normalization, 'c' to change colormap, 'q' to skip.\n"
        'Close the plot when done ("q"). If no points are clicked, the frame will be skipped.'
    )

    # plot the image and let the user hover over the object centers and press
    # '+' to add, '-' to delete, 'h' to toggle histogram normalization,
    # 'c' to change colormap
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_norm = True
    colormap = "gray"
    ax.imshow(hist_normalize(data) if hist_norm else data, cmap=colormap)
    ax.set_title(title)

    # disable the default key bindings, so we can use the keys without conflicts
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    points = []

    # this is the interactive part that reacts on every key press
    def on_key(event):
        nonlocal points, hist_norm, colormap
        if event.key == "+":
            x, y = event.xdata, event.ydata
            points.append((x, y))
            ax.plot(x, y, "+", c="r")
            fig.canvas.draw()
        elif event.key == "-":
            try:
                points.pop()
            except IndexError:
                pass
            ax.clear()
            ax.imshow(hist_normalize(data) if hist_norm else data, cmap=colormap)
            ax.set_title(title)
            for x, y in points:
                ax.plot(x, y, "+", c="r")
            fig.canvas.draw()
        elif event.key == "h":
            hist_norm = not hist_norm
            ax.clear()
            ax.imshow(hist_normalize(data) if hist_norm else data, cmap=colormap)
            ax.set_title(title)
            for x, y in points:
                ax.plot(x, y, "+", c="r")
            fig.canvas.draw()
        elif event.key == "c":
            colormap = "viridis" if colormap == "gray" else "gray"
            ax.clear()
            ax.imshow(hist_normalize(data) if hist_norm else data, cmap=colormap)
            ax.set_title(title)
            for x, y in points:
                ax.plot(x, y, "+", c="r")
            fig.canvas.draw()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    # extract the x and y positions of the clicked points
    x_positions = [p[0] for p in points]
    y_positions = [p[1] for p in points]

    if len(x_positions) == 0:
        logger.warning("No object centers clicked. Skipping this frame.")
        return

    logger.info("Fitting object centers...")

    spectral_pixels, centers_fit_pix = fit_distribution_parameter(
        params["use_bspline_obj"],
        "Object Center",
        params,
        x_positions,
        y_positions,
        data,
        filename,
    )

    # make a dummy FWHM array to comply with the interface
    fwhm_fit_pix = np.full_like(centers_fit_pix, params["fwhm_guess"])

    # show the model QA plot

    objmodel_QA(data, params, centers_fit_pix, fwhm_fit_pix, filename, figsize=figsize)

    # write to the file right away, so we don't lose the results - manual
    # tracing is time consuming

    # change to output directory
    os.chdir(output_dir)

    # prepare a filename
    filename_out = filename.replace("reduced_", "obj_").replace(".fits", ".dat")

    with open(filename_out, "w") as f:
        for x, center, fwhm in zip(spectral_pixels, centers_fit_pix, fwhm_fit_pix):
            f.write(f"{x}\t{center}\t{fwhm}\n")

    # close the file
    f.close()

    logger.info(
        f"Object trace results written to directory {output_dir}, filename: {filename_out}."
    )


def find_obj(filenames):
    """
    Driver method for object finding in every frame.

    Loops through the frames and calls `find_obj_frame` for every frame.

    Parameters
    ----------
    center_dict : dict
        A dictionary containing the user clicked object centers.
        Format: {filename: (x, y)}

    Returns
    -------
    obj_dict : dict
        A dictionary containing the results of the object finding routine.
        Format: {filename: (good_x, centers_fit_val, fwhm_fit_val)}
    """

    from pylongslit.logger import logger
    from pylongslit.obj_trace import get_params

    # loop through the files
    for filename in filenames:

        logger.info(f"Finding object in {filename}...")

        params = get_params(filename)

        find_obj_frame_manual(filename, params)

        logger.info(f"Procedure finished for {filename}.")
        print("----------------------------\n")


def run_obj_trace():
    """
    Driver method for the manual object tracing routine.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import get_reduced_frames

    logger.info("Starting object tracing routine...")

    filenames = get_reduced_frames()

    if len(filenames) == 0:
        logger.error("No reduced frames found in the output directory.")
        logger.error("Make sure to run the reduction procedure first.")
        exit()

    find_obj(filenames)

    logger.info("Manual object tracing routine finished.")
    print("----------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit manual object tracing procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_obj_trace()


if __name__ == "__main__":
    main()
