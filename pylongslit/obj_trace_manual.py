import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval
import os
import argparse

def find_obj_frame_manual(filename, FWHM_AP):
    """
    Driver method for finding an object in a single frame.

    First, uses `find_obj_one_column` to find the object in every
    column of the detector image.

    Then, uses `interactive_adjust_obj_limits` to interactively adjust the object limits.

    Finally, fits a Chebyshev polynomial to the object centers and FWHMs,
    and shows QA for the results.

    Parameters
    ----------
    filename : str
        The filename of the observation.

    spacial_center : float
        The user-guess for the object center.

    FWHM_AP : float
        The user-guess for the FWHM of the object.

    Returns
    -------
    good_x : array
        The spectral pixel array.

    centers_fit_pix : array
        The fitted object centers.

    fwhm_fit_pix : array
        The fitted FWHMs.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import extract_params, output_dir
    from pylongslit.utils import open_fits, hist_normalize, show_1d_fit_QA, PyLongslit_frame

 
    # get polynomial degree for fitting
    fit_deg = extract_params["OBJ_FIT_DEG"]

    # open the file
    frame = PyLongslit_frame.read_from_disc(filename)
    data = frame.data

    header = frame.header
    # get the cropped y offset for global detector coordinates
    y_lower = header["CROPY1"]
    y_upper = header["CROPY2"]

    print("Got cropped values: ", y_lower, y_upper)

    # final containers for the results
    centers = []

    logger.info(f"Finding object in {filename}...")

    # plot the image and let the user hover over the object centers and press 'a' to add, 'd' to delete, 'h' to toggle histogram normalization, 'c' to change colormap
    fig, ax = plt.subplots(figsize=(10, 8))
    hist_norm = True
    colormap = 'gray'
    ax.imshow(hist_normalize(data) if hist_norm else data, cmap=colormap)
    ax.set_title("Hover over the object centers and press '+' to add, '-' to delete last point, 'h' to toggle histogram normalization, 'c' to change colormap. Close the plot when done.")
    points = []

    def on_key(event):
        nonlocal points, hist_norm, colormap
        if event.key == '+':
            x, y = event.xdata, event.ydata
            points.append((x, y))
            ax.plot(x, y, '+', c='r')
            fig.canvas.draw()
        elif event.key == '-':
            try:
                points.pop()
            except IndexError:
                pass
            ax.clear()
            ax.imshow(hist_normalize(data) if hist_norm else data, cmap=colormap)
            ax.set_title("Hover over the object centers and press 'a' to add, 'd' to delete last point, 'h' to toggle histogram normalization, 'c' to change colormap. Close the plot when done.")
            for x, y in points:
                ax.plot(x, y, '+', c='r')
            fig.canvas.draw()
        elif event.key == 'h':
            hist_norm = not hist_norm
            ax.clear()
            ax.imshow(hist_normalize(data) if hist_norm else data, cmap=colormap)
            ax.set_title("Hover over the object centers and press 'a' to add, 'd' to delete last point, 'h' to toggle histogram normalization, 'c' to change colormap. Close the plot when done.")
            for x, y in points:
                ax.plot(x, y, '+', c='r')
            fig.canvas.draw()
        elif event.key == 'c':
            colormap = 'viridis' if colormap == 'gray' else 'gray'
            ax.clear()
            ax.imshow(hist_normalize(data) if hist_norm else data, cmap=colormap)
            ax.set_title("Hover over the object centers and press 'a' to add, 'd' to delete last point, 'h' to toggle histogram normalization, 'c' to change colormap. Close the plot when done.")
            for x, y in points:
                ax.plot(x, y, '+', c='r')
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # extract the x and y positions of the clicked points
    x_positions = [p[0] for p in points]
    y_positions = [p[1] for p in points]


    
    logger.info("Fitting object centers...")

    centers_fit = chebfit(x_positions, y_positions, deg=fit_deg)

    spectral_pixels = np.arange(data.shape[1])


    centers_fit_val = chebval(x_positions, centers_fit)
    full_fit = chebval(spectral_pixels, centers_fit)

    # residuals
    resid_centers = y_positions - centers_fit_val


    show_1d_fit_QA(
        x_positions,
        y_positions,
        x_fit_values=spectral_pixels,
        y_fit_values=full_fit,
        residuals=resid_centers,
        x_label="Spectral pixel",
        y_label="Spatial pixel",
        legend_label="Manually clicked object centers",
        title=f"Center finding QA for {filename}.\n Ensure the fit is good and residuals are random."
        "\nIf not, adjust the fit parameters in the config file.",
    )

    # make a dummy FWHM array to comply with the interface
    fwhm_fit_val = np.full_like(full_fit, FWHM_AP)

    # change to output directory
    os.chdir(output_dir)

    # write to the file right away, so we don't lose the results - manual 
    # tracing is time consuming

    # prepare a filename
    filename_out = filename.replace("reduced_", "obj_manual_").replace(".fits", ".dat")
    
    with open(filename_out, "w") as f:
        for x, center, fwhm in zip(spectral_pixels, full_fit, fwhm_fit_val):
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
    from pylongslit.parser import extract_params

    # extract the user-guess for the FWHM of the object
    FWHM_AP = extract_params["FWHM_AP"]

    # this is the container for the results
    obj_dict = {}

    # loop through the files
    for filename in filenames:
        logger.info(f"Finding object in {filename}...")

        find_obj_frame_manual(filename, FWHM_AP)

        logger.info(f"Object found in {filename}.")
        print("----------------------------\n")



def run_obj_trace():
    """
    Driver method for the object tracing routine.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import get_reduced_frames

    logger.info("Starting object tracing routine...")

    filenames = get_reduced_frames()

    find_obj(filenames)

    logger.info("Manual object tracing routine finished.")
    print("----------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit manual object tracing procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_obj_trace()


if __name__ == "__main__":
    main()
