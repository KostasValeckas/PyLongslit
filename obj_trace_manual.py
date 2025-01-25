from logger import logger
from astropy.io import fits
from astropy.modeling.models import Gaussian1D
from parser import extract_params, output_dir
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from utils import open_fits, choose_obj_centrum, get_skysub_files
import matplotlib.pyplot as plt
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Const1D
import matplotlib.pyplot as plt
from utils import hist_normalize
from numpy.polynomial.chebyshev import chebfit, chebval
from utils import estimate_sky_regions
from utils import show_1d_fit_QA
import os
from tqdm import tqdm




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

 
    # get polynomial degree for fitting
    fit_deg = extract_params["OBJ_FIT_DEG"]

    # open the file
    hdul = open_fits(output_dir, filename)
    data = hdul[0].data

    header = hdul[0].header
    # get the cropped y offset for global detector coordinates
    y_lower = header["CROPY1"]
    y_upper = header["CROPY2"]

    print("Got cropped values: ", y_lower, y_upper)

    # final containers for the results
    centers = []



    logger.info(f"Finding object in {filename}...")

    # plot the image and let the user hover over the object centers and press 'a' to add and 'd' to delete
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(hist_normalize(data), cmap='gray')
    ax.set_title("Hover over the object centers and press 'a' to add, 'd' to delete last point. Close the plot when done.")
    points = []

    def on_key(event):
        nonlocal points
        if event.key == '+':
            x, y = event.xdata, event.ydata
            points.append((x, y))
            ax.plot(x, y, '+', c = 'r')
            fig.canvas.draw()
        elif event.key == '-':
            #ax.plot(points[-1][0], points[-1][1], 'x', markersize=10, markeredgewidth=2)
            try:
                points.pop()
            except IndexError:
                pass
            ax.clear()
            ax.imshow(hist_normalize(data), cmap='gray')
            ax.set_title("Hover over the object centers and press 'a' to add, 'd' to delete last point. Close the plot when done.")
            for x, y in points:
                ax.plot(x, y, '+', c = 'r')
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
    filename_out = filename.replace("skysub_", "obj_manual_").replace(".fits", ".dat")
    
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
    logger.info("Starting object tracing routine...")

    filenames = get_skysub_files()

    find_obj(filenames)

    logger.info("Manual object tracing routine finished.")
    print("----------------------------\n")


if __name__ == "__main__":
    run_obj_trace()
