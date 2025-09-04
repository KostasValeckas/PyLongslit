import os
import numpy as np
from photutils.aperture import RectangularAperture
import matplotlib.pyplot as plt
import argparse


def plot_trace_QA(image, pixel, trace, fwhm, filename, num_plots=6, figsize=(10, 6)):
    """
    A plotting method that shows the rectangular aperture around the object trace
    that will be used for the extraction.

    Parameters
    ----------
    image : numpy.ndarray
        The reduced frame image.

    pixel : numpy.ndarray
        The pixel coordinate of the object trace.

    trace : numpy.ndarray
        The object trace.

    fwhm : numpy.ndarray
        The full width at half maximum of the object trace.

    filename : str
        The filename of the reduced frame to extract the object from.

    num_plots : int
        The number of plots to show. Default is 6.

    figsize : tuple
        The figure size. Default is (10, 6).
    """
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize)

    # chop the image into segments for easier viewing
    segment_length = image.shape[1] // num_plots

    for i, ax in enumerate(axes):
        start = i * segment_length
        end = (i + 1) * segment_length if i < (num_plots - 1) else image.shape[1]

        segment = image[:, start:end]
        segment_pixel = pixel[start:end]
        segment_trace = trace[start:end]
        segment_fwhm = fwhm[start:end]

        ax.imshow(segment, origin="lower", cmap="gray", aspect="auto")
        ax.plot(
            segment_pixel - start,
            segment_trace - segment_fwhm,
            "-",
            color="red",
            alpha=0.5,
            linewidth=0.5,
        )
        ax.plot(
            segment_pixel - start,
            segment_trace + segment_fwhm,
            "-",
            color="red",
            alpha=0.5,
            linewidth=0.5,
        )

        # set the x-ticks so the user can trace where on the spectral
        # this segment is located - 5 ticks per segment
        x_tick_array = np.arange(
            segment_pixel[0], segment_pixel[-1], segment_length // 5, dtype=int
        )
        ax.set_xticks(np.arange(0, len(segment_pixel), segment_length // 5))
        ax.set_xticklabels(x_tick_array)

    fig.suptitle(
        f"Object aperture QA for {filename}. For simple extraction, the aperture is center +/- FWHM.\n"
        f"If the aperture is not correct, re-run the object trace procedure with a different FWHM parameters in the configuration file."
    )

    fig.text(0.5, 0.04, "Spectral Pixels", ha="center", va="center", fontsize=12)
    fig.text(
        0.04,
        0.5,
        "Spacial Pixels",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    plt.show()


def extract_object_simple(trace_data, trace_params, filename):
    """
    A simple 1D extraction method that sums the counts in a rectangular aperture
    around the object trace. The rectangle had has a height of +/- FWHM
    around the center of the object trace. This is useful when the object is
    close to other objects or detector artifacts.

    Parameters
    ----------
    trace_data : tuple
        A tuple of the shape:
        (pixel, center, FWHM)
        where pixel is the pixel coordinate of the object trace,
        center is the center of the object trace, and
        FWHM is the full width at half maximum of the object trace.

    trace_params : dict
        A dictionary of the trace parameters. Not used, but kept to keep
        the same function signature as the other extraction methods.

    filename : str
        The filename of the reduced frame to extract the object from.

    Returns
    -------
    pixel : numpy.ndarray
        The pixel coordinate of the object trace.

    spec : numpy.ndarray
        The extracted spectrum.

    spec_var : numpy.ndarray
        The variance of the extracted spectrum.

    y_offset : float
        The y-offset from the cropping procedure. This is used in wavelength calibration
        to match the global pixel coordinates with the wavelength
        solution.
    """

    from pylongslit.utils import PyLongslit_frame, check_crr_and_sky

    # unpack the trace data
    pixel, center, FWHM = trace_data

    # Open the reduced frame and extract needed data
    frame = PyLongslit_frame.read_from_disc(filename)
    check_crr_and_sky(frame.header, filename)

    reduced_data = frame.data.copy()
    data_error = frame.sigma.copy()
    header = frame.header.copy()

    # the y-offset from the cropping procedure. This is used in wavelength calibration
    # to match the global pixel coordinates with the wavelength solution. We
    # extract it here as we do not want to handle header data in the wavelength calibration.
    y_offset = header["CROPY1"]

    # these are the containers that will be filled for every value
    spec = []
    spec_var = []

    # first, plot the aperture box on the image for QA purposes
    plot_trace_QA(reduced_data, pixel, center, FWHM, filename)

    # the extraction loop for every spectral pixel
    # this builds a box around the center +/- fwhm of the object and sums the counts
    for i in range(len(center)):

        obj_center = center[i]
        pixel_coord = pixel[i]
        fwhm = FWHM[i]

        # define the aperture
        aperture = RectangularAperture((pixel_coord, obj_center), 1, 2 * fwhm)

        # extract the spectrum
        spec_sum = aperture.do_photometry(reduced_data, error=data_error)

        # extract the counts and the error
        spec_sum_counts = spec_sum[0][0]
        spec_err_counts = spec_sum[1][0]

        spec.append(spec_sum_counts)
        spec_var.append(spec_err_counts**2)

    # convert to numpy arrays for further processing
    spec = np.array(spec)
    spec_var = np.array(spec_var)

    return pixel, spec, spec_var, y_offset


def run_extract_1d_simple():

    from pylongslit.utils import get_reduced_frames
    from pylongslit.logger import logger
    from pylongslit.extract_1d import (
        load_object_traces,
        extract_objects,
        write_extracted_1d_to_disc,
    )

    logger.warning(
        "This is the simple 1D extraction procedure, it extracts a rectangular aperture +/- fwhm around the center of the object trace."
    )
    logger.warning(
        "This is useful when the object is close to other objects or detector artifacts, but is not the most accurate method."
    )
    logger.warning(
        "For more accurate extraction, use the regular 1D extraction procedure."
    )

    logger.info("Starting 1d simple extraction procedure...")

    trace_dict = load_object_traces()

    logger.info("Loading corresponding reduced frames...")
    reduced_files = get_reduced_frames()

    if len(reduced_files) != len(trace_dict):
        logger.error("Number of reduced files and object traces do not match.")
        logger.error("Re-run both procedures or remove left-over files.")
        exit()

    results = extract_objects(reduced_files, trace_dict, method=extract_object_simple)

    write_extracted_1d_to_disc(results)

    logger.info("Extraction procedure complete.")
    print("-------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit simple extract-1d procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_extract_1d_simple()


if __name__ == "__main__":
    main()
