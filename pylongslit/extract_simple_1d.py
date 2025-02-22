import os
import numpy as np
from photutils.aperture import RectangularAperture
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse


def load_object_traces(only_science=True):
    """
    Loads the object traces from the output directory.
    """

    from pylongslit.parser import output_dir
    from pylongslit.utils import list_files,  get_filenames
    from pylongslit.logger import logger

    logger.info("Loading object traces")

    # Get all filenames from output_dir starting with "obj_"
    filenames = get_filenames(starts_with="obj_science") if only_science else get_filenames(starts_with="obj_")

    if len(filenames) == 0:
        logger.error("No object traces found.")
        logger.error("Run the object tracing procedure first.")
        exit()

    else:
        logger.info(f"Found {len(filenames)} object traces:")
        list_files(filenames)

    # sort as this is needed when cross referencing with reduced files
    filenames.sort()

    # this is the container that will be returned
    trace_dict = {}

    # change to output_dir
    os.chdir(output_dir)

    for filename in filenames:
        with open(filename, "r") as file:
            trace_data = np.loadtxt(file)
            pixel = trace_data[:, 0]
            obj_center = trace_data[:, 1]
            obj_fwhm = trace_data[:, 2]

            trace_dict[filename] = (pixel, obj_center, obj_fwhm)

        file.close()

    # reading done, change back to original directory
    os.chdir("..")

    # Process the filenames as needed

    logger.info("All object traces loaded.")

    return trace_dict


def extract_object_simple(trace_data, reduced_frame):
    """ """

    from pylongslit.parser import output_dir, detector_params
    from pylongslit.utils import open_fits, PyLongslit_frame
    from pylongslit.extract_1d import estimate_variance

    pixel, center, FWHM = trace_data

    # Open the reduced frame
    frame = PyLongslit_frame.read_from_disc(reduced_frame)

    reduced_data = frame.data

    data_error = frame.sigma

    header = frame.header
    y_offset = header["CROPY1"]  # the y-offset from the cropping procedure

    x_row_array = np.arange(reduced_data.shape[0])

    # these are the containers that will be filled for every value
    spec = []
    spec_var = []

    # the extraction loop for every spectral pixel
    for i in range(len(center)):

        obj_center = center[i]
        pixel_coord = pixel[i]
        fwhm = FWHM[i]

        # sum around FWHM

        # define the aperture
        aperture = RectangularAperture((pixel_coord, obj_center), 1, fwhm)

        # extract the spectrum
        spec_sum = aperture.do_photometry(reduced_data, error=data_error)

        spec_sum_counts = spec_sum[0][0]
        spec_err_counts = spec_sum[1][0]

        spec.append(spec_sum_counts)
        spec_var.append(spec_err_counts**2)
        


    plot_trace_QA(reduced_data, pixel, center, FWHM, reduced_frame)

    spec = np.array(spec)
    spec_var = np.array(spec_var)

    return pixel, spec, spec_var, y_offset


def wavelength_calibrate(pixels, centers, spec, var, y_offset):
    """
    Wavelegth calibration of the extracted 1D spectrum,
    to convert from ADU/pixel to ADU/Å.

    Parameters
    ----------
    pixels : array-like
        The pixel values of the extracted 1D spectrum.

    centers : array-like
        The center of the object trace.
        These define the points where to evaluate the wavelength solution.

    spec : array-like
        The extracted 1D spectrum. (in ADU)

    var : array-like
        The variance of the extracted 1D spectrum. (in ADU)

    fit2d_REID : chebyshev1d object
        The wavelength calibration solution.
        For type, see output of `wavecalib.load_fit2d_REID_from_disc`.

    Returns
    -------
    wavelen_homogenous : array-like
        The homogenous wavelength grid.

    spec_calibrated : array-like
        The calibrated 1D spectrum. (in ADU/Å)

    var_calibrated : array-like
        The variance of the calibrated 1D spectrum. (in ADU/Å)
    """

    from pylongslit.wavecalib import get_tilt_fit_from_disc, get_wavelen_fit_from_disc
    from pylongslit.utils import wavelength_sol

    centers_global = centers + y_offset

    wavelen_fit = get_wavelen_fit_from_disc()
    tilt_fit = get_tilt_fit_from_disc()

    # evaluate the wavelength solution at the object trace centers
    ap_wavelen = wavelength_sol(pixels, centers_global, wavelen_fit, tilt_fit)

    # interpolate the spectrum and variance to a homogenous wavelength grid
    wavelen_homogenous = np.linspace(ap_wavelen[0], ap_wavelen[-1], len(spec))

    spec_interpolate = interp1d(
        ap_wavelen, spec, fill_value="extrapolate", kind="cubic"
    )
    spec_calibrated = spec_interpolate(wavelen_homogenous)

    var_interpolate = interp1d(ap_wavelen, var, fill_value="extrapolate", kind="cubic")
    var_calibrated = var_interpolate(wavelen_homogenous)

    return wavelen_homogenous, spec_calibrated, var_calibrated


def plot_extracted_1d(filename, wavelengths, spec_calib, var_calib, figsize=(18, 12)):
    """
    Plot of the extracted 1D spectrum (counts [ADU] vs. wavelength [Å]).

    Parameters
    ----------
    filename : str
        The filename from which the spectrum was extracted.

    wavelengths : array-like
        The homogenous wavelength grid.

    spec_calib : array-like
        The calibrated 1D spectrum. (in ADU/Å)

    var_calib : array-like
        The variance of the calibrated 1D spectrum. (in ADU/Å)

    figsize : tuple
        The figure size. Default is (18, 12).
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(wavelengths, spec_calib, label="Calibrated spectrum")
    ax.plot(
        wavelengths, np.sqrt(var_calib), label="1-sigma error spectrum", linestyle="--"
    )

    ax.set_title(f"Extracted 1D spectrum from {filename}")
    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Counts [ADU]")
    # any negative values may be due to numerical instability - don't show them
    ax.set_ylim(-0.5, 1.1 * np.nanmax(spec_calib))
    ax.legend()
    ax.grid()

    plt.show()


def plot_trace_QA(image, pixel, trace, fwhm, filename, num_plots=6, figsize=(10, 18)):

    from pylongslit.parser import output_dir
    from pylongslit.utils import  hist_normalize

    fig, axes = plt.subplots(num_plots, 1, figsize=figsize)

    segment_length = image.shape[1] // num_plots

    for i, ax in enumerate(axes):
        start = i * segment_length
        end = (i + 1) * segment_length if i < (num_plots - 1) else image.shape[1]

        segment = image[:, start:end]
        segment_pixel = pixel[start:end]
        segment_trace = trace[start:end]
        segment_fwhm = fwhm[start:end]

        ax.imshow(hist_normalize(segment), origin="lower", cmap="gray", aspect="auto")
        ax.plot(segment_pixel - start, segment_trace - segment_fwhm, color="red")
        ax.plot(segment_pixel - start, segment_trace + segment_fwhm, color="red")

        ax.set_yticks([])
        ax.set_xticklabels(segment_pixel.astype(int))

    os.chdir(output_dir)
    plt.suptitle(f"Object trace QA for {filename}")
    plt.savefig("trace_QA " + filename + ".png")

    plt.show()


def extract_objects(reduced_files, trace_dir):
    """
    Driver for the extraction of 1D spectra from reduced frames.

    First used `extract_object_simple` to extract the 1D spectrum, and then
    uses `wavelength_calibrate` to calibrate the spectrum to wavelength.
    Plots results for QA.

    Parameters
    ----------
    reduced_files : list
        List of filenames of reduced frames.

    trace_dir : dict
        Dictionary containing the object traces.
        Format is {filename: (pixel, center, FWHM)}

    Returns
    -------
    results : dict
        Dictionary containing the extracted 1D spectra.
        Format is {filename: (wavelength, spectrum_calib, var_calib)}
    """

    from pylongslit.logger import logger

    # This is the container for the resulting one-dimensional spectra
    results = {}

    for filename in reduced_files:

        logger.info(f"Extracting 1D spectrum from {filename}...")

        filename_obj = filename.replace("reduced_", "obj_").replace(".fits", ".dat")

        trace_data = trace_dir[filename_obj]

        pixel, spec, spec_var, y_offset = extract_object_simple(trace_data, filename)

        logger.info("Spectrum extracted.")
        logger.info("Wavelength calibrating the extracted 1D spectrum...")

        wavelength, spectrum_calib, var_calib = wavelength_calibrate(
            pixel, trace_data[1], spec, spec_var, y_offset
        )

        # make a new filename
        new_filename = filename.replace("reduced_", "1d_").replace(".fits", ".dat")

        results[new_filename] = (wavelength, spectrum_calib, var_calib)

        # plot results for QA
        plot_extracted_1d(new_filename, wavelength, spectrum_calib, var_calib)

    return results


def write_extracted_1d_to_disc(results):

    from pylongslit.parser import output_dir
    from pylongslit.logger import logger

    logger.info("Writing extracted 1D spectra to disc")

    os.chdir(output_dir)

    for filename, data in results.items():
        with open(filename, "w") as file:
            file.write(f"# Extracted 1D spectrum from {filename}\n")
            file.write("# Wavelength Flux Variance\n")
            for i in range(len(data[0])):
                file.write(f"{data[0][i]} {data[1][i]} {data[2][i]}\n")

        logger.info(f"{filename} written to disc in directory {output_dir}")

    file.close()

    os.chdir("..")

    logger.info("All extracted 1D spectra written to disc")


def run_extract_1d():

    from pylongslit.utils import get_reduced_frames
    from pylongslit.logger import logger

    logger.info("Running extract_1d")

    trace_dir = load_object_traces()

    reduced_files = get_reduced_frames(only_science=True)

    if len(reduced_files) != len(trace_dir):
        logger.error("Number of reduced files and object traces do not match.")
        logger.error("Re-run both procedures or remove left-over files.")
        exit()

    results = extract_objects(reduced_files, trace_dir)

    write_extracted_1d_to_disc(results)

    logger.info("extract_1d done")
    print("-------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit simple extract-1d procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_extract_1d()


if __name__ == "__main__":
    main()
