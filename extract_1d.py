from parser import output_dir, detector_params
from utils import open_fits, list_files, get_skysub_files, get_filenames
from logger import logger
import os
import numpy as np
from photutils.aperture import RectangularAperture
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, gaussian_fwhm_to_sigma
from scipy.interpolate import interp1d
from wavecalib import get_tilt_fit_from_disc, get_wavelen_fit_from_disc
from wavecalib import wavelength_sol 


def load_object_traces():
    """
    Loads the object traces from the output directory.
    """

    logger.info("Loading object traces")

    # Get all filenames from output_dir starting with "obj_"
    filenames = get_filenames(starts_with="obj_")

    if len(filenames) == 0:
        logger.error("No object traces found.")
        logger.error("Run the object tracing procedure first.")
        exit()

    else:
        logger.info(f"Found {len(filenames)} object traces:")
        list_files(filenames)

    # sort as this is needed when cross referencing with skysubbed files
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


def gaussweight(x, mu, sig):
    """
    This method calculates the probability that a photon is detected at a certain
    spacial pixel on the detector row. This is used in `extract_object_optimal`
    , as the P factor in the Horne (1986) optimal extraction algorithm.

    Parameters
    ----------
    x : array-like
        The pixel values.

    mu : float
        The center of the Gaussian object profile.

    sig : float
        The standard deviation of the Gaussian object profile.

    Returns
    -------
    array-like
        The weight for each pixel in the extraction aperture (normalized).
    """

    P = np.exp(-0.5 * (x - mu) ** 2 / sig ** 2) / (np.sqrt(2.0 * np.pi) * sig)

    if np.round(P.sum(), decimals=0) != 1:
        logger.error(
            "Probability distribution for extraction aperture not normalized correctly."
        )
        logger.error(f"Sum of probabilities: {P.sum()} - should be 1.")
        logger.error("Revisit earlier procedures and check for warning and errors.")
        exit()

    return P


def estimate_variance(data, gain, read_out_noise):

    """
    Taken from Horne, K. (1986).
    An optimal extraction algorithm for CCD spectroscopy.
    Publications of the Astronomical Society of the Pacific,
    98(609), 609-617, eq. 12.

    Parameters
    ----------
    data : array-like
        The data array.

    gain : float
        The gain of the CCD. (electrons/ADU)

    read_out_noise : float
        The read out noise of the CCD. (electrons)

    Returns
    -------
    array-like
        The variance of the data array. (in ADU)
    """

    return (read_out_noise / gain) ** 2 + np.abs(data)


def extract_object_optimal(trace_data, skysubbed_frame, gain, read_out_noise):
    """
    Extraction algorithm taken from Horne, K. (1986).
    An optimal extraction algorithm for CCD spectroscopy.
    Publications of the Astronomical Society of the Pacific,
    98(609), 609-617.

    Parameters
    ----------
    trace_data : tuple
        The trace data from the object tracing procedure.
        Contains the pixel, center and FWHM of the object traces.

    skysubbed_frame : str
        The filename of the skysubbed frame to extract the object from.

    gain : float
        The gain of the CCD. (electrons/ADU)

    read_out_noise : float
        The read out noise of the CCD. (electrons)

    Returns
    -------
    pixel : array-like
        The spectral pixel values of the object trace.

    spec : array-like
        The extracted 1D spectrum. (in ADU)

    spec_var : array-like
        The variance of the extracted 1D spectrum. (in ADU)
    """

    pixel, center, FWHM = trace_data

    # Open the skysubbed frame
    hdul = open_fits(output_dir, skysubbed_frame)

    skysubbed_data = hdul[0].data
    x_row_array = np.arange(skysubbed_data.shape[0])

    variance = estimate_variance(skysubbed_data, gain, read_out_noise)

    # these are the containers that will be filled for every value
    spec = []
    spec_var = []

    # the extraction loop for every spectral pixel
    for i in range(len(center)):

        obj_center = center[i]
        obj_fwhm = FWHM[i] * gaussian_fwhm_to_sigma
        weight = gaussweight(x_row_array, obj_center, obj_fwhm)

        skysubbed_data_slice = skysubbed_data[:, int(pixel[0]) + i]

        # Horne (1986) eq. 8
        spec.append(
            np.sum(weight * skysubbed_data_slice / variance[:, i])
            / np.sum(weight ** 2 / variance[:, i])
        )
        # Horne (1986) eq. 9
        spec_var.append(1 / np.sum((weight ** 2) / variance[:, i]))

    spec = np.array(spec)
    spec_var = np.array(spec_var)

    return pixel, spec, spec_var


def wavelength_calibrate(pixels, centers, spec, var):
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

    wavelen_fit = get_wavelen_fit_from_disc()
    tilt_fit = get_tilt_fit_from_disc()

    # evaluate the wavelength solution at the object trace centers
    ap_wavelen = wavelength_sol(pixels, centers, wavelen_fit, tilt_fit)

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
    ax.set_ylim(-0.5, 1.1 * np.max(spec_calib))
    ax.legend()
    ax.grid()

    plt.show()


def extract_objects(skysubbed_files, trace_dir):
    """
    Driver for the extraction of 1D spectra from skysubbed frames.

    First used `extract_object_optimal` to extract the 1D spectrum, and then
    uses `wavelength_calibrate` to calibrate the spectrum to wavelength.
    Plots results for QA.

    Parameters
    ----------
    skysubbed_files : list
        List of filenames of skysubbed frames.

    trace_dir : dict
        Dictionary containing the object traces.
        Format is {filename: (pixel, center, FWHM)}

    Returns
    -------
    results : dict
        Dictionary containing the extracted 1D spectra.
        Format is {filename: (wavelength, spectrum_calib, var_calib)}
    """

    # get gain and read out noise parameters
    gain = detector_params["gain"]
    read_out_noise = detector_params["read_out_noise"]

    # This is the container for the resulting one-dimensional spectra
    results = {}

    for filename in skysubbed_files:

        logger.info(f"Extracting 1D spectrum from {filename}...")

        filename_obj = filename.replace("skysub_", "obj_").replace(".fits", ".dat")

        trace_data = trace_dir[filename_obj]

        pixel, spec, spec_var = extract_object_optimal(
            trace_data, filename, gain, read_out_noise
        )

        logger.info("Spectrum extracted.")
        logger.info("Wavelength calibrating the extracted 1D spectrum...")

        wavelength, spectrum_calib, var_calib = wavelength_calibrate(
            pixel, trace_data[1], spec, spec_var
        )

        # make a new filename
        new_filename = filename.replace("skysub_", "1d_").replace(".fits", ".dat")

        results[new_filename] = (wavelength, spectrum_calib, var_calib)

        # plot results for QA
        plot_extracted_1d(new_filename, wavelength, spectrum_calib, var_calib)

    return results


def write_extracted_1d_to_disc(results):

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
    logger.info("Running extract_1d")

    trace_dir = load_object_traces()

    skysubbed_files = get_skysub_files()

    if len(skysubbed_files) != len(trace_dir):
        logger.error("Number of skysubbed files and object traces do not match.")
        logger.error("Re-run both procedures or remove left-over files.")
        exit()

    results = extract_objects(skysubbed_files, trace_dir)

    write_extracted_1d_to_disc(results)

    logger.info("extract_1d done")
    print("-------------------------\n")


if __name__ == "__main__":
    run_extract_1d()
