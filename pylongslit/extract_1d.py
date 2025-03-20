import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.interpolate import interp1d
import argparse


def load_object_traces():
    """
    Loads the object traces from the output directory.

    Returns
    -------
    trace_dict : dict
        Dictionary containing the object traces.
        Format is {filename: (pixel, center, FWHM)}
    """

    from pylongslit.parser import output_dir
    from pylongslit.utils import get_filenames
    from pylongslit.logger import logger

    logger.info("Loading object traces...")

    # Get all filenames from output_dir starting with "obj_"
    filenames = get_filenames(starts_with="obj_")

    if len(filenames) == 0:
        logger.error("No object traces found.")
        logger.error("Run the object tracing procedure first.")
        exit()

    else:
        logger.info(f"Found {len(filenames)} object traces:")
        for filename in filenames:
            print(filename)

    # sort as this is needed when cross referencing with reduced files
    filenames.sort()

    # this is the container that will be returned from this method
    trace_dict = {}

    # save the current directory
    cwd = os.getcwd()

    # change to output_dir
    os.chdir(output_dir)

    for filename in filenames:
        with open(filename, "r") as file:
            trace_data = np.loadtxt(file)
            pixel = trace_data[:, 0]
            obj_center = trace_data[:, 1]
            obj_fwhm = trace_data[:, 2]

            if pixel is None or obj_center is None or obj_fwhm is None:
                logger.error(f"Error reading {filename}.")
                logger.error("Check the file for errors.")
                logger.error(
                    "The file should contain three columns: pixel, center, FWHM."
                )
                logger.error(
                    "Re-run the object tracing procedure, contact developers if error persists."
                )
                exit()

            trace_dict[filename] = (pixel, obj_center, obj_fwhm)

        file.close()

    # reading done, change back to original directory
    os.chdir(cwd)

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

    from pylongslit.logger import logger

    P = np.exp(-0.5 * (x - mu) ** 2 / sig**2) / (np.sqrt(2.0 * np.pi) * sig)

    if np.round(P.sum(), decimals=0) != 1:
        logger.error(
            "Probability distribution for extraction aperture not normalized correctly."
        )
        logger.error(f"Sum of probabilities: {P.sum()} - should be 1.")
        logger.error("Revisit earlier procedures and check for warning and errors.")
        exit()

    return P


def cauchyweight(x, mu, gamma):
    """
    This method calculates the probability that a photon is detected at a certain
    spacial pixel on the detector row. This is used in `extract_object_optimal`
    , as the P factor in the Horne (1986) optimal extraction algorithm.

    Parameters
    ----------
    x : array-like
        The pixel values.

    mu : float
        The center of the Cauchy object profile.

    gamma : float
        The HWHM of the Cauchy object profile.

    Returns
    -------
    array-like
        The weight for each pixel in the extraction aperture (normalized).
    """

    from pylongslit.logger import logger

    P = 1 / (np.pi * gamma * (1 + ((x - mu) / gamma) ** 2))

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
    NOT USED

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


def extract_object_optimal(trace_data, trace_params, filename):
    """
    Extraction algorithm taken from Horne, K. (1986).
    An optimal extraction algorithm for CCD spectroscopy.
    Publications of the Astronomical Society of the Pacific,
    98(609), 609-617.

    Parameters
    ----------
    trace_data : tuple
        The object trace data.
        Format is (pixel, center, FWHM).

    trace_params : dict
        Dictionary containing the object trace parameters.
        These are the parameters used to determine the object model, taken
        from the configuration file.

    filename : str
        The filename of the reduced frame from which the 1D spectrum is extracted.

    Returns
    -------
    pixel : array-like
        The spectral pixel values of the object trace.

    spec : array-like
        The extracted 1D spectrum. (in ADU)

    spec_var : array-like
        The variance of the extracted 1D spectrum. (in ADU)
    """
    from pylongslit.utils import PyLongslit_frame, check_crr_and_sky
    from pylongslit.logger import logger
    from pylongslit.parser import developer_params

    # unpack the trace data
    pixel, center, FWHM = trace_data

    if developer_params["debug_plots"]:
        plt.plot(pixel, center, label="Object trace")
        plt.legend()
        plt.show()

    # get the reduced frame and unpack
    frame = PyLongslit_frame.read_from_disc(filename)

    check_crr_and_sky(frame.header, filename)

    reduced_data = frame.data.copy()

    if developer_params["debug_plots"]:
        plt.imshow(reduced_data, origin="lower", aspect="auto")
        plt.title(f"Reduced frame {filename}")
        plt.show()

    header = frame.header
    # the y-offset from the cropping procedure. This is used in wavelength calibration
    # to match the global pixel coordinates with the wavelength solution. We
    # extract it here as we do not want to handle header data in the wavelength calibration.
    y_offset = header["CROPY1"]

    # the spatial pixel array
    x_row_array = np.arange(reduced_data.shape[0])

    variance = frame.sigma.copy() ** 2

    if developer_params["debug_plots"]:
        plt.imshow(variance, origin="lower", aspect="auto")
        plt.title(f"Variance frame {filename}")
        plt.show()

    # these are the containers that will be filled for every value
    spec = []
    spec_var = []

    # the extraction loop for every spectral pixel
    for i in range(len(center)):

        obj_center = center[i]
        obj_fwhm = FWHM[i]
        if trace_params["model"] == "Gaussian":
            weight = gaussweight(
                x_row_array, obj_center, obj_fwhm * gaussian_fwhm_to_sigma
            )
        elif trace_params["model"] == "Cauchy":
            # gamma for cauchy is FWHM/2
            weight = cauchyweight(x_row_array, obj_center, obj_fwhm / 2)
        else:
            logger.error("Trace model not recognized.")
            logger.error("Check the configuration file.")
            logger.error("Re-run the object tracing procedure.")
            logger.error("Allowed models are: Gaussian, Cauchy.")
            exit()

        reduced_data_slice = reduced_data[:, i].copy()

        # Horne (1986) eq. 8
        spec.append(
            np.nansum(weight * (reduced_data_slice / variance[:, i]))
            / np.nansum(weight**2 / variance[:, i])
        )

        # Horne (1986) eq. 9
        spec_var.append(1 / np.nansum((weight**2) / variance[:, i]))

    # convert to numpy arrays for easier later handling
    spec = np.array(spec)
    spec_var = np.array(spec_var)

    if developer_params["debug_plots"]:
        plt.figure(figsize=(10, 6))
        plt.plot(spec, label="Extracted 1D spectrum")
        plt.plot(spec_var, label="Variance")
        plt.xlabel("Spectral pixel")
        plt.ylabel("Flux [ADU]")
        plt.title("Extracted 1D spectrum and variance")
        plt.legend()
        plt.show()

    return pixel, spec, spec_var, y_offset


def wavelength_calibrate(pixels, centers, spec, var, y_offset):
    """
    Wavelegth calibration of the extracted 1D spectrum,
    to convert from ADU/spectral pixel to ADU/Å.

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

    y_offset : float
        The y-offset from the cropping procedure. This is used in wavelength calibration
        to match the global pixel coordinates with the wavelength solution.

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

    # the object trace centers are local pixel coordinates, we need to add the y-offset
    # to get the global pixel coordinates for wavelength calibration
    centers_global = centers + y_offset

    # get the wavelength and tilt fits from disc
    wavelen_fit = get_wavelen_fit_from_disc()
    tilt_fit = get_tilt_fit_from_disc()

    # evaluate the wavelength solution at the object trace centers
    obj_wavelen = wavelength_sol(pixels, centers_global, wavelen_fit, tilt_fit)

    # interpolate the spectrum and variance to a homogenous wavelength grid
    wavelen_homogenous = np.linspace(obj_wavelen[0], obj_wavelen[-1], len(spec))

    spec_interpolate = interp1d(
        obj_wavelen, spec, fill_value="extrapolate", kind="cubic"
    )
    spec_calibrated = spec_interpolate(wavelen_homogenous)

    var_interpolate = interp1d(obj_wavelen, var, fill_value="extrapolate", kind="cubic")
    var_calibrated = var_interpolate(wavelen_homogenous)

    return wavelen_homogenous, spec_calibrated, var_calibrated


def plot_extracted_1d(filename, wavelengths, spec_calib, var_calib, figsize=(10, 6)):
    """
    Plot of the extracted 1D spectrum (counts [ADU] vs. wavelength [Å]).
    Wrapper for `pylongslit.utils.plot_1d_spec_interactive_limits`.

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
        The figure size. Default is (10, 6).
    """

    from pylongslit.utils import plot_1d_spec_interactive_limits

    plot_1d_spec_interactive_limits(
        wavelengths,
        spec_calib,
        y_error=np.sqrt(var_calib),
        x_label="Wavelength [Å]",
        y_label="Counts [ADU]",
        label="Extracted 1D spectrum",
        title=f"Extracted 1D spectrum from {filename}. Use the sliders to crop out noisy edges if needed.",
        figsize=figsize,
    )


def extract_objects(reduced_files, trace_dict, method=extract_object_optimal):
    """
    Driver for the extraction of 1D spectra from reduced frames.

    First used `extract_object_optimal` to extract the 1D spectrum, and then
    uses `wavelength_calibrate` to calibrate the spectrum to wavelength.
    Plots results for QA.

    Parameters
    ----------
    reduced_files : list
        List of filenames of reduced frames.

    trace_dict : dict
        Dictionary containing the object traces.
        Format is {filename: (pixel, center, FWHM)}

    method : function
        The extraction method to use. Default is `extract_object_optimal`.

        The method should take the following input vector:
        (trace_data, trace_params, filename), with:
        - trace_data : tuple
            The object trace data.
            Format is (pixel, center, FWHM).
        - trace_params : dict
            Dictionary containing the object trace parameters
            taken from the configuration file.
        - filename : str
            The filename of the reduced frame from which the 1D spectrum is extracted.

        The method should return the following output vector:
        (pixel, spec, spec_var, y_offset), with:
        - pixel : array-like
            The spectral pixel values of the object trace.
        - spec : array-like
            The extracted 1D spectrum. (in ADU)
        - spec_var : array-like
            The variance of the extracted 1D spectrum. (in ADU)
        - y_offset : float
            The y-offset from the cropping procedure. This is used in wavelength calibration
            to match the global pixel coordinates with the wavelength solution.


    Returns
    -------
    results : dict
        Dictionary containing the extracted 1D spectra.
        Format is {filename: (wavelength, spectrum_calib, var_calib)}
    """

    from pylongslit.logger import logger
    from pylongslit.obj_trace import get_params

    # This is the container for the resulting one-dimensional spectra
    results = {}

    for filename in reduced_files:

        logger.info(f"Extracting 1D spectrum from {filename}...")

        # get the trace parameters, as these determine the object model
        trace_params = get_params(filename)

        filename_obj = filename.replace("reduced_", "obj_").replace(".fits", ".dat")

        trace_data = trace_dict[filename_obj]

        pixel, spec, spec_var, y_offset = method(trace_data, trace_params, filename)

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

        logger.info(f"1D spectrum extracted and wavelength calibrated for {filename}.")
        print("-------------------------\n")

    return results


def write_extracted_1d_to_disc(results):
    """
    Writes the extracted 1D spectra to disc, in the output directory
    specified in the configuration file.

    Parameters
    ----------
    results : dict
        Dictionary containing the extracted 1D spectra.
        Format is {filename: (wavelength, spectrum_calib, var_calib)}
    """

    from pylongslit.parser import output_dir
    from pylongslit.logger import logger

    logger.info("Writing extracted 1D spectra to disc...")

    # get the current directory
    cwd = os.getcwd()

    os.chdir(output_dir)

    for filename, data in results.items():
        with open(filename, "w") as file:
            file.write(f"# Extracted 1D spectrum from {filename}\n")
            file.write("# Wavelength Flux Variance\n")
            for i in range(len(data[0])):
                file.write(f"{data[0][i]} {data[1][i]} {data[2][i]}\n")

        logger.info(f"{filename} written to disc in directory {output_dir}")

    file.close()

    os.chdir(cwd)

    logger.info("All extracted 1D spectra written to disc.")


def run_extract_1d():
    """
    Main driver for the extract-1d procedure.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import get_reduced_frames

    logger.info("Starting 1d extraction procedure...")

    trace_dict = load_object_traces()

    logger.info("Loading corresponding reduced frames...")
    reduced_files = get_reduced_frames()

    if len(reduced_files) != len(trace_dict):
        logger.error("Number of reduced files and object traces do not match.")
        logger.error("Re-run both procedures or remove left-over files.")
        exit()

    results = extract_objects(reduced_files, trace_dict)

    write_extracted_1d_to_disc(results)

    logger.info("Extraction procedure complete.")
    print("-------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit extract-1d procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_extract_1d()


if __name__ == "__main__":

    main()
