import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse


def calibrate_spectrum(
    wavelength, counts, var, fit, error, exptime, wavelength_ext, data_ext, airmass
):
    """
    Flux calibrates the spectrum using the sensitivity function.

    Firstly, the spectrum is corrected for atmospheric extinction using the
    airmass and the extinction data. Then, the sensitivity function is evaluated
    at the wavelength points of the spectrum and the calibrated flux is calculated.

    Parameters
    ----------
    wavelength : array
        Wavelength points of the spectrum in Å.

    counts : array
        Counts in the spectrum.

    var : array
        Variance in the counts.

    fit : scipy.interpolate.BSpline or numpy.ndarray
        Bspline or Chebyshev polynomial fit object to the sensitivity function,
        depending on the configuration file (see pylongslit.sensitivity_function).

    error : float
        Error in the sensitivity function (see pylongslit.sensitivity_function).

    exptime : float
        Exposure time of the observation.

    wavelength_ext : array
        Wavelength points of the extinction data in Å.

    data_ext : array
        Extinction data in magnitudes/airmass.

    airmass : float
        Airmass of the observation.

    Returns
    -------
    calibrated_flux : array
        Calibrated flux in erg/s/cm2/Å.

    calibrated_var : array
        Variance in the calibrated flux.
    """

    from pylongslit.sensitivity_function import (
        eval_sensfunc,
        estimate_transmission_factor,
    )
    from pylongslit.logger import logger

    # convert to counds/sec
    counts_pr_sec = counts / exptime

    # interpolate the extinction data to the wavelength grid of the spectrum
    ext_interp = interp1d(
        wavelength_ext, data_ext, kind="linear", fill_value="extrapolate"
    )
    data_ext = ext_interp(wavelength)

    logger.info("Correcting for atmospheric extinction...")
    transmission_factor = estimate_transmission_factor(
        wavelength, airmass, data_ext, show_QA=True
    )

    counts_corr = counts_pr_sec * transmission_factor

    # evaluate the sensitivity at the wavelength points

    conv_factors, conv_error = eval_sensfunc(fit, error, wavelength)

    logger.info("Calibrating the spectrum and errors...")
    calibrated_flux = counts_corr * conv_factors

    # error propagation - see the documentation for detailss
    error_counts = np.sqrt(var)

    error_flux = np.sqrt(
        ((counts / exptime) * conv_error) ** 2
        + ((conv_factors / exptime) * error_counts) ** 2
    )

    calibrated_var = error_flux**2

    return calibrated_flux, calibrated_var


def calibrate_flux(spectra, fit, error, good_wavelength_start, good_wavelength_end):
    """
    Driver function to calibrate the flux of the extracted spectra.
    Loops and calls the calibrate_spectrum function for each spectrum.

    Parameters
    ----------
    spectra : dict
        Dictionary containing the extracted spectra.
        Format: {filename: (wavelength, counts, var)}

    fit : scipy.interpolate.BSpline or numpy.ndarray
        Bspline or Chebyshev polynomial fit object to the sensitivity function,
        depending on the configuration file (see pylongslit.sensitivity_function).

    error : float
        Error in the sensitivity function (see pylongslit.sensitivity_function).

    good_wavelength_start : float
        Starting wavelength of the good wavelength range for plotting.
        This is the range where the sensitivity function is well-behaved.

    good_wavelength_end : float
        Ending wavelength of the good wavelength range for plotting.
        This is the range where the sensitivity function is well-behaved.
    """

    from pylongslit.parser import science_params
    from pylongslit.sensitivity_function import load_extinction_data
    from pylongslit.utils import plot_1d_spec_interactive_limits
    from pylongslit.logger import logger

    # extract the needed parameters
    exptime = science_params["exptime"]
    airmass = science_params["airmass"]

    wavelength_ext, data_ext = load_extinction_data()

    # final product
    calibrated_spectra = {}

    for filename, (wavelength, counts, var) in spectra.items():
        # calibrate the spectrum

        logger.info(f"Calibrating the spectrum for {filename}...")

        calibrated_flux, calibrated_var = calibrate_spectrum(
            wavelength,
            counts,
            var,
            fit,
            error,
            exptime,
            wavelength_ext,
            data_ext,
            airmass,
        )
        # save the calibrated spectrum
        calibrated_spectra[filename] = (wavelength, calibrated_flux, calibrated_var)

        # for QA we don't want to plot the very noisy parts where the sensitivity function is not reliable
        mask = (wavelength > good_wavelength_start) & (wavelength < good_wavelength_end)

        # plot for QA
        plot_1d_spec_interactive_limits(
            wavelength[mask],
            calibrated_flux[mask],
            y_error=np.sqrt(calibrated_var)[mask],
            x_label="Wavelength [Å]",
            y_label="Flux [erg/s/cm2/Å]",
            label="Calibrated flux",
            title=f"Calibrated spectrum for {filename}\n"
            f"Only showing the wavelength range where the sensitivity function is well-behaved (from {good_wavelength_start} to {good_wavelength_end} Å).",
        )

        logger.info(f"Spectrum for {filename} calibrated.")
        print("\n---------------------------------------------\n")

    return calibrated_spectra


def write_calibrated_spectra_to_disc(calibrated_spectra):
    """
    Writes the calibrated spectra to disk.

    Parameters
    ----------
    calibrated_spectra : dict
        Dictionary containing the calibrated spectra.
        Format: {filename: (wavelength, calibrated_flux, calibrated_var)}
    """

    from pylongslit.parser import output_dir
    from pylongslit.logger import logger

    logger.info("Writing the calibrated spectra to disk...")
    # change to output directory

    # get the current working directory
    cwd = os.getcwd()

    for filename, (
        wavelength,
        calibrated_flux,
        calibrated_var,
    ) in calibrated_spectra.items():

        new_filename = filename.replace("1d_science", "1d_fluxed_science")

        logger.info(
            f"Writing the calibrated spectrum for {filename} to {new_filename}..."
        )

        # change to output directory
        os.chdir(output_dir)

        # write to the file
        with open(new_filename, "w") as f:
            f.write("# wavelength calibrated_flux\n")
            for i in range(len(wavelength)):
                f.write(f"{wavelength[i]} {calibrated_flux[i]} {calibrated_var[i]}\n")

        f.close()

        # change back to the original directory
        # this is useful when the user is using relative pathes in the configuration file
        os.chdir(cwd)


        logger.info(f"Calibrated spectrum for {filename} written to {new_filename}.")

    logger.info("Calibrated spectra written to disk.")


def run_flux_calib():
    """
    Main function to run the flux calibration procedure.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import load_spec_data
    from pylongslit.sensitivity_function import load_sensfunc_from_disc

    logger.info("Running flux calibration procedure...")

    logger.info("Loading the extracted spectra...")
    spectra = load_spec_data(group="science")
    if len(spectra) == 0:
        logger.error("No extracted spectra found.")
        logger.error("Run the 1d-extraction procedure first.")
        exit()

    fit, error, good_wavelength_start, good_wavelength_end = load_sensfunc_from_disc()

    calibrated_spectra = calibrate_flux(
        spectra, fit, error, good_wavelength_start, good_wavelength_end
    )

    write_calibrated_spectra_to_disc(calibrated_spectra)

    logger.info("Flux calibration procedure completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit flux-calibration procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_flux_calib()


if __name__ == "__main__":
    main()
