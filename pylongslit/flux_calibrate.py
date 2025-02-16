import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval
import argparse


def calibrate_spectrum(wavelength, counts, var, sens_coeffs, error, exptime):

    from pylongslit.sensitivity_function import eval_sensfunc

    # evaluate the sensitivity at the wavelength points
    # and convert back from logspaces
    conv_factors, conv_error = eval_sensfunc(sens_coeffs, error, wavelength)


    # divide by exposure time and multiply by evaluated sensitivity
    calibrated_flux = (counts / exptime) * conv_factors
    
    error_counts = np.sqrt(var)

    error_flux = np.sqrt(((counts/exptime)*conv_error)**2 + ((conv_factors/exptime)*error_counts)**2)

    calibrated_var = error_flux**2

    return calibrated_flux, calibrated_var


def plot_calibrated_spectrum(
    filename, wavelength, calibrated_flux, calibrated_var, figsize=(18, 12)
):

    plt.figure(figsize=figsize)
    plt.plot(wavelength, calibrated_flux, label="Calibrated flux")
    plt.plot(wavelength, np.sqrt(calibrated_var), label="Sigma")

    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux [erg/s/cm2/Å]")
    plt.title(f"Calibrated spectrum for {filename}")
    plt.grid()
    plt.show()


def calibrate_flux(spectra, sens_coeffs, error):

    from pylongslit.parser import science_params

    exptime = science_params["exptime"]

    # final product
    calibrated_spectra = {}

    for filename, (wavelength, counts, var) in spectra.items():
        # calibrate the spectrum
        calibrated_flux, calibrated_var = calibrate_spectrum(
            wavelength, counts, var, sens_coeffs, error, exptime
        )
        # save the calibrated spectrum
        calibrated_spectra[filename] = (wavelength, calibrated_flux, calibrated_var)
        # plot for QA
        plot_calibrated_spectrum(filename, wavelength, calibrated_flux, calibrated_var)

    return calibrated_spectra


def write_calibrated_spectra_to_disc(calibrated_spectra):

    from pylongslit.parser import output_dir
    from pylongslit.logger import logger

    for filename, (
        wavelength,
        calibrated_flux,
        calibrated_var,
    ) in calibrated_spectra.items():

        new_filename = filename.replace("1d_science", "1d_fluxed_science")

        # change to output directory
        os.chdir(output_dir)

        # write to the file
        with open(new_filename, "w") as f:
            f.write("# wavelength calibrated_flux\n")
            for i in range(len(wavelength)):
                f.write(f"{wavelength[i]} {calibrated_flux[i]} {calibrated_var[i]}\n")

        f.close()

    logger.info("Calibrated spectra written to disk.")


def run_flux_calib():

    from pylongslit.logger import logger
    from pylongslit.utils import load_spec_data
    from pylongslit.sensitivity_function import load_sensfunc_from_disc


    logger.info("Running flux calibration...")

    spectra = load_spec_data(group="science")

    sens_coeffs, error = load_sensfunc_from_disc()

    calibrated_spectra = calibrate_flux(spectra, sens_coeffs, error)

    write_calibrated_spectra_to_disc(calibrated_spectra)

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit flux-calibration procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_flux_calib()


if __name__ == "__main__":
    main()
