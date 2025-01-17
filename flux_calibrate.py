from logger import logger
from parser import output_dir, science_params
from utils import load_spec_data
import os
import numpy as np
import matplotlib.pyplot as plt
from sensitivity_function import load_sensfunc_from_disc
from numpy.polynomial.chebyshev import chebval


def calibrate_spectrum(wavelength, counts, var, sens_coeffs, exptime):

    # evaluate the sensitivity at the wavelength points
    # and convert back from logspaces
    conv_factors = 10**chebval(wavelength, sens_coeffs)

    # divide by exposure time and multiply by evaluated sensitivity
    calibrated_flux = (counts/exptime) * conv_factors
    calibrated_var = (var/(exptime**2)) * (conv_factors**2)


    return calibrated_flux, calibrated_var

def plot_calibrated_spectrum(filename, wavelength, calibrated_flux, calibrated_var, figsize=(18,12)):

    plt.figure(figsize=figsize) 
    plt.plot(wavelength, calibrated_flux, label="Calibrated flux")
    plt.plot(wavelength, np.sqrt(calibrated_var), label="Sigma")

    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux [erg/s/cm2/Å]")
    plt.title(f"Calibrated spectrum for {filename}")
    plt.grid()
    plt.show()


def calibrate_flux(spectra, sens_coeffs):

    exptime = science_params["exptime"]

    # final product
    calibrated_spectra = {}

    for filename, (wavelength, counts, var) in spectra.items():
        # calibrate the spectrum
        calibrated_flux, calibrated_var = calibrate_spectrum(wavelength, counts, var, sens_coeffs, exptime)
        # save the calibrated spectrum
        calibrated_spectra[filename] = (wavelength, calibrated_flux, calibrated_var)
        # plot for QA
        plot_calibrated_spectrum(filename, wavelength, calibrated_flux, calibrated_var)


    return calibrated_spectra


def write_calibrated_spectra_to_disc(calibrated_spectra):

    for filename, (wavelength, calibrated_flux, calibrated_var) in calibrated_spectra.items():

        new_filename = filename.replace("1d_science", "1d_fluxed_science")

        # change to output directory
        os.chdir(output_dir)

        # write to the file
        with open(new_filename, "w") as f:
            f.write("# wavelength calibrated_flux\n")
            for i in range(len(wavelength)):
                print("writing", wavelength[i], calibrated_flux[i], calibrated_var[i])
                f.write(f"{wavelength[i]} {calibrated_flux[i]} {calibrated_var[i]}\n")

        f.close()

    logger.info("Calibrated spectra written to disk.")



def run_flux_calib():
    logger.info("Running flux calibration...")

    spectra = load_spec_data(group = "science")

    sens_coeffs = load_sensfunc_from_disc()

    calibrated_spectra = calibrate_flux(spectra, sens_coeffs)

    write_calibrated_spectra_to_disc(calibrated_spectra)



if __name__ == "__main__":
    run_flux_calib()