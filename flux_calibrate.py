from logger import logger
from parser import output_dir, science_params
from utils import list_files
import os
import numpy as np
import matplotlib.pyplot as plt
from sensitivity_function import load_sensfunc_from_disc
from numpy.polynomial.chebyshev import chebval

def load_sciece_spec():
    #TODO: combine this with the load_standard_star_spec function, move to utils.py

    filenames = [filename for filename in os.listdir(output_dir) if filename.startswith("1d_science")]

    if len(filenames) == 0:
        logger.error("No science spectra found.")
        logger.error("Run the extract 1d procedure first.")
        logger.error("If you have already run the procedure, check the \"skip_science\" parameter in the config file.")
        exit()

    
    logger.info(f"Found {len(filenames)} science spectra:")
    list_files(filenames)

    os.chdir(output_dir)

    # container for the spectra
    spectra = {}

    for filename in filenames:
        data = np.loadtxt(filename, skiprows=2)
        wavelength = data[:,0]
        counts = data[:,1]

        spectra[filename] = (wavelength, counts)


    return spectra


def calibrate_spectrum(wavelength, counts, sens_coeffs, exptime):

    print(sens_coeffs)

    conv_factors = chebval(wavelength, sens_coeffs)


    plt.plot(conv_factors)
    plt.title("Conversion factors")
    plt.show()

    print(exptime)

    calibrated_flux = (counts/exptime) * conv_factors

    plt.plot(wavelength, calibrated_flux)
    plt.show()

    return calibrated_flux


def calibrate_flux(spectra, sens_coeffs):
    # TODO probably don't need this wrapper - just call calibrate_spectrum directly

    exptime = science_params["exptime"]

    # final product
    calibrated_spectra = {}

    for filename, (wavelength, counts) in spectra.items():
        calibrated_flux = calibrate_spectrum(wavelength, counts, sens_coeffs, exptime)

        calibrated_spectra[filename] = (wavelength, calibrated_flux)

    return calibrated_spectra


def write_calibrated_spectra_to_disc(calibrated_spectra):

    for filename, (wavelength, calibrated_flux) in calibrated_spectra.items():

        new_filename = filename.replace("1d_science", "1d_fluxed_science")

        # change to output directory
        os.chdir(output_dir)

        # write to the file
        with open(new_filename, "w") as f:
            f.write("# wavelength calibrated_flux\n")
            for i in range(len(wavelength)):
                f.write(f"{wavelength[i]} {calibrated_flux[i]}\n")

        f.close()

    logger.info("Calibrated spectra written to disk.")



def run_flux_calib():
    logger.info("Running flux calibration...")

    spectra = load_sciece_spec()

    sens_coeffs = load_sensfunc_from_disc()

    calibrated_spectra = calibrate_flux(spectra, sens_coeffs)

    write_calibrated_spectra_to_disc(calibrated_spectra)



if __name__ == "__main__":
    run_flux_calib()