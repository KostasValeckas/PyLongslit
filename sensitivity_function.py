from logger import logger
import numpy as np
from parser import output_dir, detector_params, standard_params
from utils import list_files
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval
import pickle


def read_sensfunc_params():

    exptime = standard_params["exptime"]
    airmass = standard_params["airmass"]
    star_name = standard_params["starname"]
    flux_file = standard_params["flux_file_path"]

    print(exptime, airmass, star_name, flux_file)

    return exptime, airmass, star_name, flux_file

def load_standard_star_spec():

    filenames = [filename for filename in os.listdir(output_dir) if filename.startswith("1d_std")]

    if len(filenames) == 0:
        logger.error("No standard star spectra found.")
        logger.error("Run the extract 1d procedure first.")
        logger.error("If you have already run the procedure, check the \"skip_standard\" parameter in the config file.")
        exit()

    else:
        logger.info(f"Found {len(filenames)} standard star spectra:")
        list_files(filenames)
        if len(filenames) > 1:
            logger.warning("Multiple standard star spectra found. Software only supports one.")
            logger.warning("Using the first one.")

    os.chdir(output_dir)

    data = np.loadtxt(filenames[0], skiprows=2)

    wavelength = data[:,0]
    counts = data[:,1]

    plt.plot(wavelength, counts)
    plt.show()

    return wavelength, counts

def load_ref_spec(file_path):

    data = np.loadtxt(file_path)

    wavelength = data[:,0]
    flux = data[:,1]

    # calculated the distances between wavelengths
    # thhis arra has the form: bandwidth[i] = wavelength[i+1] - wavelength[i]
    bandwidth = np.diff(wavelength)

    plt.plot(wavelength, flux)
    plt.show()

    return wavelength, flux, bandwidth

def convert_from_AB_mag_to_flux(mag, ref_wavelength):
    """
    Converts from AB magnitudes to erg/s/cm^2/Å   

    From Oke, J.B. 1974, ApJS, 27, 21 
    """

    flux = 2.998e18*10**((mag + 48.6)/(-2.5))/(ref_wavelength**2)

    plt.plot(flux)
    plt.show()

    return flux

def refrence_counts_to_flux(wavelength, counts, ref_wavelength, ref_flux, bandwidth, exptime):    

    #TODO: Clean up in naming!
    
    flux_converted = convert_from_AB_mag_to_flux(ref_flux, ref_wavelength)

    counts_pr_sec = counts/exptime

    flux_converted = flux_converted[(ref_wavelength >= np.min(wavelength)) & (ref_wavelength <= np.max(wavelength))]
    ref_wavelength = ref_wavelength[(ref_wavelength >= np.min(wavelength)) & (ref_wavelength <= np.max(wavelength))]
    

    # Interpolate counts per second onto reference wavelength
    interpolated_counts = interp1d(wavelength, counts_pr_sec)(ref_wavelength)

    # try a simple appraoch : TODO this is while developing only

    sens_points = flux_converted / interpolated_counts

    return ref_wavelength, sens_points



def fit_sensfunc(wavelength, sens_points):

    coeff = chebfit(wavelength[500:], sens_points[500:], deg = 9)

    fit_eval = chebval(wavelength, coeff)

    plt.plot(wavelength, sens_points, "x", label = "data")
    plt.plot(wavelength, fit_eval, label = "fit")
    plt.legend()
    plt.show()

    print(coeff)

    return coeff

def flux_standard_QA(coeff, wavelength, counts, ref_wavelength, ref_flux):

    conv_factors = chebval(wavelength, coeff)

    plt.plot(conv_factors, "x")
    plt.title("Conversion factors")
    plt.show()

    fluxed_counts = (counts/standard_params["exptime"]) * conv_factors

    converted_ref_spec = convert_from_AB_mag_to_flux(ref_flux, ref_wavelength)


    plt.plot(wavelength, fluxed_counts, label="Fluxed standard star spec")
    plt.plot(ref_wavelength, converted_ref_spec, label="Reference spectrum")
    plt.legend()

    plt.show()

    #TODO: write to disc for debugging, remove later

    os.chdir(output_dir)

    with open("std_spec.dat", "wb") as f:
        pickle.dump((wavelength, fluxed_counts), f)

    f.close()
    

def write_sensfunc_to_disc(coeff):

    logger.info("Writing sensitivity function coefficients to disk...")

    os.chdir(output_dir)
        
    
    with open("sens_coeff.dat", "wb") as f:
        pickle.dump(coeff, f)
    
    logger.info(
        f"Sensitivity function coefficients written to {output_dir}, filename : sens_coeff.dat."
    )


def load_sensfunc_from_disc():
    logger.info("Loading sensitivity function coefficients from disk...")

    os.chdir(output_dir)

    with open("sens_coeff.dat", "rb") as f:
        coeff = pickle.load(f)

    logger.info("Sensitivity function coefficients loaded.")

    return coeff




def run_sensitivity_function():
    
    logger.info("Staring the process of producing the sensitivity function...")
    
    exptime, airmass, star_name, flux_file = read_sensfunc_params()

    obs_wavelength, obs_counts = load_standard_star_spec()

    ref_wavelength, ref_flux, bandwith = load_ref_spec(flux_file)

    ref_wavelength_temp = ref_wavelength

    ref_wavelength, sens_points = refrence_counts_to_flux(obs_wavelength, obs_counts, ref_wavelength, ref_flux, bandwith, exptime)
    
    coeff = fit_sensfunc(ref_wavelength, sens_points)

    # TODO: clean up when done developing!

    flux_standard_QA(coeff, obs_wavelength, obs_counts, ref_wavelength_temp, ref_flux)

    write_sensfunc_to_disc(coeff)




if __name__ == "__main__":
    run_sensitivity_function()