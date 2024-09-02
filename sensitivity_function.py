from logger import logger
import numpy as np
from parser import output_dir, standard_params, sens_params, flux_params
from utils import list_files
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval
import pickle
from utils import get_filenames, show_1d_fit_QA, load_spec_data


def read_sensfunc_params():
    """
    Reads the star parameters needed to run the sensitivity function procedure

    Returns:
    --------
    exptime : float
        Exposure time of the standard star observation in seconds.

    airmass : float
        Airmass of the standard star observation. In sec(z) units.

    star_name : str
        Name of the standard star.

    flux_file : str
        Path to the reference spectrum of the standard star.
    """

    exptime = standard_params["exptime"]
    airmass = standard_params["airmass"]
    star_name = standard_params["starname"]
    flux_file = standard_params["flux_file_path"]

    return exptime, airmass, star_name, flux_file


def load_standard_star_spec():

    spectra = load_spec_data("standard")

    if len(spectra) > 1:
        logger.warning(
            "Multiple standard star spectra found. Software only supports one."
        )
        logger.warning(f"Using the first one - {list(spectra.keys())[0]}.")

    os.chdir(output_dir)


    wavelength = spectra[list(spectra.keys())[0]][0]
    counts = spectra[list(spectra.keys())[0]][1]

    logger.info("Standard star spectrum loaded.")

    return wavelength, counts


def load_ref_spec(file_path):
    """
    Loads the reference spectrum of the standard star.

    Parameters:
    file_path : str
        Path to the reference spectrum file.
        The file should have two columns: wavelength and flux.
        These should be in Ångström and AB Magnitude units.

    Returns:
    --------
    wavelength : numpy.ndarray
        Wavelength of the reference spectrum in Ångström.

    flux : numpy.ndarray
        Flux of the reference spectrum in AB Magnitude units.
    """

    logger.info("Loading standard star reference spectrum...")
    try:
        data = np.loadtxt(file_path)
    except FileNotFoundError:
        logger.error("Reference spectrum file not found.")
        logger.error("Check the path in the config file.")
        exit()

    wavelength = data[:, 0]
    flux = data[:, 1]

    logger.info("Reference spectrum loaded.")

    return wavelength, flux

def estimate_transmission_factor(wavelength, airmass, figsize=(18, 12), show_QA = False):
    """
    Estimates the transmission factor of the atmosphere at the given wavelength.

    Uses the extinction curve of the observatory, and 
    F_true / F_obs = 10 ** (0.4 * A * X) where A is the extinction AB mag / airmass
    and X is the airmass. I.e. the transmission factor is 10 ** (0.4 * A * X).

    Parameters:
    -----------
    wavelength : numpy.ndarray
        Wavelength of the observed spectrum in Ångström.

    airmass : float
        Airmass of the observation.

    figsize : tuple
        Size of the QA plot.

    show_QA : bool
        If True, the QA plot of the extinction curve and transmission factor is shown.

    Returns:
    --------
    transmission_factor : numpy.ndarray
        Transmission factor of the atmosphere at the given wavelength.
    """

    logger.info("Estimating the transmission factor of the atmosphere...")

    # load extinction file - should be AB magnitudes / air mass
    extinction_file_name = flux_params["path_extinction_curve"]

    # open the file

    #make sure we are in the output_dir
    os.chdir(output_dir)

    try:
        data = np.loadtxt(extinction_file_name)
    except FileNotFoundError:
        logger.error("Extinction file not found.")
        logger.error("Check the path in the config file.")
        exit()

    wavelength_ext = data[:, 0]
    extinction_data = data[:, 1]

    # interpolate the extinction file onto the wavelength grid of the object spectrum
    f = interp1d(wavelength_ext, extinction_data, kind="cubic")
    ext_interp1d = f(wavelength)

    # multiply the extinction by the airmass
    extinction = ext_interp1d * airmass

    transmission_factor = 10 ** (0.4 * extinction)

    if show_QA:

        # plot the transmission factor and extinction curve for QA purposes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        ax1.plot(wavelength_ext, extinction_data, color = "black", label="Extinction Curve")
        ax1.set_xlabel("Wavelength (Å)")
        ax1.set_ylabel("Extinction (AB mag / airmass)")
        ax1.legend()

        ax2.plot(wavelength, 1/transmission_factor, color = "black", label="Calculated Transmission Factor")
        ax2.set_xlabel("Wavelength (Å)")
        ax2.set_ylabel("Transmission Factor (Observed flux / True flux)")
        ax2.legend()

        fig.suptitle(
            "Extinction curve and transmission factor of the atmosphere.\n"
            "These are calculated based on the user provided extinction curve for the observatory.\n"
            "Make sure the extinction curve is correct and the transmission factor is reasonable."
        )
        plt.show()

    return transmission_factor



def convert_from_AB_mag_to_flux(mag, ref_wavelength):
    """
    Converts from AB magnitudes to erg/s/cm^2/Å

    From Oke, J.B. 1974, ApJS, 27, 21

    Parameters:
    -----------
    mag : float
        AB magnitude of the standard star.

    ref_wavelength : numpy.ndarray
        Wavelength of the reference spectrum in Ångström.

    Returns:
    --------
    flux : numpy.ndarray
        Flux of the standard star in erg/s/cm^2/Å.
    """

    flux = 2.998e18 * 10 ** ((mag + 48.6) / (-2.5)) / (ref_wavelength ** 2)

    return flux


def refrence_counts_to_flux(wavelength, counts, ref_wavelength, ref_flux, exptime):
    """
    Estimates the conversion factors between counts and flux across the spectrum.
    Applies extinction correction for the conversion factors.

    Parameters:
    -----------
    wavelength : numpy.ndarray
        Wavelength of the observed spectrum in Ångström.

    counts : numpy.ndarray
        Counts of the observed spectrum.

    ref_wavelength : numpy.ndarray
        Wavelength of the reference spectrum in Ångström.

    ref_flux : numpy.ndarray
        Flux of the reference spectrum in AB Magnitude units.

    exptime : float
        Exposure time of the observation in seconds.

    Returns:
    --------    
    ref_wavelength_cropped : numpy.ndarray
        Cropped wavelength of the reference spectrum in Ångström.
    
    conv_factors : numpy.ndarray
        Conversion factors between counts and flux across the spectrum.
    """

    # firstly, convert the reference spectrum to flux units
    ref_flux_converted = convert_from_AB_mag_to_flux(ref_flux, ref_wavelength)

    # convert counts to counts per second - still prior to extinction correction
    counts_pr_sec_with_atmosphere = counts / exptime

    # Estimate the transmission factor of the atmosphere at the given wavelength
    # and apply it to the observed spectrum

    transmission_factor = estimate_transmission_factor(wavelength, standard_params["airmass"], show_QA=True)

    counts_pr_sec = counts_pr_sec_with_atmosphere * transmission_factor

    # Crop the reference spectrum to the wavelength range of the observed spectrum
    ref_flux_converted_croppped = ref_flux_converted[
        (ref_wavelength >= np.min(wavelength)) & (ref_wavelength <= np.max(wavelength))
    ]
    ref_wavelength_cropped = ref_wavelength[
        (ref_wavelength >= np.min(wavelength)) & (ref_wavelength <= np.max(wavelength))
    ]

    # Interpolate counts per second onto reference wavelength
    interpolated_counts = interp1d(wavelength, counts_pr_sec)(ref_wavelength_cropped)

    # Estimate the conversion factors between counts and flux across the spectrum
    conv_factors = ref_flux_converted_croppped / interpolated_counts

    
    return ref_wavelength_cropped, conv_factors


def fit_sensfunc(wavelength, sens_points):
    """
    Fits the sensitivity function to the estimated conversion factors in
    `refrence_counts_to_flux` function. The fit is performed using Chebyshev
    polynomials in log space.

    Parameters:
    -----------
    wavelength : numpy.ndarray
        Wavelength of the reference spectrum in Ångström.

    sens_points : numpy.ndarray
        Conversion factors between counts and flux across the spectrum.

    Returns:
    --------
    coeff : numpy.ndarray
        Coefficients of the Chebyshev polynomial fit to the conversion factors.
    """
    # Load chebyshev degree
    fit_degree = sens_params["fit_order"]

    # Try to fit in log space

    sens_points_log = np.log10(sens_points)

    coeff = chebfit(wavelength, sens_points_log, deg = fit_degree)

    fit_eval = 10**(chebval(wavelength, coeff))

    residuals = sens_points - fit_eval

    show_1d_fit_QA(
        wavelength, 
        sens_points, 
        x_fit_values=wavelength, 
        y_fit_values=fit_eval,
        residuals=residuals,
        x_label="Wavelength (Å)",
        y_label="Conversion factor",
        legend_label="Fit",
        title=f"Sensitivity function fit of order {fit_degree}."
        "\nInspect the fit and residuals. "
        "Deviations around the edges are hard to avoid and should be okay." 
        "\nChange the fit degree in the config file if needed."
        "\nNote that the fit is performed in log space."
        )

    
    return coeff


def flux_standard_QA(coeff, wavelength, counts, ref_wavelength, ref_flux, figsize=(18, 12)):

    """
    Flux calibrates the standard star spectrum and compares it to the reference spectrum.
    This is done for QA purposes in order to check the validity of the sensitivity function.
    """

    # Calculate the conversion factors, convert back from log space.
    conv_factors = 10**chebval(wavelength, coeff)

    # Estimate the transmission factor of the atmosphere at the given wavelength
    # and apply it to the observed spectrum
    transmission_factor = estimate_transmission_factor(wavelength, standard_params["airmass"])

    # Flux the standard star spectrum
    fluxed_counts = (counts * transmission_factor / standard_params["exptime"]) * conv_factors

    # Convert the reference spectrum to flux units
    converted_ref_spec = convert_from_AB_mag_to_flux(ref_flux, ref_wavelength)

    plt.figure(figsize=figsize)

    plt.plot(wavelength, fluxed_counts, color = "green", label="Fluxed standard star spec")
    plt.plot(ref_wavelength, converted_ref_spec, color = "black", label="Reference spectrum")
    plt.legend()
    plt.title(
        "Fluxed standard star spectrum vs reference spectrum.\n"
        "Check that the observed spectrum resembles the refference spectrum strongly -"
        "this valides the correctness of the sensitivity function.\n"
        "Deviations around the edges are hard to avoid and should be okay."

    )
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (erg/s/cm^2/Å)")
    plt.show()



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

    ref_wavelength, ref_flux = load_ref_spec(flux_file)

    logger.info("Estimating the conversion factors between counts and flux...")

    ref_wavelength_cropped, sens_points = refrence_counts_to_flux(
        obs_wavelength, obs_counts, ref_wavelength, ref_flux, exptime
    )

    logger.info("Fitting the sensitivity function...")

    coeff = fit_sensfunc(ref_wavelength_cropped, sens_points)

    flux_standard_QA(coeff, obs_wavelength, obs_counts, ref_wavelength, ref_flux)

    write_sensfunc_to_disc(coeff)

    logger.info("Sensitivity function procedure done.")
    print("----------------------------\n")


if __name__ == "__main__":
    run_sensitivity_function()
