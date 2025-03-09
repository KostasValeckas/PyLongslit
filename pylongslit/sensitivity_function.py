import argparse
import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval
import pickle
from scipy.interpolate import make_lsq_spline, BSpline

def eval_sensfunc(fit, RMS_residuals_log, wavelength):
    """
    Takes the Chebyshev sensitivity taken in log space
    and evaluates it at the given wavelength. The error
    is also converted from log space.

    For derivation of the error propagation formula, see
    the docs.

    Parameters:
    -----------
    fit : scipy.interpolate.BSpline or numpy.ndarray
        Bspline or Chebyshev polynomial fit object to the sensitivity function,
        depending on the configuration file.

    RMS_residuals_log : float
        The RMS of the residuals of the sensitivity function fit in log space
        (this is the initial error in log space).

    wavelength : numpy.ndarray
        Wavelength at which to evaluate the sensitivity function.

    Returns:
    --------
    fit_eval : numpy.ndarray
        Evaluated sensitivity function at the given wavelength.

    error : numpy.ndarray
        Error of the evaluated sensitivity function at the given wavelength.
    """
    from pylongslit.parser import sens_params

    # the evaluation is different if bspline was used or not.
    # It is determined from the configuration file.
    if sens_params["use_bspline"]:
        fit_eval = 10 ** fit(wavelength)
        error = np.abs((10 ** fit(wavelength))*np.log(10)*RMS_residuals_log)
    else:
        fit_eval = 10 ** (chebval(wavelength, fit))
        error = np.abs((10 ** chebval(wavelength, fit))*np.log(10)*RMS_residuals_log)

    return fit_eval, error

def read_sensfunc_params():
    """
    Reads the star parameters needed to run the sensitivity function procedure, 
    taken from the configuration file.

    Returns:
    --------
    exptime : float
        Exposure time of the standard star observation in seconds.

    airmass : float
        Airmass of the standard star observation.

    flux_file : str
        Path to the reference spectrum of the standard star.
    """

    from pylongslit.parser import standard_params

    exptime = standard_params["exptime"]
    airmass = standard_params["airmass"]
    flux_file = standard_params["flux_file_path"]

    return exptime, airmass, flux_file


def load_standard_star_spec():
    """
    Loads the extracted 1D spectrum of the standard star (counts/Å).
    
    If multiple standard star spectra are found, the first one is used.

    Returns:
    --------
    wavelength : numpy.ndarray
        Wavelength of the standard star spectrum in Ångström.

    counts : numpy.ndarray
        Counts of the standard star spectrum.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir
    from pylongslit.utils import load_spec_data

    spectra = load_spec_data("standard")

    if len(spectra) == 0:
        logger.error("No standard star spectrum found.")
        logger.error("Run the extraction procedure first to extract the standard star spectrum.")
        exit()

    if len(spectra) > 1:
        logger.warning(
            "Multiple standard star spectra found. Software only supports one."
        )
        logger.warning(f"Using the first one - {list(spectra.keys())[0]}.")

    os.chdir(output_dir)

    # the spectra dictionary has the filename as the key and the spectrum and 
    # wavelength as the values.
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

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir, standard_params, sens_params, flux_params
    from pylongslit.utils import show_1d_fit_QA

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


def load_extinction_data():

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir, flux_params

    # load extinction file - should be AB magnitudes / air mass
    extinction_file_name = flux_params["path_extinction_curve"]

    # open the file

    # make sure we are in the output_dir
    os.chdir(output_dir)

    try:
        data = np.loadtxt(extinction_file_name)
    except FileNotFoundError:
        logger.error("Extinction file not found.")
        logger.error("Check the path in the config file.")
        exit()

    wavelength_ext = data[:, 0]
    extinction_data = data[:, 1]

    return wavelength_ext, extinction_data


def crop_all_spec(obs_wave, obs_count, ref_wave, ref_spec, ext_wave, ext_data):

    """
    Crops all data used in the sensitivity function procedure to the same wavelength range,
    i.e. the intersection of the wavelength ranges of the observed spectrum, reference spectrum,
    and the extinction curve.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import developer_params

    min_array = [np.min(obs_wave), np.min(ref_wave), np.min(ext_wave)]
    max_array = [np.max(obs_wave), np.max(ref_wave), np.max(ext_wave)]

    min_global = np.max(min_array)
    max_global = np.min(max_array)

    obs_count_cropped = obs_count[(obs_wave >= min_global) & (obs_wave <= max_global)]
    obs_wave_cropped = obs_wave[(obs_wave >= min_global) & (obs_wave <= max_global)]

    ref_spec_cropped = ref_spec[(ref_wave >= min_global) & (ref_wave <= max_global)]
    ref_wave_cropped = ref_wave[(ref_wave >= min_global) & (ref_wave <= max_global)]

    ext_data_cropped = ext_data[(ext_wave >= min_global) & (ext_wave <= max_global)]
    ext_wave_cropped = ext_wave[(ext_wave >= min_global) & (ext_wave <= max_global)]

    # extrapolate all data to the same wavelength array
    global_wavelength = np.arange(min_global, max_global, 1)

    f = interp1d(obs_wave_cropped, obs_count_cropped, kind="cubic", fill_value="extrapolate")
    obs_count_cropped = f(global_wavelength)

    f = interp1d(ref_wave_cropped, ref_spec_cropped, kind="cubic", fill_value="extrapolate")
    ref_spec_cropped = f(global_wavelength)

    f = interp1d(ext_wave_cropped, ext_data_cropped, kind="cubic", fill_value="extrapolate")
    ext_data_cropped = f(global_wavelength)

    logger.info("All spectra cropped to the same wavelength range.")



    if developer_params["debug_plots"]:
        plt.figure(figsize=(18, 12))
        plt.plot(global_wavelength, obs_count_cropped, "o", color="black", label="Observed Spectrum")
        plt.plot(global_wavelength, ref_spec_cropped, "o", color="red", label="Reference Spectrum")
        plt.plot(global_wavelength, ext_data_cropped, "o", color="blue", label="Extinction Curve")
        plt.axvline(min_global, color='black', linestyle='--', linewidth=0.5, label = "Cropped Range")
        plt.axvline(max_global, color='black', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.title("Cropped spectra for sensitivity function procedure.")
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Counts / Flux")
        plt.show()
    

    return global_wavelength, obs_count_cropped, ref_spec_cropped, ext_data_cropped


def estimate_transmission_factor(
    wavelength, airmass, ext_data, figsize=(18, 12), show_QA=False
):
    """
    Estimates the transmission factor of the atmosphere at the given wavelength.

    Uses the extinction curve of the observatory, and
    F_true / F_obs = 10 ** (0.4 * A * X) "from Ryden, B. and Peterson, B.M. (2020) 
    Foundations of Astrophysics. Cambridge: Cambridge University Press",
    where A is the extinction AB mag / airmass
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

    from pylongslit.logger import logger

    logger.info("Estimating the transmission factor of the atmosphere...")

    # multiply the extinction by the airmass
    extinction = ext_data * airmass

    transmission_factor = 10 ** (0.4 * extinction)

    if show_QA:

        # plot the transmission factor and extinction curve for QA purposes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        ax1.plot(wavelength, ext_data, color="black", label="Extinction Curve")
        ax1.set_xlabel("Wavelength (Å)")
        ax1.set_ylabel("Extinction (AB mag / airmass)")
        ax1.legend()

        ax2.plot(
            wavelength,
            transmission_factor,
            color="black",
            label=f"Calculated Transmission Factor for airmass {airmass}",
        )
        ax2.set_xlabel("Wavelength (Å)")
        ax2.set_ylabel("Transmission Factor (True flux / Observed flux)")
        ax2.legend()

        fig.suptitle(
            "Extinction curve and transmission factor of the atmosphere.\n"
            "These are calculated based on the user provided extinction curve for the observatory and the airmass for the observation.\n"
            "Revise these parameters in the configuration file, if the extinction curve or transmission factor is not reasonable."
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

    flux = 2.998e18 * 10 ** ((mag + 48.6) / (-2.5)) / (ref_wavelength**2)

    return flux


def crop_senspoints(wavelength, sens_points, figsize=(18, 18)):
    from pylongslit.utils import interactively_crop_spec


    wavelength = wavelength.copy()
    sens_points = sens_points.copy()

    # Try to fit in log space

    # Remove any sens_points <= 0 and corresponding wavelengths
    valid_indices = sens_points > 0
    wavelength = wavelength[valid_indices]
    sens_points = sens_points[valid_indices]

    sens_points_log = np.log10(sens_points)

    # Interactively crop the spectrum
    smin, smax = interactively_crop_spec(
        wavelength, 
        sens_points_log,
        x_label="Wavelength (Å)",
        y_label="Log Sensitivity Points",
        title="Use the sliders to crop out any noisy edges. Close the plot when done.",
        figsize=figsize,
    )

    # Get the final selected range
    min_wavelength = smin
    max_wavelength = smax
    valid_indices = (wavelength >= min_wavelength) & (wavelength <= max_wavelength)
    wavelength = wavelength[valid_indices]
    sens_points_log = sens_points_log[valid_indices]

    # the next plot is for masking any strong emission/ansorption lines

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)
    (l,) = plt.plot(wavelength, sens_points_log, "o")
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Log Sensitivity Points")
    plt.title(
        "Mask out any strong emission/absorption lines by clicking on the graph.\n"
        "You can click multiple times to mask out multiple regions.\n"
        "Close the plot when done.\n"
    )

    # Add interactive masking
    masked_indices = np.zeros_like(wavelength, dtype=bool)

    def onclick(event):
        if event.inaxes is not None:
            x = event.xdata
            # Find the closest wavelength index
            idx = (np.abs(wavelength - x)).argmin()
            # Mask +/- 20 points
            mask_range = 20
            start_idx = max(0, idx - mask_range)
            end_idx = min(len(wavelength), idx + mask_range + 1)
            masked_indices[start_idx:end_idx] = True
            # Update plot
            l.set_xdata(wavelength[~masked_indices])
            l.set_ydata(sens_points_log[~masked_indices])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    # Mask the selected regions
    wavelength = wavelength[~masked_indices]
    sens_points_log = sens_points_log[~masked_indices]

    return wavelength, sens_points_log

    




def fit_sensfunc(wavelength, sens_points_log):
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
    from pylongslit.parser import sens_params
    from pylongslit.utils import show_1d_fit_QA
    # Load chebyshev degree
    fit_degree = sens_params["fit_order"]
    use_bspline = sens_params["use_bspline"]
    
    if use_bspline:
        n_knots = sens_params["knots_bspline"]
                # Create the knots array

        t = np.concatenate(
            (
                np.repeat(wavelength[0], fit_degree + 1),  # k+1 knots at the beginning
                np.linspace(
                    wavelength[1], wavelength[-2], n_knots
                ),  # interior knots
                np.repeat(wavelength[-1], fit_degree + 1),  # k+1 knots at the end
            )
        )

        # fit and construct the spline
        spl = make_lsq_spline(wavelength, sens_points_log, t=t, k=fit_degree)
        fit = BSpline(spl.t, spl.c, spl.k)

        fit_eval = fit(wavelength)

        residuals_log = sens_points_log - fit(wavelength)

        RMS_residuals_log = np.sqrt(np.mean(residuals_log ** 2))

    else:    
        fit = chebfit(wavelength, sens_points_log, deg=fit_degree)

        fit_eval = chebval(wavelength, fit)

        residuals_log = sens_points_log - chebval(wavelength, fit)

        RMS_residuals_log = np.sqrt(np.mean(residuals_log ** 2))

    if use_bspline:
        title = f"B-spline sensitivity function fit for with {n_knots} interior knots, degree {fit_degree} (this is set in the configuration file).\n" \
                "You should aim for very little to no large-scale structure in the residuals, " \
                "with the lowest amount of knots possible."
        
    else:
        title = f"Chebyshev sensitivity function fit of order {fit_degree} (this is set in the configuration file).\n" \
                "Inspect the fit and residuals. Residuals should be random, but might show some structure due to spectral lines in star spectrum.\n" \
                "Deviations around the edges are hard to avoid and should be okay."


    show_1d_fit_QA(
        wavelength,
        sens_points_log,
        x_fit_values=wavelength,
        y_fit_values=fit_eval,
        residuals=residuals_log,
        x_label="Wavelength (Å)",
        y_label="Sensitivity points log",
        legend_label="Fit",
        title=title
    )

    return fit, RMS_residuals_log


def flux_standard_QA(
        fit,
        transmision_factor,
        global_wavelength,
        obs_count_cropped,
        ref_spec_cropped,
        good_wavelength_start,
        good_wavelength_end,
        figsize=(18, 12)
    ):
    """
    Flux calibrates the standard star spectrum and compares it to the reference spectrum.
    This is done for QA purposes in order to check the validity of the sensitivity function.
    """

    from pylongslit.parser import standard_params, sens_params

    good_wavelength_indices = (global_wavelength >= good_wavelength_start) & (global_wavelength <= good_wavelength_end)

    global_wavelength = global_wavelength[good_wavelength_indices].copy()
    obs_count_cropped = obs_count_cropped[good_wavelength_indices].copy()
    ref_spec_cropped = ref_spec_cropped[good_wavelength_indices].copy()
    transmision_factor = transmision_factor[good_wavelength_indices].copy()

    # Calculate the conversion factors, convert back from log space.
    if sens_params["use_bspline"]:
        conv_factors = 10 ** fit(global_wavelength)
    else:
        conv_factors = 10 ** chebval(global_wavelength, fit)


    # Flux the standard star spectrum
    fluxed_counts = (
        obs_count_cropped * transmision_factor / standard_params["exptime"]
    ) * conv_factors


    # Convert the reference spectrum to flux units
    converted_ref_spec = convert_from_AB_mag_to_flux(ref_spec_cropped, global_wavelength)

    plt.figure(figsize=figsize)

    plt.plot(
        global_wavelength, fluxed_counts, color="green", label="Fluxed standard star spec"
    )
    plt.plot(
        global_wavelength, converted_ref_spec, color="black", label="Reference spectrum"
    )
    plt.legend()
    plt.title(
        "Fluxed standard star spectrum vs reference spectrum.\n"
        "Check that the observed spectrum resembles the refference spectrum strongly -"
        "this valides the correctness of the sensitivity function.\n"
        "Deviations around the edges are hard to avoid and should be okay."
    )
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (erg/s/cm^2/Å)")
    plt.xlim(good_wavelength_start, good_wavelength_end)
    
    plt.show()


def write_sensfunc_to_disc(coeff, RMS_residuals, good_wavelength_start, good_wavelength_end):

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Writing sensitivity function coefficients and error to disk...")

    os.chdir(output_dir)

    output = (coeff, RMS_residuals)

    with open("sensfunc.dat", "wb") as f:
        pickle.dump(output, f)

    logger.info(
        f"Sensitivity function fitting results written to {output_dir}, filename : sensfunc.dat."
    )


def load_sensfunc_from_disc():

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Loading sensitivity function from disk...")

    os.chdir(output_dir)

    try:
        with open("sensfunc.dat", "rb") as f:
            out = pickle.load(f)
            print(out)
    except FileNotFoundError:
        logger.error("Sensitivity function file not found.")
        logger.error("Run the sensitivity function procedure first.")
        exit()

    coeff, error, good_wavelength_start, good_wavelength_end = out

    logger.info("Sensitivity function loaded.")

    return coeff, error, good_wavelength_start, good_wavelength_end


def run_sensitivity_function():

    from pylongslit.logger import logger

    logger.info("Staring the process of producing the sensitivity function...")

    exptime, airmass, flux_file = read_sensfunc_params()

    obs_wavelength, obs_counts = load_standard_star_spec()

    ref_wavelength, ref_flux = load_ref_spec(flux_file)

    wavelength_ext, data_ext = load_extinction_data()

    global_wavelength, obs_count_cropped, ref_spec_cropped, ext_data_cropped = crop_all_spec(
        obs_wavelength, obs_counts, ref_wavelength, ref_flux, wavelength_ext, data_ext
    )

    transmision_factor = estimate_transmission_factor(
        global_wavelength, airmass, ext_data_cropped, show_QA=True
    )

    counts_pr_sec = obs_count_cropped / exptime

    counts_pr_sec = counts_pr_sec * transmision_factor

    logger.info("Estimating the conversion factors between counts and flux...")

    ref_spec_flux = convert_from_AB_mag_to_flux(ref_spec_cropped, global_wavelength)

    sens_points = ref_spec_flux / counts_pr_sec

    wavelength_sens, sens_points_log = crop_senspoints(global_wavelength, sens_points)

    good_wavelength_start = wavelength_sens[0]
    good_wavelength_end = wavelength_sens[-1]

    logger.info("Fitting the sensitivity function...")

    fit, RMS_residuals = fit_sensfunc(wavelength_sens, sens_points_log)

    flux_standard_QA(
        fit,
        transmision_factor,
        global_wavelength,
        obs_count_cropped,
        ref_spec_cropped,
        good_wavelength_start,
        good_wavelength_end,
    )


    write_sensfunc_to_disc(fit, RMS_residuals, good_wavelength_start, good_wavelength_end)

    logger.info("Sensitivity function procedure done.")
    print("----------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit sensitivity function procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_sensitivity_function()


if __name__ == "__main__":
    main()
