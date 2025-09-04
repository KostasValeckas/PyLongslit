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
        error = np.abs((10 ** fit(wavelength)) * np.log(10) * RMS_residuals_log)
    else:
        fit_eval = 10 ** (chebval(wavelength, fit))
        error = np.abs(
            (10 ** chebval(wavelength, fit)) * np.log(10) * RMS_residuals_log
        )

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

    If multiple standard star spectra are found in the output directory,
    the first one is used.

    Returns:
    --------
    wavelength : numpy.ndarray
        Wavelength of the standard star spectrum in Ångström.

    counts : numpy.ndarray
        Counts of the standard star spectrum.
    """

    from pylongslit.logger import logger
    from pylongslit.utils import load_spec_data

    logger.info("Loading standard star 1d spectrum...")

    spectra = load_spec_data("standard")

    if len(spectra) == 0:
        logger.error("No standard star spectrum found.")
        logger.error(
            "Run the extraction procedure first to extract the standard star spectrum."
        )
        exit()

    if len(spectra) > 1:
        logger.warning(
            "Multiple standard star spectra found. Software only supports one."
        )
        logger.warning(f"Using the first one - {list(spectra.keys())[0]}.")


    # the spectra dictionary has the filename as the key and the spectrum and
    # wavelength as the values.
    wavelength = spectra[list(spectra.keys())[0]][0]
    counts = spectra[list(spectra.keys())[0]][1]

    logger.info(f"Standard star spectrum loaded from {list(spectra.keys())[0]}.")

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

    logger.info("Loading standard star reference spectrum...")
    try:
        data = np.loadtxt(file_path)
    except FileNotFoundError:
        logger.error("Reference spectrum file not found.")
        logger.error("Check the path in the configuration file.")
        exit()

    wavelength = data[:, 0]
    flux = data[:, 1]

    logger.info(f"Reference spectrum loaded from {file_path}.")

    return wavelength, flux


def load_extinction_data():
    """
    Loads the atmospheric extintion data from the path provided in the configuration file.
    This file should contain the extinction curve of the observatory.
    It shoulf have two columns: wavelength and extinction in units of Ångström and AB mag / airmass.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir, flux_params

    # load extinction file - should be AB magnitudes / airmass
    extinction_file_name = flux_params["path_extinction_curve"]

    logger.info(f"Loading the extinction curve from {extinction_file_name}...")

    # open the file

    try:
        data = np.loadtxt(extinction_file_name)
    except FileNotFoundError:
        logger.error("Extinction file not found.")
        logger.error("Check the path in the configuration file.")
        exit()

    wavelength_ext = data[:, 0]
    extinction_data = data[:, 1]

    return wavelength_ext, extinction_data


def crop_all_spec(obs_wave, obs_count, ref_wave, ref_spec, ext_wave, ext_data):
    """
    Crops all data used in the sensitivity function procedure to the same wavelength range,
    i.e. the intersection of the wavelength ranges of the observed spectrum, reference spectrum,
    and the extinction curve. The data is then extrapolated to the same wavelength array,
    as this is needed for the sensitivity function production.

    Parameters:
    -----------
    obs_wave : numpy.ndarray
        Wavelength of the observed spectrum in Ångström.

    obs_count : numpy.ndarray
        Counts of the observed spectrum.

    ref_wave : numpy.ndarray
        Wavelength of the reference spectrum in Ångström.

    ref_spec : numpy.ndarray
        Flux of the reference spectrum in AB Magnitude units.

    ext_wave : numpy.ndarray
        Wavelength of the extinction curve in Ångström.

    ext_data : numpy.ndarray
        Extinction data in AB mag / airmass.

    Returns:
    --------
    global_wavelength : numpy.ndarray
        Wavelength array to which all data is extrapolated.

    obs_count_cropped : numpy.ndarray
        Cropped and extrapolated observed spectrum.

    ref_spec_cropped : numpy.ndarray
        Cropped and extrapolated reference spectrum.

    ext_data_cropped : numpy.ndarray
        Cropped and extrapolated extinction curve.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import developer_params

    logger.info(
        "Cropping all sensitivity calibration data to the same wavelength range..."
    )

    # find the smallest and largest wavelengths values and crop all data to this range
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

    logger.info("Extrapolating to the same wavelength range...")

    f = interp1d(
        obs_wave_cropped, obs_count_cropped, kind="cubic", fill_value="extrapolate"
    )
    obs_count_cropped = f(global_wavelength)

    f = interp1d(
        ref_wave_cropped, ref_spec_cropped, kind="cubic", fill_value="extrapolate"
    )
    ref_spec_cropped = f(global_wavelength)

    f = interp1d(
        ext_wave_cropped, ext_data_cropped, kind="cubic", fill_value="extrapolate"
    )
    ext_data_cropped = f(global_wavelength)

    logger.info("All data cropped and extrapolated to the same wavelength range.")

    if developer_params["debug_plots"]:
        plt.figure(figsize=(10, 6))
        plt.plot(
            global_wavelength,
            obs_count_cropped,
            "o",
            color="black",
            label="Observed Spectrum",
        )
        plt.plot(
            global_wavelength,
            ref_spec_cropped,
            "o",
            color="red",
            label="Reference Spectrum",
        )
        plt.plot(
            global_wavelength,
            ext_data_cropped,
            "o",
            color="blue",
            label="Extinction Curve",
        )
        plt.axvline(
            min_global,
            color="black",
            linestyle="--",
            linewidth=0.5,
            label="Cropped Range",
        )
        plt.axvline(max_global, color="black", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.title("Cropped spectra for sensitivity function procedure.")
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Counts / Flux")
        plt.show()

    return global_wavelength, obs_count_cropped, ref_spec_cropped, ext_data_cropped


def estimate_transmission_factor(
    wavelength, airmass, ext_data, figsize=(10, 6), show_QA=False
):
    """
    Estimates the transmission factor of the atmosphere at the given wavelength.

    All the data is assumed to be extrapolaed to the same wavelength array
    at this point.

    Uses the extinction curve of the observatory, and
    F_true / F_obs = 10 ** (0.4 * A * X) "from Ryden, B. and Peterson, B.M. (2020)
    Foundations of Astrophysics. Cambridge: Cambridge University Press",
    where A is the extinction AB mag / airmass
    and X is the airmass. I.e. the transmission factor is 10 ** (0.4 * A * X).

    Parameters:
    -----------
    wavelength : numpy.ndarray
        Wavelength in Ångström.

    airmass : float
        Airmass of the observation.

    ext_data : numpy.ndarray
        Extinction data in AB mag / airmass.

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

    # see the formula above in the docstring
    transmission_factor = 10 ** (0.4 * extinction)

    logger.info("Transmission factor estimated.")
    # option to plot QA, ex. when running the sensitivity function procedure
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
        Flux in AB magnitude.

    ref_wavelength : numpy.ndarray
        Wavelength of the spectrum in Ångström.

    Returns:
    --------
    flux : numpy.ndarray
        Flux in erg/s/cm^2/Å.
    """

    flux = 2.998e18 * 10 ** ((mag + 48.6) / (-2.5)) / (ref_wavelength**2)

    return flux


def prep_senspoints(wavelength, sens_points, figsize=(10, 6)):
    """
    A method to crop the sensitivity points of the standard star spectrum,
    and convert to logspace.

    Allows the user to crop out any noisy edges and
    mask out any strong emission/absorption lines.

    Parameters:
    -----------
    wavelength : numpy.ndarray
        Wavelengths of the sensitivity points in Ångström.

    sens_points : numpy.ndarray
        Sensitivity points of the standard star spectrum.

    figsize : tuple, optional
        Size of the plot. Default is (10, 6).

    Returns:
    --------
    wavelength : numpy.ndarray
        Cropped wavelength array.

    sens_points_log : numpy.ndarray
        Cropped sensitivity points in log space.
    """
    from pylongslit.utils import interactively_crop_spec

    # we might need the original arrays later, so we copy them
    wavelength = wavelength.copy()
    sens_points = sens_points.copy()

    # Remove any sens_points <= 0 and corresponding wavelengths
    valid_indices = sens_points > 0
    wavelength = wavelength[valid_indices]
    sens_points = sens_points[valid_indices]

    # convert to logspace
    sens_points_log = np.log10(sens_points)

    # Interactively crop the spectrum edges
    smin, smax = interactively_crop_spec(
        wavelength,
        sens_points_log,
        x_label="Wavelength (Å)",
        y_label="Log Sensitivity Points ((erg/cm^2/Å) / counts)",
        title="Use the sliders to crop out any noisy edges. Close the plot when done.",
        figsize=figsize,
    )

    # Get the selected range
    min_wavelength = smin
    max_wavelength = smax
    valid_indices = (wavelength >= min_wavelength) & (wavelength <= max_wavelength)
    wavelength = wavelength[valid_indices]
    sens_points_log = sens_points_log[valid_indices]

    # the next plot is for masking any strong emission/ansorption lines

    fig, _ = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)
    (l,) = plt.plot(wavelength, sens_points_log, "o")
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Log Sensitivity Points ((erg/cm^2/Å) / counts)")
    plt.title(
        "Mask out any strong emission/absorption lines by clicking on the graph.\n"
        "You can click multiple times to mask out multiple regions.\n"
        "Close the plot when done.\n"
    )

    # this array will be used to mask out the selected regions
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

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()

    # Mask the selected regions and return
    wavelength = wavelength[~masked_indices]
    sens_points_log = sens_points_log[~masked_indices]

    return wavelength, sens_points_log


def fit_sensfunc(wavelength, sens_points_log):
    """
    Fits the sensitivity function to the estimated conversion factors
    (sensitivity points) in log space. The fit is either
    a Chebyshev polynomial or a B-spline, depending on the configuration file.

    Parameters:
    -----------
    wavelength : numpy.ndarray
        Wavelengths in Ångström.

    sens_points : numpy.ndarray
        Conversion factors ((erg/s/cm^2/Å)/(counts/s)) across the spectrum
        in log space.

    Returns:
    --------
    fit : astropy.modeling.polynomial.Chebyshev1D or scipy.interpolate.BSpline
        The fitted sensitivity function in log space.
        The type depends on the configuration file.

    RMS_residuals_log : float
        The RMS of the residuals of the sensitivity function fit in log space.
    """
    from pylongslit.parser import sens_params
    from pylongslit.utils import show_1d_fit_QA
    from pylongslit.logger import logger

    logger.info("Fitting the sensitivity function...")

    # load needed parameters from the configuration file
    fit_degree = sens_params["fit_order"]
    use_bspline = sens_params["use_bspline"]

    # B-spline option
    if use_bspline:
        logger.warning("Fitting the sensitivity function with a B-spline...")
        logger.warning("This is set in the configuration file.")
        logger.warning(
            "This option should only be used if the regular polynomial fit does not work well."
        )
        logger.warning("Watch out for over-fitting.")
        logger.warning(
            "The sensitivity function will not be well-behaved in the edges."
        )
        n_knots = sens_params["knots_bspline"]

        # Create the knots array

        t = np.concatenate(
            (
                np.repeat(wavelength[0], fit_degree + 1),  # k+1 knots at the beginning
                np.linspace(wavelength[1], wavelength[-2], n_knots),  # interior knots
                np.repeat(wavelength[-1], fit_degree + 1),  # k+1 knots at the end
            )
        )

        # fit and construct the spline
        spl = make_lsq_spline(wavelength, sens_points_log, t=t, k=fit_degree)
        fit = BSpline(spl.t, spl.c, spl.k)

        fit_eval = fit(wavelength)

        residuals_log = sens_points_log - fit(wavelength)

        RMS_residuals_log = np.sqrt(np.mean(residuals_log**2))

    # regular Chebyshev polynomial fit
    else:
        fit = chebfit(wavelength, sens_points_log, deg=fit_degree)

        fit_eval = chebval(wavelength, fit)

        residuals_log = sens_points_log - chebval(wavelength, fit)

        RMS_residuals_log = np.sqrt(np.mean(residuals_log**2))

    # different titles depending on fitting method
    if use_bspline:
        title = (
            f"B-spline sensitivity function fit for with {n_knots} interior knots, degree {fit_degree} (this is set in the configuration file).\n"
            "Residuals should be random, but might show some structure due to spectral lines in star spectrum.\n"
            "Aim to fit with the lowest amount of knots possible."
        )

    else:
        title = (
            f"Chebyshev sensitivity function fit of order {fit_degree} (this is set in the configuration file).\n"
            "Residuals should be random, but might show some structure due to spectral lines in star spectrum.\n"
            "Deviations around the edges are hard to avoid and should be okay."
        )

    # show the QA plot
    show_1d_fit_QA(
        wavelength,
        sens_points_log,
        x_fit_values=wavelength,
        y_fit_values=fit_eval,
        residuals=residuals_log,
        x_label="Wavelength (Å)",
        y_label="Sensitivity points log ((erg/cm^2/Å) / counts)",
        legend_label="Fit",
        title=title,
    )

    logger.info("Sensitivity function fitted.")

    return fit, RMS_residuals_log


def flux_standard_QA(
    fit,
    transmision_factor,
    wavelength,
    obs_count_cropped,
    ref_spec_cropped,
    good_wavelength_start,
    good_wavelength_end,
    figsize=(10, 6),
):
    """
    Flux calibrates the standard star spectrum and compares it to the reference spectrum.
    This is done for QA purposes in order to check the validity of the sensitivity function.

    The observed and refrence spectra are assumed to be cropped and extrapolated
    to the same wavelength grid.

    Parameters:
    -----------
    fit : astropy.modeling.polynomial.Chebyshev1D or scipy.interpolate.BSpline
        The fitted sensitivity function. The type depends on the configuration file.

    transmision_factor : numpy.ndarray
        Transmission factor of the atmosphere at the given wavelength.

    wavelength : numpy.ndarray
        Wavelength array in Ångström.

    obs_count_cropped : numpy.ndarray
        Cropped and extrapolated observed spectrum.

    ref_spec_cropped : numpy.ndarray
        Cropped and extrapolated reference spectrum.

    good_wavelength_start, good_wavelength_end : float
        Start and end of the wavelength range used in the sensitivity function procedure.
        Anything outside this range is not considered in the QA plot, as it
        is almost certainly extremely noisy.

    figsize : tuple, optional
        Size of the plot. Default is (10, 6).
    """

    from pylongslit.parser import standard_params, sens_params

    good_wavelength_indices = (wavelength >= good_wavelength_start) & (
        wavelength <= good_wavelength_end
    )

    wavelength = wavelength[good_wavelength_indices].copy()
    obs_count_cropped = obs_count_cropped[good_wavelength_indices].copy()
    ref_spec_cropped = ref_spec_cropped[good_wavelength_indices].copy()
    transmision_factor = transmision_factor[good_wavelength_indices].copy()

    # Calculate the conversion factors, convert back from log space.
    if sens_params["use_bspline"]:
        conv_factors = 10 ** fit(wavelength)
    else:
        conv_factors = 10 ** chebval(wavelength, fit)

    # Flux the standard star spectrum
    fluxed_counts = (
        obs_count_cropped * transmision_factor / standard_params["exptime"]
    ) * conv_factors

    # Convert the reference spectrum to flux units
    converted_ref_spec = convert_from_AB_mag_to_flux(ref_spec_cropped, wavelength)

    plt.figure(figsize=figsize)

    plt.plot(
        wavelength, fluxed_counts, color="green", label="Fluxed standard star spec"
    )
    plt.plot(wavelength, converted_ref_spec, color="black", label="Reference spectrum")
    plt.legend()
    plt.title(
        "Fluxed standard star spectrum vs reference spectrum.\n"
        "Check that the observed spectrum resembles the refference spectrum strongly -"
        "this valides the correctness of the sensitivity function.\n"
        "If not, revise the sensitivity function routine parameters in the configuration file, and restart.\n"
        "Deviations around the edges are hard to avoid and should be okay."
    )
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (erg/s/cm^2/Å)")
    plt.xlim(good_wavelength_start, good_wavelength_end)

    plt.show()


def write_sensfunc_to_disc(
    fit, RMS_residuals, good_wavelength_start, good_wavelength_end
):
    """
    Writes the results of the sensitivity function procedure to disk,
    in machine-readable format.

    Parameters:
    -----------
    fit : astropy.modeling.polynomial.Chebyshev1D or scipy.interpolate.BSpline
        The fitted sensitivity function. The type depends on the configuration file.

    RMS_residuals : float
        The RMS of the residuals of the sensitivity function fit in log space.

    good_wavelength_start, good_wavelength_end : float
        Start and end of the wavelength range used in the sensitivity function procedure.
        Anything outside this range is not considered in the QA plot, as it
        is almost certainly extremely noisy.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Writing sensitivity function to disk...")

    # get current working directory
    cwd = os.getcwd()

    os.chdir(output_dir)

    output = (fit, RMS_residuals, good_wavelength_start, good_wavelength_end)

    with open("sensfunc.dat", "wb") as f:
        pickle.dump(output, f)

    logger.info(
        f"Sensitivity function fitting results written to {output_dir}, filename : sensfunc.dat."
    )

    # change back to the original working directory
    # this is useful when the user uses relative pathes in the configuration file
    os.chdir(cwd)


def load_sensfunc_from_disc():
    """
    Loads the sensitivity function from disk.

    Returns:
    --------
    fit : astropy.modeling.polynomial.Chebyshev1D or scipy.interpolate.BSpline
        The fitted sensitivity function. The type depends on the configuration file.

    RMS_residuals : float
        The RMS of the residuals of the sensitivity function fit in log space.

    good_wavelength_start, good_wavelength_end : float
        Start and end of the wavelength range used in the sensitivity function procedure.
        Anything outside this range is not considered in the QA plot, as it
        is almost certainly extremely noisy.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    logger.info("Loading sensitivity function from disk...")

    # get current working directory
    cwd = os.getcwd()

    os.chdir(output_dir)

    try:
        with open("sensfunc.dat", "rb") as f:
            out = pickle.load(f)
    except FileNotFoundError:
        logger.error("Sensitivity function file not found.")
        logger.error("Run the sensitivity function procedure first.")
        exit()

    fit, error, good_wavelength_start, good_wavelength_end = out

    logger.info("Sensitivity function loaded.")

    # change back to the original working directory
    # this is useful when the user uses relative pathes in the configuration file
    os.chdir(cwd)

    return fit, error, good_wavelength_start, good_wavelength_end


def run_sensitivity_function():

    from pylongslit.logger import logger

    logger.info("Staring the process of producing the sensitivity function...")

    logger.info("Reading the standard star parameters...")
    exptime, airmass, flux_file = read_sensfunc_params()

    obs_wavelength, obs_counts = load_standard_star_spec()

    ref_wavelength, ref_flux = load_ref_spec(flux_file)

    wavelength_ext, data_ext = load_extinction_data()

    global_wavelength, obs_count_cropped, ref_spec_cropped, ext_data_cropped = (
        crop_all_spec(
            obs_wavelength,
            obs_counts,
            ref_wavelength,
            ref_flux,
            wavelength_ext,
            data_ext,
        )
    )

    transmision_factor = estimate_transmission_factor(
        global_wavelength, airmass, ext_data_cropped, show_QA=True
    )

    # convert the observed spectrum to counts per second
    counts_pr_sec = obs_count_cropped / exptime

    logger.info("Preparing the sensitivity points for fitting...")
    logger.info("Adjusting the observed spectrum for atmospheric extinction...")
    counts_pr_sec = counts_pr_sec * transmision_factor

    logger.info("Estimating the conversion factors between flux and counts...")

    # convert the reference spectrum from AB to erg/s/cm^2/Å flux units
    ref_spec_flux = convert_from_AB_mag_to_flux(ref_spec_cropped, global_wavelength)

    sens_points = ref_spec_flux / counts_pr_sec

    logger.info("Converting to log space and cropping...")
    # convert to log space and crop / remove line artifacts before fitting
    wavelength_sens, sens_points_log = prep_senspoints(global_wavelength, sens_points)

    # these are the good wavelength ranges for the sensitivity function,
    # anything outside this range is almost certainly extremely noisy.
    good_wavelength_start = wavelength_sens[0]
    good_wavelength_end = wavelength_sens[-1]

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

    write_sensfunc_to_disc(
        fit, RMS_residuals, good_wavelength_start, good_wavelength_end
    )

    logger.info("Sensitivity function procedure done.")
    print("----------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit sensitivity function procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_sensitivity_function()


if __name__ == "__main__":
    main()
