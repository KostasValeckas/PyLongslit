import numpy as np
import argparse
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import make_lsq_spline, BSpline
from matplotlib.widgets import Slider

"""
Module for creating a master flat from from raw flat frames.
"""


def normalize_spectral_response(medianflat):

    from pylongslit.parser import detector_params, wavecalib_params
    from pylongslit.wavecalib import get_tilt_fit_from_disc, get_wavelen_fit_from_disc, construct_wavelen_map
    from pylongslit.utils import wavelength_sol, show_1d_fit_QA, check_rotation, flip_and_rotate

    y_size = detector_params["ysize"]
    x_size = detector_params["xsize"]

    # offset is needed for cases where the detector middle is a bad place to
    # take a cut. 
    middle_offset = wavecalib_params["offset_middle_cut"]


    # TODO: right now we flip and rotate to universal frame configuration
    # when master frames are reduced, so for flats we need to
    # account for that here. Consider flipping the raw frames instead
    # for more readable code.

    # extract the spectrum of the central 5 rows of the frame

    if detector_params["dispersion"]["spectral_dir"] == "x":

        # extract the spectrum of the central 5 rows of the frame
        middle_row = (y_size // 2) + middle_offset
        spectrum = np.mean(medianflat[middle_row - 2 : middle_row + 2, :], axis=0)
        spectral_array = np.arange(x_size)

        spectrum = spectrum[::-1]
        plt.plot(spectral_array, spectrum)  
        plt.show()


    else:

        # extract the spectrum of the central 5 rows of the frame
        middle_row = (x_size // 2) + middle_offset
        spectrum = np.mean(medianflat[:, middle_row - 2 : middle_row + 2], axis=1)
        spectral_array = np.arange(y_size)

    # read the wavecalib data
    wavelen_fit = get_wavelen_fit_from_disc()
    tilt_fit = get_tilt_fit_from_disc()

    # create a row of middle spacial pixel coordinates in order to map
    # wavelengths.
    middlew_row_array = np.full(len(spectral_array), middle_row)

    # create the wavelength array
    wavelength = wavelength_sol(
        spectral_array, middlew_row_array, wavelen_fit, tilt_fit
    )

    # TODO: make these errors more user - friendly
    # Check for NaN or infinite values
    if np.any(np.isnan(wavelength)) or np.any(np.isnan(spectrum)):
        raise ValueError("NaN values found in wavelength or spectrum.")
    if np.any(np.isinf(wavelength)) or np.any(np.isinf(spectrum)):
        raise ValueError("Infinite values found in wavelength or spectrum.")

    # Ensure wavelength is sorted
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("Wavelength values are not sorted in ascending order.")

    # TODO: make this a utils method, as it is used several places

    # Initial plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    (l,) = plt.plot(wavelength, spectrum, label="Flat-field lamp spectrum")
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Counts (ADU)")
    plt.legend()

    # Add sliders for selecting the range
    axcolor = "lightgoldenrodyellow"
    axmin = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    smin = Slider(
        axmin,
        "Min Wavelength",
        np.min(wavelength),
        np.max(wavelength),
        valinit=np.min(wavelength),
    )
    smax = Slider(
        axmax,
        "Max Wavelength",
        np.min(wavelength),
        np.max(wavelength),
        valinit=np.max(wavelength),
    )

    def update(val):
        min_wavelength = smin.val
        max_wavelength = smax.val
        valid_indices = (wavelength >= min_wavelength) & (wavelength <= max_wavelength)
        l.set_xdata(wavelength[valid_indices])
        l.set_ydata(spectrum[valid_indices])
        fig.canvas.draw_idle()

    smin.on_changed(update)
    smax.on_changed(update)

    plt.show()

    # Get the final selected range
    min_wavelength = smin.val
    max_wavelength = smax.val
    valid_indices = (wavelength >= min_wavelength) & (wavelength <= max_wavelength)
    wavelength_cut = wavelength[valid_indices]
    spectrum_cut = spectrum[valid_indices]

    num_interior_knots = len(wavelength_cut) // 50  # (len(wavelength) - 2)//2

    # Create the knots array
    t = np.concatenate(
        (
            np.repeat(wavelength_cut[0], 4),  # k+1 knots at the beginning
            np.linspace(
                wavelength_cut[1], wavelength_cut[-2], num_interior_knots
            ),  # interior knots
            np.repeat(wavelength_cut[-1], 4),  # k+1 knots at the end
        )
    )

    spl = make_lsq_spline(wavelength_cut, spectrum_cut, t=t, k=3)
    bspline = BSpline(spl.t, spl.c, spl.k)

    residuals = spectrum_cut - bspline(wavelength_cut)
    RMS = np.sqrt(np.mean(residuals ** 2))

    show_1d_fit_QA(
        wavelength_cut,
        spectrum_cut,
        x_fit_values=wavelength_cut,
        y_fit_values=bspline(wavelength_cut),
        residuals=residuals,
        x_label="Wavelength (Å)",
        y_label="Counts (ADU)",
        legend_label="Extracted flat-field lamp spectrum",
        title="Spectral response B-spline fit",
    )

    wave_map = construct_wavelen_map(wavelen_fit, tilt_fit)

    transpose, flip = check_rotation()

    spectral_response_model = bspline(wave_map)

    bpm = np.zeros_like(spectral_response_model, dtype=bool)

    bpm[wave_map < min_wavelength] = True
    bpm[wave_map > max_wavelength] = True

    spectral_response_model[bpm] = 1.0

    spectral_response_model = flip_and_rotate(
        spectral_response_model, transpose, flip, inverse=True
    )
    bpm = flip_and_rotate(bpm, transpose, flip, inverse=True)

    return spectral_response_model, bpm, RMS


def normalize_spacial_response(medianflat):

    from pylongslit.parser import detector_params

    y_size = detector_params["ysize"]
    x_size = detector_params["xsize"]

    # TODO: right now we flip and rotate to universal frame configuration
    # when master frames are reduced, so for flats we need to
    # account for that here. Consider flipping the raw frames instead
    # for more readable code.

    # extract the spectrum of the central 5 rows of the frame

    spectral_axis = 0 if detector_params["dispersion"]["spectral_dir"] == "x" else 1
    spacial_axis = 1 if detector_params["dispersion"]["spectral_dir"] == "x" else 0

    fig, ax = plt.subplots(5, 2, figsize=(10, 35))

    indices_to_plot = np.linspace(
        10, x_size if spectral_axis == 0 else y_size, 10, endpoint=False, dtype=int
    )

    plot_num = 0

    spacial_model = np.zeros((y_size, x_size))

    residuals = np.zeros((y_size, x_size))

    for spacial_row_index in range(x_size) if spectral_axis == 0 else range(y_size):

        spacial_slice = (
            medianflat[:, spacial_row_index].copy()
            if spacial_axis == 1
            else medianflat[spacial_row_index, :].copy()
        )

        x_axis = np.arange(len(spacial_slice))

        num_interior_knots = len(x_axis) // 100  # (len(wavelength) - 2)//2

        # Create the knots array
        t = np.concatenate(
            (
                np.repeat(x_axis[0], 4),  # k+1 knots at the beginning
                np.linspace(
                    x_axis[1], x_axis[-2], num_interior_knots
                ),  # interior knots
                np.repeat(x_axis[-1], 4),  # k+1 knots at the end
            )
        )

        spl = make_lsq_spline(x_axis, spacial_slice, t=t, k=3)
        bspline = BSpline(spl.t, spl.c, spl.k)

        if spacial_axis == 1:
            spacial_model[:, spacial_row_index] = bspline(x_axis)
            residuals[:, spacial_row_index] = spacial_slice - bspline(x_axis)

        else:
            spacial_model[spacial_row_index, :] = bspline(x_axis)
            residuals[spacial_row_index, :] = spacial_slice - bspline(x_axis)

        if spacial_row_index in indices_to_plot:
            if plot_num <= 9:
                ax[plot_num // 2, plot_num % 2].plot(
                    x_axis[1:-1], spacial_slice[1:-1], label="Data"
                )
                ax[plot_num // 2, plot_num % 2].plot(
                    x_axis[1:-1], bspline(x_axis)[1:-1], label="Fit"
                )
                ax[plot_num // 2, plot_num % 2].set_title(
                    f"Spectral pixel: {spacial_row_index}"
                )
                ax[plot_num // 2, plot_num % 2].legend()
                plot_num += 1

    plt.suptitle(
        "Slit illumination B-spline fits at different spectral pixels", fontsize=16
    )
    fig.text(0.5, 0.04, "Spacial pixel", ha="center", fontsize=16)
    fig.text(
        0.04,
        0.5,
        "Normalized Counts (ADU)",
        va="center",
        rotation="vertical",
        fontsize=16,
    )
    plt.show()

    RMS = np.sqrt(np.mean(residuals ** 2))

    return spacial_model, RMS


def show_flat_norm_region():
    """
    Show the user defined flat normalization region.

    Fetches a raw flat frame from the user defined directory
    and displays the normalization region overlayed on it.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import flat_params
    from pylongslit.utils import show_flat

    logger.info(
        "Showing the normalization region on a raw flat frame for user inspection..."
    )

    show_flat()

    width = flat_params["norm_area_end_x"] - flat_params["norm_area_start_x"]
    height = flat_params["norm_area_end_y"] - flat_params["norm_area_start_y"]

    rect = Rectangle(
        (flat_params["norm_area_start_x"], flat_params["norm_area_start_y"]),
        width,
        height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
        linestyle="--",
        label="Region used for estimation of normalization factor",
    )
    plt.gca().add_patch(rect)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("Pixels in x-direction")
    plt.ylabel("Pixels in y-direction")
    plt.title(
        "Region used for estimation of normalization factor overlayed on a raw flat frame.\n"
        "The region should somewhat brightly illuminated with no abnormalities or artifacts.\n"
        "If it is not, check the normalization region definition in the config file."
    )
    plt.show()


def run_flats():
    """
    Driver for the flat-fielding procedure.

    The function reads the raw flat frames from the directory specified in the
    'flat_dir' parameter in the 'config.json' file. It then subtracts the bias and normalizes the frames
    by the median value of the frame. The final master flat-field is written to
    disc in the output directory.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import detector_params, flat_params, output_dir, data_params
    from pylongslit.utils import FileList, check_dimensions, open_fits, PyLongslit_frame
    from pylongslit.utils import list_files, load_bias
    from pylongslit.overscan import subtract_overscan_from_frame, detect_overscan_direction
    from pylongslit.stats import bootstrap_median_errors_framestack

    # Extract the detector parameters
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    use_overscan = detector_params["overscan"]["use_overscan"]

    # TODO: specify what direction is the spectral direction
    logger.info("Flat-field procedure running...")
    logger.info("Using the following parameters:")
    logger.info(f"xsize = {xsize}")
    logger.info(f"ysize = {ysize}")

    # read the names of the flat files from the directory
    file_list = FileList(flat_params["flat_dir"])

    logger.info(f"Found {file_list.num_files} flat frames.")
    logger.info(f"Files used for flat-fielding:")
    list_files(file_list)

    # Check if all files have the wanted dimensions
    # Will exit if they don't
    check_dimensions(file_list, xsize, ysize)

    # initialize a big array to hold all the flat frames for stacking
    bigflat = np.zeros((file_list.num_files, ysize, xsize), float)

    if use_overscan:

        overscan_dir = detect_overscan_direction()

    logger.info("Fetching the master bias frame...")

    BIASframe = load_bias()

    BIAS = np.array(BIASframe[0].data)
    logger.info("Master bias frame found and loaded.")

    print("\n------------------------------------------------------------\n")

    # loop over all the falt files, subtract bias and stack them in the bigflat array
    for i, file in enumerate(file_list):

        rawflat = open_fits(flat_params["flat_dir"], file)

        logger.info(f"Processing file: {file}")

        data = np.array(
            rawflat[data_params["raw_data_hdu_index"]].data, dtype=np.float64
        )

        # Subtract the bias
        if use_overscan:
            data = subtract_overscan_from_frame(data, overscan_dir)
        else:
            data = data - BIAS
            logger.info("Subtracted the bias.")

        
        bigflat[i] = data

        # close the file handler
        rawflat.close()

        logger.info(f"File {file} processed.\n")

    logger.info("Normalizing the final master flat-field....")

    # Calculate flat is median at each pixel
    medianflat = np.median(bigflat, axis=0)

    if file_list.num_files < 30 and (not flat_params["bootstrap_errors"]):
        logger.warning(
            f"Number of flat frames ({file_list.num_files}) is less than 30. Error estimation might not be accurate."
        )
        logger.warning("Please consider taking more flat frames or activating error bootstrapping in the config file.")
   
    if  not flat_params["bootstrap_errors"]:
        medianflat_error =  1.2533*np.std(bigflat, axis=0)/np.sqrt(file_list.num_files)

    else:
        medianflat_error = bootstrap_median_errors_framestack(bigflat)

    spectral_response_model, _, RMS_spectral = normalize_spectral_response(medianflat)

    spectral_normalized = medianflat / spectral_response_model


    medianflat_error = spectral_normalized * np.sqrt(
        ((medianflat_error / medianflat)) ** 2 + ((RMS_spectral/spectral_response_model) ** 2)
    )



    spectral_normalized[spectral_normalized < 0.5] = 1
    spectral_normalized[spectral_normalized > 1.5] = 1

    if not flat_params["skip_spacial"]:

        spacial_response_model, RMS_spacial = normalize_spacial_response(spectral_normalized)
        master_flat = spectral_normalized / spacial_response_model
        medianflat_error = master_flat * np.sqrt(
            ((medianflat_error / medianflat)) ** 2 + ((RMS_spacial/spacial_response_model) ** 2)
        )   

    else:

        master_flat = spectral_normalized



    fig, ax = plt.subplots(5 if not flat_params["skip_spacial"] else 3, 2, figsize=(15, 12))

    ax[0][0].imshow(medianflat.T, cmap="gray", origin="lower")
    ax[0][0].set_title("Master flat prior to normalization")
    ax[0][0].axis("off")

    ax[1][0].imshow(spectral_response_model.T, cmap="gray", origin="lower")
    ax[1][0].set_title("2D spectral response model")
    ax[1][0].axis("off")

    ax[2][0].imshow(spectral_normalized.T, cmap="gray", origin="lower")
    ax[2][0].set_title("Master flat normalized by spectral response model")
    ax[2][0].axis("off")

    if not flat_params["skip_spacial"]:
        ax[3][0].imshow(spacial_response_model.T, cmap="gray", origin="lower")
        ax[3][0].set_title("2D slit illumination model")
        ax[3][0].axis("off")

        ax[4][0].imshow(master_flat.T, cmap="gray", origin="lower")
        ax[4][0].set_title(
            "Final master flat - normalized by slit illumination and spectral response models"
        )
        ax[4][0].axis("off")

    N_bins = int(np.sqrt(len(medianflat.flatten())))

    ax[0][1].hist(
        medianflat.flatten(),
        bins=N_bins,
        range=(0, np.max(medianflat)),
        color="black",
    )
    ax[1][1].hist(spectral_response_model.flatten(), bins=N_bins, color="black")
    ax[2][1].hist(spectral_normalized.flatten(), bins=N_bins, color="black")
    if not flat_params["skip_spacial"]:
        ax[3][1].hist(spacial_response_model.flatten(), bins=N_bins, color="black")
        ax[4][1].hist(master_flat.flatten(), bins=N_bins, color="black")

    for a in ax[:, 1]:
        a.set_ylabel("N pixels")
    for a in ax[-1, :]:
        a.set_xlabel("Counts (ADU)")
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
    fig.align_ylabels(ax[:, 1])

    plt.show()

    logger.info("Flat frames processed.")

    logger.info(
        "Mean pixel value of the final master flat-field: "
        f"{round(np.nanmean(master_flat),5)} (should be 1.0)."
    )

    # check if the median is 1 to within 5 decimal places
    if round(np.nanmean(master_flat), 5) != 1:
        logger.warning(
            "The mean pixel value of the final master flat-field is not 1.0."
        )
        logger.warning("This may indicate a problem with the normalisation.")
        logger.warning("Check the normalisation region in the flat-field frames.")

    if use_overscan:
        # if overscan was used, set the overscan region to one to avoid
        # explosive values in the final flat

        # Extract the overscan region
        overscan_x_start = detector_params["overscan"]["overscan_x_start"]
        overscan_x_end = detector_params["overscan"]["overscan_x_end"]
        overscan_y_start = detector_params["overscan"]["overscan_y_start"]
        overscan_y_end = detector_params["overscan"]["overscan_y_end"]

        medianflat[overscan_y_start:overscan_y_end, overscan_x_start:overscan_x_end] = (
            1.0
        )

    logger.info("Attaching header and writing to disc...")

    # Write out result to fitsfile
    hdr = rawflat[0].header

    master_flat_frame = PyLongslit_frame(master_flat, medianflat_error, hdr, "master_flat")

    master_flat_frame.show_frame(normalize=False, save=True)    
    master_flat_frame.show_frame(normalize=True)
    master_flat_frame.write_to_disc()

    

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit flatfield procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_flats()


if __name__ == "__main__":
    main()
