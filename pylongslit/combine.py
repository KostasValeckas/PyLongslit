import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
PyLongslit module for combinning fluxed spectra.
"""


def check_combine_params(fluxed_spectra):
    """
    Checks if the files set to be combined in the configuration file
    exist and can be loaded. Prepares a dictionary with object
    names as keys and the fluxed data as values.

    Parameters
    ----------
    fluxed_spectra : dict
        Dictionary containing the fluxed spectra.
        Format: {raw data filename: (wavelength, flux, var)}

    Returns
    -------
    fluxed_data_dict : dict
        Dictionary containing the fluxed data for the objects to be combined.
        Format: {object: [(wavelength, flux, var)]}
    """

    from pylongslit.logger import logger
    from pylongslit.parser import combine_params, developer_params

    logger.info("Reading combination parameters...")

    if len(combine_params) == 0:
        logger.warning("No objects set to be combined in the configuration file.")
        logger.warning("Aborting the combine procedure.")
        exit()

    logger.info(f"Found {len(combine_params)} objects to be combined.")

    # Strip '1d_fluxed_science_*.dat' for all strings in the list for searching,
    # as the user inputs the raw file names as input in the configuration file.
    fluxed_spectra_names = list(fluxed_spectra.keys())
    fluxed_spectra_names = [
        key.replace("1d_fluxed_science_", "") for key in fluxed_spectra_names
    ]
    fluxed_spectra_names = [
        key.replace(".dat", ".fits") for key in fluxed_spectra_names
    ]

    fluxed_data_dict = {}

    for obj_name, file_list in combine_params.items():
        logger.info(f"Checking if files exist for {obj_name}...")
        fluxed_data_dict[obj_name] = []
        for file in file_list:
            if file not in fluxed_spectra_names:
                logger.error(f"{file} not found in fluxed spectra.")
                logger.error(
                    "Run the flux calibration procedure first, or check the configuration file."
                )
                logger.error("Aborting the combine procedure.")
                exit()
            else:
                # copy data from the input dictionary to the output dictionary so
                # the data is grouped under the same object name
                fluxed_data_dict[obj_name].append(
                    fluxed_spectra[f"1d_fluxed_science_{file.replace('.fits', '.dat')}"]
                )
        logger.info(f"All files found for {obj_name}.")

    if developer_params["verbose_print"]:
        print("fluxed_data_dict:", fluxed_data_dict)

    return fluxed_data_dict


def combine_spectra(fluxed_data_dict, good_wavelength_start, good_wavelength_end):
    """
    Loops over the fluxed data dictionary and combines the spectra that are
    placed under the same object name in the dictionary. The combined spectra
    are saved to disk.

    Parameters
    ----------
    fluxed_data_dict : dict
        Dictionary containing the fluxed data for the objects to be combined.
        Format: {object: [(wavelength_1, flux_1, var_1), (wavelength_2, flux_2, var_2), ...]}

    good_wavelength_start : float
        Lower boundary of where the sensitivity function is well behaved.
        Used for plotting.

    good_wavelength_end : float
        Upper boundary of where the sensitivity function is well behaved.
        Used for plotting.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    for obj_name, data_list in fluxed_data_dict.items():

        # num of data tuples is the same as the number of spectra
        num_spectra = len(data_list)

        logger.info(f"Combining {num_spectra} spectra for {obj_name}...")

        # matrixes for all data before combinations
        # this is more memory efficient since numpy arrays are used
        # for direct mean and variance calculations with numpy functions
        all_lambda = np.zeros((len(data_list[0][0]), len(data_list)))
        all_flux = np.zeros((len(data_list[0][0]), len(data_list)))
        all_var = np.zeros((len(data_list[0][0]), len(data_list)))

        # stack all data into the matrixes
        for i, (wave, flux, var) in enumerate(data_list):
            all_lambda[:, i] = wave
            all_flux[:, i] = flux
            all_var[:, i] = var

        # take the weighted mean and variance of the fluxes
        combined_flux = np.sum(all_flux / all_var, axis=1) / np.sum(1 / all_var, axis=1)
        combined_var = 1 / np.sum(1 / all_var, axis=1)

        # the final data container for the object
        combined_spectrum = np.vstack((all_lambda[:, 0], combined_flux, combined_var)).T

        # crop the spec to a wavelength range where the sensitivity function is well behaved
        # this is done for prettier plotting

        mask = (combined_spectrum[:, 0] > good_wavelength_start) & (
            combined_spectrum[:, 0] < good_wavelength_end
        )

        plt.figure(figsize=(10, 6))

        for i, (lambda_, flux, var) in enumerate(data_list):
            plt.plot(lambda_[mask], flux[mask], label=f"Spectrum {i+1}")
            plt.plot(lambda_[mask], np.sqrt(var)[mask], label=f"Sigma {i+1}")

        plt.plot(
            combined_spectrum[:, 0][mask],
            combined_spectrum[:, 1][mask],
            color="black",
            label="Combined spectrum",
        )
        plt.plot(
            combined_spectrum[:, 0][mask],
            np.sqrt(combined_spectrum[:, 2])[mask],
            "--",
            color="black",
            label="Sigma Combined",
        )
        plt.legend()
        plt.title(
            f"{obj_name}\n"
            f"Only showing data between {good_wavelength_start} Å and {good_wavelength_end} Å, where the sensitivity function is well behaved."
        )
        plt.ylim(bottom=0)
        plt.savefig(f"{output_dir}/{obj_name}_all.png")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(
            combined_spectrum[:, 0][mask],
            combined_spectrum[:, 1][mask],
            color="black",
            label="Combined spectrum",
        )
        plt.plot(
            combined_spectrum[:, 0][mask],
            np.sqrt(combined_spectrum[:, 2])[mask],
            "--",
            color="black",
            label="Sigma Combined",
        )
        plt.legend()
        plt.title(
            f"{obj_name}\n"
            f"Only showing data between {good_wavelength_start} Å and {good_wavelength_end} Å, where the sensitivity function is well behaved."
        )
        plt.ylim(bottom=0)
        plt.savefig(f"{output_dir}/{obj_name}_combined.png")
        plt.show()

        logger.info(f"Saving combined spectrum for {obj_name}...")

        with open(f"{output_dir}/{obj_name}_combined.dat", "w") as f:
            f.write("wavelength calibrated_flux flux_var\n")
            for i in range(len(combined_spectrum)):
                f.write(
                    f"{combined_spectrum[i, 0]} {combined_spectrum[i, 1]} {combined_spectrum[i, 2]}\n"
                )

        logger.info(
            f"Combined spectrum for {obj_name} saved to {output_dir}/{obj_name}_combined.dat"
        )
        print("\n------------------------\n")


def run_combine_spec():
    """
    Main function to run the combination procedure.
    """
    from pylongslit.logger import logger
    from pylongslit.utils import load_fluxed_spec
    from pylongslit.sensitivity_function import load_sensfunc_from_disc

    logger.info("Running combination rutine...")

    fluxed_spectra = load_fluxed_spec()

    fluxed_data_dict = check_combine_params(fluxed_spectra)

    if fluxed_data_dict is None:
        logger.warning("No objects to combine. Exiting...")
        exit()

    # load the boundaries at which the sensitivity functiion is well behaved,
    # this is used for prettier plotting

    _, _, good_wavelength_start, good_wavelength_end = load_sensfunc_from_disc()

    combine_spectra(fluxed_data_dict, good_wavelength_start, good_wavelength_end)


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit combine fluxed spectrum procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_combine_spec()


if __name__ == "__main__":
    main()
