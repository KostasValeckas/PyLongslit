import numpy as np
import matplotlib.pyplot as plt
import argparse

def check_combine_params(fluxed_spectra):
    from pylongslit.logger import logger
    from pylongslit.parser import combine_params
    logger.info("Reading combination parameters...")

    if len(combine_params) == 0:
        logger.warning("No objects set to be combined in the config file.")
        logger.warning("Skipping the combine procedure.")
        return None

    logger.info(f"Found {len(combine_params)} objects to be combined.")

    # Strip '1d_fluxed_science_*.dat' for all strings in the list for searching
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
                logger.error("Run the flux calibration procedure first.")
                exit()
            else:
                fluxed_data_dict[obj_name].append(
                    fluxed_spectra[f"1d_fluxed_science_{file.replace('.fits', '.dat')}"]
                )
        logger.info(f"All files found for {obj_name}.")

    return fluxed_data_dict


def combine_spectra(fluxed_data_dict):

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir

    for obj_name, data_list in fluxed_data_dict.items():

        num_spectra = len(data_list)

        logger.info(f"Combining {num_spectra} spectra for {obj_name}...")

        all_lambda = np.zeros((len(data_list[0][0]), len(data_list)))
        all_flux = np.zeros((len(data_list[0][0]), len(data_list)))
        all_var = np.zeros((len(data_list[0][0]), len(data_list)))

        for i, (lambda_, flux, var) in enumerate(data_list):
            all_lambda[:, i] = lambda_
            all_flux[:, i] = flux
            all_var[:, i] = var

        combined_flux = np.sum(all_flux / all_var, axis=1) / np.sum(1 / all_var, axis=1)
        combined_var = 1 / np.sum(1 / all_var, axis=1)

        combined_spectrum = np.vstack((all_lambda[:, 0], combined_flux, combined_var)).T

        plt.figure(figsize=(10, 6))

        for i, (lambda_, flux, var) in enumerate(data_list):
            plt.plot(lambda_, flux, label=f"Spectrum {i+1}")
            plt.plot(lambda_, np.sqrt(var), label=f"Sigma {i+1}")
           

        plt.plot(
            combined_spectrum[:, 0], combined_spectrum[:, 1], color="black", label="Combined spectrum"
        )
        plt.plot(
            combined_spectrum[:, 0],
            np.sqrt(combined_spectrum[:, 2]),
            "--",
            color="black",
            label="Sigma Combined",
        )
        plt.legend()
        plt.title(f"{obj_name}")
        plt.ylim(bottom=0)
        plt.savefig(f"{output_dir}/{obj_name}_all.png")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(
            combined_spectrum[:, 0], combined_spectrum[:, 1], color="black", label="Combined spectrum"
        )
        plt.plot(
            combined_spectrum[:, 0],
            np.sqrt(combined_spectrum[:, 2]),
            "--",
            color="black",
            label="Sigma Combined",
        )
        plt.legend()
        plt.title(f"{obj_name}")
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



def run_combine_spec():
    from pylongslit.logger import logger
    from pylongslit.utils import load_fluxed_spec
    
    logger.info("Running combination rutine...")

    fluxed_spectra = load_fluxed_spec()

    fluxed_data_dict = check_combine_params(fluxed_spectra)

    if fluxed_data_dict is None:
        logger.warning("No objects to combine. Exiting...")
        return

    combine_spectra(fluxed_data_dict)

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit combine-spectrum procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_combine_spec()


if __name__ == "__main__":
    main()
