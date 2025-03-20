import astroscrappy
import argparse
import os as os
import numpy as np
import matplotlib.pyplot as plt


"""
PyLongslit module for removing cosmic rays from raw science and standard star frames.
"""


def run_crremoval(figsize=(10, 6)):
    """
    This whole method is a wrapper for the astroscrappy.detect_cosmics method.
    It reads the reduced files from the disc,
    removes cosmic rays from them and writes the cleaned files back to the disc.

    The reduced files are being modified in place.

    Parameters
    ----------
    figsize : tuple, optional
        Size of the figure to display the QA plot. The default is (10, 6).
    """

    from pylongslit.logger import logger
    from pylongslit.parser import (
        detector_params,
        crr_params,
        skip_science_or_standard_bool,
    )
    from pylongslit.parser import science_params, standard_params
    from pylongslit.utils import get_file_group, PyLongslit_frame
    from pylongslit.stats import safe_mean

    # initiate user-set parameters from the configuration file
    gain = detector_params["gain"]
    read_out_noise = detector_params["read_out_noise"]

    logger.info("Cosmic-ray removal procedure running...")
    logger.info("Using the following detector parameters:")
    logger.info(f"gain = {gain}")
    logger.info(f"read_out_noise = {read_out_noise}")

    # check skip science or standard bool when fetching files
    # should not be a problem if it passed the reduction step, but just in case
    if skip_science_or_standard_bool == 0:
        logger.error(
            "Both science and standard star frames are set to be skipped in the configuration file."
        )
        logger.error("Only calibration frames can be processed in this configuration.")
        logger.error("Please set at least one of them to False.")
        exit()

    if skip_science_or_standard_bool == 1:
        logger.warning(
            "Standard star frames are set to be skipped in the configuration file."
        )
        logger.warning("Only science frames will be processed.")
    else:
        logger.info("Getting standard star files...")
        standard_star_file_list = get_file_group("reduced_standard")

        if len(standard_star_file_list) == 0:
            logger.error(
                "No reduced standard star files found. Please run the reduction procedure first."
            )
            exit()

    if skip_science_or_standard_bool == 2:
        logger.warning(
            "Science frames are set to be skipped in the configuration file."
        )
        logger.warning("Only standard star frames will be processed.")
    else:
        logger.info("Getting science files...")
        science_file_list = get_file_group("reduced_science")

        if len(science_file_list) == 0:
            logger.error(
                "No reduced science files found. Please run the reduction procedure first."
            )
            exit()

    # we do science and stananrd star frames separately, as they
    # usually need different parameters due to very different S/N ratios
    # - this calls for a different parameter set for cosmic-ray removal

    # but all files in a dict for later looping:
    if skip_science_or_standard_bool == 3:
        file_list = {"standard": standard_star_file_list, "science": science_file_list}
    elif skip_science_or_standard_bool == 1:
        file_list = {"science": science_file_list}
    else:
        file_list = {"standard": standard_star_file_list}

    # standard star frames:

    sigclip_std = crr_params["standard"]["sigclip"]
    frac_std = crr_params["standard"]["frac"]
    objlim_std = crr_params["standard"]["objlim"]
    niter_std = crr_params["standard"]["niter"]

    # science frames:

    sigclip_sci = crr_params["science"]["sigclip"]
    frac_sci = crr_params["science"]["frac"]
    objlim_sci = crr_params["science"]["objlim"]
    niter_sci = crr_params["science"]["niter"]

    for key, file_list in file_list.items():

        if key == "standard":
            sigclip = sigclip_std
            frac = frac_std
            objlim = objlim_std
            niter = niter_std
            exptime = standard_params["exptime"]
        else:
            sigclip = sigclip_sci
            frac = frac_sci
            objlim = objlim_sci
            niter = niter_sci
            exptime = science_params["exptime"]

        for file in file_list:

            print("-----------------------------------")
            logger.info(f"Removing cosmic rays from {file}...")

            frame = PyLongslit_frame.read_from_disc(file)

            # for stability, don't allow the same file to be processed twice
            if frame.header["CRRREMOVD"]:
                logger.warning(
                    f"File {file} already had cosmic rays removed. Skipping..."
                )
                logger.warning(
                    "If you want to re-run the cosmic-ray removal, reduce the frame again to reset."
                )
                continue

            # take backup data to show the difference
            data_backup = frame.data.copy()

            # mask is a boolean array with True values where cosmic rays are detected
            # clean_arr is the cleaned data array
            mask, clean_arr = astroscrappy.detect_cosmics(
                frame.data,
                sigclip=sigclip,
                sigfrac=frac,
                objlim=objlim,
                cleantype="medmask",
                invar=np.array(frame.sigma**2, dtype=np.float32),
                niter=niter,
                sepmed=True,
                verbose=True,
                gain=gain,
                readnoise=read_out_noise,
            )

            # swap out the data arrays, and replace the sigma array with the mean value
            frame.data = clean_arr
            frame.sigma[mask] = safe_mean(frame.sigma)

            # Show QA plot
            fig, ax = (
                plt.subplots(1, 2, figsize=figsize)
                if data_backup.shape[0] > data_backup.shape[1]
                else plt.subplots(2, 1, figsize=figsize)
            )
            ax[0].imshow(data_backup, cmap="gray", origin="lower")
            ax[0].scatter(np.where(mask)[1], np.where(mask)[0], color="red", s=1)
            ax[0].set_title(
                f"Detected cosmic rays (red)- {np.sum(mask)} pixels found.\n"
                f"Use the zoom function to inspect the cosmic rays."
            )
            ax[0].set_xlabel("Spectral Pixels")
            ax[0].set_ylabel("Spatial Pixels")

            ax[1].imshow(clean_arr, cmap="gray", origin="lower")
            ax[1].set_title("Cleaned data")
            ax[1].set_xlabel("Spectral Pixels")

            fig.suptitle(
                f"Cosmic-ray removal QA plot for {file} with exptime {exptime} seconds.\n"
                f"If the detection under/overestimating cosmic-rays, try changing the parameters in the configuration file."
            )
            plt.show()

            logger.info(f"Cosmic rays removed on {file}.")

            frame.header["CRRREMOVD"] = True

            logger.info(f"Writing output to disc...")

            frame.write_to_disc()

    logger.info("Cosmic-ray removal procedure finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit cosmic-ray removal procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_crremoval()


if __name__ == "__main__":
    main()
