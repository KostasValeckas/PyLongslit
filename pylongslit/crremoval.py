import astroscrappy
import argparse
import os as os
import numpy as np
import matplotlib.pyplot as plt


"""
Module for removing cosmic rays from raw science and standard star frames.
"""

# TODO is there a sensful way to make QA plots for crremoval?


def run_crremoval():

    from pylongslit.logger import logger
    from pylongslit.parser import detector_params, crr_params, skip_science_or_standard_bool
    from pylongslit.parser import science_params, standard_params
    from pylongslit.utils import FileList, list_files, get_file_group, PyLongslit_frame

    # initiate user parameters

    # detecctor
    gain = detector_params["gain"]
    read_out_noise = detector_params["read_out_noise"]

    # astroscrappy.detect_cosmics
    sigclip = crr_params["sigclip"]
    frac = crr_params["frac"]
    objlim = crr_params["objlim"]
    niter = crr_params["niter"]

    logger.info("Cosmic-ray removal procedure running...")
    logger.info("Using the following detector parameters:")
    logger.info(f"gain = {gain}")
    logger.info(f"read_out_noise = {read_out_noise}")

    file_list = get_file_group("reduced")

    for file in file_list:
        logger.info(f"Removing cosmic rays from {file}...")

        frame = PyLongslit_frame.read_from_disc(file)

        frame_data_temp = frame.data.copy()



        if frame.header["CRRREMOVD"]:
            logger.warning(f"File {file} already had cosmic rays removed. Skipping...")
            continue
                                 
        mask, clean_arr = astroscrappy.detect_cosmics(
            frame.data,
            sigclip=sigclip,
            sigfrac=frac,
            objlim=objlim,
            cleantype="medmask",
            invar = np.array(frame.sigma**2, dtype = np.float32),
            niter=niter,
            sepmed=True,
            verbose=True,
            gain=gain,
            readnoise=read_out_noise,
        )

        frame.data = clean_arr

        frame.sigma[mask] = np.nanmean(frame.sigma)

        _, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(frame.data, cmap="gray", origin="lower")
        ax.imshow(mask, cmap="Reds", alpha=0.5, origin="lower")
        ax.set_title(f"Cleaned data\n Red: cosmic rays - {np.sum(mask)} pixels found")
        plt.show()  

        logger.info(f"Cosmic rays removed on {file}.")

        frame.header["CRRREMOVD"] = True

        logger.info(f"Writing output to disc...")

        frame.write_to_disc()
    

    logger.info("Cosmic-ray removal procedure finished.")

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit cosmic-ray removal procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_crremoval() 


if __name__ == "__main__":
    main()
