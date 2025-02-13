import astroscrappy
import argparse
import os as os


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
                                 
        _, clean_arr = astroscrappy.detect_cosmics(
            frame.data,
            sigclip=sigclip,
            sigfrac=frac,
            objlim=objlim,
            cleantype="medmask",
            niter=niter,
            sepmed=True,
            verbose=True,
        )

        frame.data = clean_arr

        logger.info(f"Cosmic rays removed on {file}.")

        frame.header["CRREMOVD"] = True

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
