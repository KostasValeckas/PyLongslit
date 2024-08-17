import astroscrappy
from logger import logger
from parser import detector_params, crr_params, skip_science_or_standard_bool
from parser import output_dir
from parser import science_params, standard_params
from utils import FileList, open_fits, write_to_fits
import matplotlib.pyplot as plt
from skimage import exposure

import os as os

from skimage import exposure

"""
Module for removing cosmic rays from raw science and standard star frames.
"""

#TODO is there a sensful way to make QA plots for crremoval?

def remove_cosmics(file_list: FileList, sigclip, sigfrac, objlim, niter):
    """
    A wrapper for astroscrappy.detect_cosmics.

    Writes the cosmic-ray removed image to disc in the output directory
    with the prefix 'crr_'.

    Parameters
    ----------
    file_list : FileList
        A list of files to remove cosmic rays from.

    sigclip : float
        Laplacian-to-noise limit for cosmic ray detection.

    sigfrac : float
        Fractional detection limit for neighboring pixels.

    objlim : float
        Minimum contrast between Laplacian image and the fine structure image.

    niter : int
        Number of iterations to perform.
    """

    for file in file_list:

        logger.info(f"Removing cosmic rays from {file}...")

        hdu = open_fits(file_list.path, file)

        _, clean_arr = astroscrappy.detect_cosmics(
            hdu[1].data,
            sigclip=sigclip,
            sigfrac=sigfrac,
            objlim=objlim,
            cleantype="medmask",
            niter=niter,
            sepmed=True,
            verbose=True,
        )

        # Replace data array with cleaned image
        hdu[1].data = clean_arr

        logger.info(f"Cosmic rays removed on {file}.")      

        logger.info(f"Writing output to disc...")

        write_to_fits(hdu[1].data, hdu[0].header, "crr_" + file, output_dir)

        logger.info(
            f"Cosmic-ray removed file written to disc at in {output_dir}, "
            f"filename crr_{file}."
        )

        hdu.close()


def run_crremoval():

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

    if skip_science_or_standard_bool == 0:
        logger.critical(
            'Both skip_science and skip_standard are set to "true" in the '
            "config file. There is nothing to perform the reduction on."
        )
        logger.error('Set at least one of them "false" and try again.')

        exit()

    elif skip_science_or_standard_bool == 1:
        star_file_list = None
        science_file_list = FileList(science_params["science_dir"])

    elif skip_science_or_standard_bool == 2:
        star_file_list = FileList(standard_params["standard_dir"])
        science_file_list = None

    else:
        star_file_list = FileList(standard_params["standard_dir"])
        science_file_list = FileList(science_params["science_dir"])

    if star_file_list is not None:
        logger.info(
            f"Removing cosmic rays from {star_file_list.num_files} "
            "standard star frames:"
        )

        print("------------------------------------")
        for file in star_file_list:
            print(file)
        print("------------------------------------")

        remove_cosmics(star_file_list, sigclip, frac, objlim, niter)

    if science_file_list is not None:
        logger.info(
            f"Removing cosmic rays from {science_file_list.num_files} "
            "science frames:"
        )

        print("------------------------------------")
        for file in science_file_list:
            print(file)
        print("------------------------------------")

        remove_cosmics(science_file_list, sigclip, frac, objlim, niter)

    logger.info("Cosmic-ray removal procedure finished.")


if __name__ == "__main__":
    run_crremoval()
