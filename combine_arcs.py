"""
Module to combine arc frames into a single master arc frame.
"""

from logger import logger
from parser import output_dir
from utils import FileList, open_fits, write_to_fits, list_files, get_filenames
import os
import numpy as np

def combine_arcs():

    logger.info("Fetching reduced arc frimes...")

    arc_files = get_filenames(starts_with="reduced_arc")

    if  len(arc_files) == 0:
        logger.critical("No reduced arc files found.")
        logger.critical(
            "Make sure you have run the 'reduce' procedure first."
        )
        exit()

    logger.info(f"Found {len(arc_files)} reduced arc files:")
    list_files(arc_files)

    logger.info("Combining arc frames...")

    arc_data = []

    for arc_file in arc_files:
        hdu = open_fits(output_dir, arc_file)
        arc_data.append(hdu[0].data)

    # Calculate the mean of the arc frames
    master_arc = np.sum(arc_data, axis=0)

    logger.info("Master arc created successfully, writing to disc...")

    # Write the master arc to a FITS file
    write_to_fits(master_arc, hdu[0].header, "master_arc.fits", output_dir)

    logger.info(f"Master arc written to disc in directory {output_dir}, filename 'master_arc.fits'.")

if __name__ == "__main__":
    combine_arcs()
