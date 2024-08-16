import astroscrappy
from logger import logger
import glob
from parser import detector_params, crr_params, skip_science_or_standard_bool

import os as os

from astropy.io import fits

"""
Module for removing cosmic rays from raw science and standard star frames.
"""


def run_crremoval():

    # initiate user parameters

    #detecctor
    gain = detector_params["gain"]
    read_out_noise = detector_params["read_out_noise"]

    # astroscrappy.detect_cosmics

    frac = crr_params["frac"]
    objlim = crr_params["objlim"]
    sigclip = crr_params["sigclip"]
    niter = crr_params["niter"]

    logger.info("Cosmic-ray removal procedure running...")
    logger.info("Using the following detector parameters:")
    logger.info(f"gain = {gain}")
    logger.info(f"read_out_noise = {read_out_noise}")


    if skip_science_or_standard_bool == 0:
        logger.critical(
            "Both skip_science and skip_standard are set to \"true\" in the "
            "config file. There is nothing to perform the reduction on."
        )
        logger.error("Set at least one of them \"false\" and try again.")

        exit()


    # Path to folder with science frames
    for nn in glob.glob("*crr*.fits"):
        os.remove(nn)
        print(nn)

    # Try to open raw_science.list, if it doesn't exist, open raw_std.list
    try:
        files = open('raw_science.list')
        print("\nScience observation list found: Using raw_science.list\n")
    except FileNotFoundError:
        try:
            files = open('raw_std.list')
            print("\nStandard star observation list found: Using raw_std.list\n")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Either raw_science.list or raw_std.list needs to be provided."
            )


    for n in files:
        n = n.strip() # Remove leading/trailing whitespaces
        fitsfile = fits.open(str(n))
        filename = os.path.basename(n)
        print('Removing cosmics from file: ' + filename + '...')
            
        crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[1].data, sigclip=sigclip, sigfrac=frac, objlim=objlim, cleantype='medmask', niter=niter, sepmed=True, verbose=True)

    # Replace data array with cleaned image
        fitsfile[1].data = clean_arr

    # Try to retain info of corrected pixel if extension is present.
        try:
            fitsfile[2].data[crmask] = 16 #Flag value for removed cosmic ray
        except:
            print("No bad-pixel extension present. No flag set for corrected pixels")

    # Update file
        fitsfile.writeto("crr"+filename, output_verify='fix')

    files.close()

if __name__ == "__main__":
    run_crremoval()
