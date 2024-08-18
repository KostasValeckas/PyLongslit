import numpy as np
import glob as glob
from astropy.io import fits
from parser import detector_params, output_dir, skip_science_or_standard_bool
from parser import science_params, standard_params, arc_params
from utils import FileList, open_fits, write_to_fits, list_files, hist_normalize
from logger import logger
import matplotlib.pyplot as plt
from overscan import subtract_overscan_from_frame
import os

"""
Module for reducing (bias subtraction, flat division) and combining 
exposures (science, standard star and arc lamps).
"""

def show_objects():
    """
    Plot the user-defined object regions on a raw science and 
    standard star frames together with user defined object centrums.
    """

    logger.info("Opening the first file in the science directory...")
    # read the names of the flat files from the directory
    file_list = FileList(science_params["science_dir"])

    # open the first file in the directory
    raw_science = open_fits(science_params["science_dir"], file_list.files[0])
    logger.info("File opened successfully.")

    data = np.array(raw_science[1].data)

    data_equalized = hist_normalize(data)
    
    # show the overscan region overlayed on a raw flat frame
    plt.imshow(data_equalized, cmap="gray")

    plt.show()

    """

    # plot the object regions
    for region in science_params["obj_regions"]:
        x1, x2 = region["x1"], region["x2"]
        y1, y2 = region["y1"], region["y2"]

        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="red", facecolor="none")

        plt.gca().add_patch(rect)

    # plot the object centrums
    for center in science_params["obj_centers"]:
        x, y = center["x"], center["y"]

        plt.scatter(x, y, color="red")

    """



def read_crr_files():
    """
    Read the cosmic-ray removed files from the output directory and
    perform some checks.

    Returns
    -------
    science_files : list
        A list of cosmic-ray removed science files.

    standard_files : list
        A list of cosmic-ray removed standard star files.

    arc_files : list
        A list of cosmic-ray removed arc files.
    """

    science_files = []
    standard_files = []
    arc_files = []

    for file in os.listdir(output_dir):
        if file.startswith("crr") or file.startswith("/crr"):
            if "science" in file:
                science_files.append(file)
            elif "std" in file:
                standard_files.append(file)
            elif "arc" in file:
                arc_files.append(file)

    logger.info(f"Found {len(science_files)} cosmic-ray removed science files.")
    logger.info(f"Found {len(standard_files)} cosmic-ray removed standard star files.")
    logger.info(f"Found {len(arc_files)} cosmic-ray removed arc files.")

    return science_files, standard_files, arc_files


def reduce_frame(frame, master_bias, master_flat, use_overscan):
    """
    Performs overscan subtraction, bias subtraction
    and flat fielding of a single frame.
    """

    if use_overscan:
        frame = subtract_overscan_from_frame(frame)

    logger.info("Subtracting the master bias frame...")

    frame = frame - master_bias

    logger.info("Dividing by the master flat frame...")

    frame = frame / master_flat



    pass


def reduce_all():
    # Extract the detector parameters
    xsize = detector_params["xsize"]
    ysize = detector_params["ysize"]

    use_overscan = detector_params["overscan"]["use_overscan"]

    # Fetch bias and flat master frames

    logger.info("Fetching the master bias frame...")

    try:
        BIAS_HDU = open_fits(output_dir, "master_bias.fits")
    except FileNotFoundError:
        logger.critical(f"Master bias frame not found in {output_dir}.")
        logger.error("Make sure you have excecuted the bias procdure first.")
        exit()

    BIAS = BIAS_HDU[0].data

    logger.info("Master bias frame found and loaded.")

    logger.info("Fetching the master flat frame...")

    try:
        FLAT_HDU = open_fits(output_dir, "master_flat.fits")
    except FileNotFoundError:
        logger.critical(f"Master flat frame not found in {output_dir}.")
        logger.error("Make sure you have excecuted the flat procdure first.")
        exit()

    FLAT = FLAT_HDU[0].data

    logger.info("Master flat frame found and loaded.")

    logger.info(f"Fetching cosmic-ray removed files from {output_dir} ...")

    science_files, standard_files, arc_files = read_crr_files()

    if skip_science_or_standard_bool == 1:
        logger.warning("Skipping standard star reduction as requested...")
    else:
        logger.info("Reducing following standard star frames:")

        list_files(standard_files)



        centers = standard_params["obj_centers"]

        if len(centers) != len(standard_files):
            logger.error(
                "The number of object centers must be equal to "
                "the number of standard star frames."
            )
            logger.error("Check the configuration file and try again.")

            exit()

        
        for file in standard_files:
            
            logger.info(f"Reducing frame {file} ...")

            hdu = open_fits(output_dir, file)

            data = hdu[0].data

            data = reduce_frame(data, BIAS, FLAT, use_overscan)

    exit()


    if len(centers) != n_rawimages:
        raise ValueError(
            "The number of centers must be equal to the number of raw images"
        )

    # Read the raw file, subtract overscan, bias and divide by the flat
    for n in range(0, n_rawimages):
        spec = fits.open(rawimages[n])
        print("Info on file:")
        print(spec.info())
        specdata = np.array(spec[1].data)
        mean = np.mean(specdata[2067 : ysize - 5, 10 : xsize - 1])
        specdata = specdata - mean
        print("Subtracted the median value of the overscan :", mean)
        specdata = (specdata - BIAS) / FLAT
        hdr = spec[0].header
        specdata1 = specdata[50:1750, centers[n] - 100 : centers[n] + 100]
        print(outnames[n])
        fits.writeto(outnames[n], specdata1, hdr, overwrite=True)

    # Add and rotate

    sum = np.zeros_like(fits.open(outnames[0])[0].data)

    for n in range(0, n_rawimages):
        sub = fits.open(outnames[n])
        sum = sub[0].data

    rot = np.rot90(sum, k=3)
    hduout = fits.PrimaryHDU(rot)
    hduout.header.extend(
        hdr, strip=True, update=True, update_first=False, useblanks=True, bottom=False
    )
    hduout.header["DISPAXIS"] = 1
    hduout.header["NEXP"] = len(rawimages)
    hduout.header["CRVAL1"] = 1
    hduout.header["CRVAL2"] = 1
    hduout.header["CRPIX1"] = 1
    hduout.header["CRPIX2"] = 1
    hduout.header["CRVAL1"] = 1
    hduout.header["CRVAL1"] = 1
    hduout.header["CDELT1"] = 1
    hduout.header["CDELT2"] = 1
    hduout.writeto(
        "../obj.fits" if not standard_star_reduction else "../std.fits", overwrite=True
    )

    # Arcframe
    arclist = open("raw_arcs.list")
    nframes = len(arclist.readlines())
    arclist.seek(0)

    specdata = np.zeros((ysize, xsize), float)
    for i in range(0, nframes):
        spec = fits.open(str.rstrip(arclist.readline()))
        data = spec[1].data
        if (len(data[0, :]) != xsize) or (len(data[:, 0]) != ysize):
            sys.exit(spec.name + " has wrong image size")
        specdata += data
    mean = np.mean(specdata[2067 : ysize - 5, 10 : xsize - 1])
    specdata = specdata - mean
    print("Subtracted the median value of the overscan :", mean)
    specdata = (specdata - BIAS) / FLAT
    hdr = spec[0].header
    center = int((centers[0]) / 1.0)
    specdata1 = specdata[50:1750, center - 100 : center + 100]
    rot = np.rot90(specdata1, k=3)
    hduout = fits.PrimaryHDU(rot)
    hduout.header.extend(
        hdr, strip=True, update=True, update_first=False, useblanks=True, bottom=False
    )
    hduout.header["DISPAXIS"] = 1
    hduout.header["CRVAL1"] = 1
    hduout.header["CRVAL2"] = 1
    hduout.header["CRPIX1"] = 1
    hduout.header["CRPIX2"] = 1
    hduout.header["CRVAL1"] = 1
    hduout.header["CRVAL1"] = 1
    hduout.header["CDELT1"] = 1
    hduout.header["CDELT2"] = 1
    hduout.writeto(
        "../arcsub.fits" if not standard_star_reduction else "../arcsub_std.fits",
        overwrite=True,
    )

    if standard_star_reduction:
        print(
            "\n\n\033[91m\n\nATTENTION:\033[0m THIS REDUCTION HAS BEEN RUN AS A STANDARD STAR REDUCTION"
        )
        print(
            'If this is an object reduction, please use "reducescience.py" script, '
            + "and re-run both the standard star and science object reductions. \n\n"
        )

    else:
        print(
            "\n\n\033[91m\n\nATTENTION:\033[0m THIS REDUCTION HAS BEEN RUN AS A SCIENCE OBJECT REDUCTION"
        )
        print(
            'If this is a standard star reduction, please use "reducestd.py" script,'
            + "and re-run both the standard star and science object reductions. \n\n"
        )


if __name__ == "__main__":
    reduce_all()
