from parser import output_dir
from utils import open_fits, list_files
from logger import logger
import os
import numpy as np

def load_wavelength_map():
    
    logger.info("Loading wavelength map")
    try:
        hdul = open_fits(output_dir, "wavelength_map.fits")
    except FileNotFoundError:
        logger.error("Wavelength map not found.")
        logger.error("Run the wavelength calibration procedure first.")
        exit()

    data = hdul[0].data
    logger.info("Wavelength map loaded")

    return data

def load_object_traces():

    logger.info("Loading object traces")

    # Get all filenames from output_dir starting with "obj_"
    filenames = [filename for filename in os.listdir(output_dir) if filename.startswith("obj_")]

    if len(filenames) == 0:
        logger.error("No object traces found.")
        logger.error("Run the object tracing procedure first.")
        exit()

    else:
        logger.info(f"Found {len(filenames)} object traces:")
        list_files(filenames)

    # this is the container that will be returned
    trace_dict = {}

    # change to output_dir
    os.chdir(output_dir)

    for filename in filenames:
        with open(filename, 'r') as file:
            trace_data = np.loadtxt(file)
            pixel = trace_data[:,0]
            obj_center = trace_data[:,1]
            obj_fwhm = trace_data[:,2]

            trace_dict[filename] = (pixel, obj_center, obj_fwhm)

        file.close()

    # reading done, change back to original directory
    os.chdir("..")
    
    # Process the filenames as needed
    
    logger.info("All object traces loaded.")
    
    return trace_dict



def run_extract_1d():
    logger.info("Running extract_1d")

    # load wavelength map
    wavelength_map = load_wavelength_map()

    trace_dir = load_object_traces()

    logger.info("extract_1d done")

if __name__ == "__main__":
    run_extract_1d()