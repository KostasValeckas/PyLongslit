from parser import output_dir, detector_params
from utils import open_fits, list_files, get_skysub_files, show_frame
from logger import logger
import os
import numpy as np
from photutils.aperture import RectangularAperture
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, gaussian_fwhm_to_sigma

#Weight function for optimal extraction
def gaussweight(x, mu, sig):
    return np.exp(-0.5*(x-mu)**2/sig**2) / (np.sqrt(2.*np.pi)*sig)

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

    
    # sort as this is needed when cross referencing with skysubbed files
    filenames.sort()

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

def estimate_variance(data, gain, read_out_noise):

    """
    Taken from Horne, K. (1986). 
    An optimal extraction algorithm for CCD spectroscopy.
    Publications of the Astronomical Society of the Pacific, 
    98(609), 609-617, eq. 12.
    """

    return (read_out_noise/ gain) **2 + np.abs(data) 


    

def extract_object_optimal(wavelength_map, trace_data, skysubbed_frame, gain, read_out_noise):

    """
    Extraction algorithm taken from Horne, K. (1986). 
    An optimal extraction algorithm for CCD spectroscopy.
    Publications of the Astronomical Society of the Pacific, 
    98(609), 609-617.
    """

    pixel, center, FWHM = trace_data

    # Open the skysubbed frame
    hdul = open_fits(output_dir, skysubbed_frame)
    
    skysubbed_data = hdul[0].data
    x_row_array = np.arange(skysubbed_data.shape[0])

    variance = estimate_variance(skysubbed_data, gain, read_out_noise)

    # these are the containers that will be filled for every value
    spec = []
    spec_var = []

    print(len(center))

    for i in range(len(center)):

        obj_center = center[i]
        obj_fwhm = FWHM[i] * gaussian_fwhm_to_sigma
        weight = gaussweight(x_row_array, obj_center, obj_fwhm)

        skysubbed_data_slice = skysubbed_data[:, int(pixel[0])+i]
        

        spec.append(np.sum(
            skysubbed_data_slice * weight / variance[:, i]
        ) / np.sum(weight ** 2 / variance[:, i]))
        spec_var.append(np.sum(weight) / np.sum(weight ** 2 / variance[:, i]))

    spec = np.array(spec)
    spec_var = np.array(spec_var)

    spec = spec
    spec_var = spec_var

    plt.plot(pixel, spec)
    plt.plot(pixel, 1/(spec_var))
    plt.show()





    

def extract_objects(skysubbed_files, trace_dir , wavelength_map):

    logger.info("Extracting 1D spectra")

    # get gain and read out noise parameters
    gain = detector_params["gain"]
    read_out_noise = detector_params["read_out_noise"]

    for filename in skysubbed_files:
        logger.info(f"Extracting 1D spectrum from {filename}")

        filename_obj = filename.replace("skysub_", "obj_").replace(".fits", ".dat")

        trace_data = trace_dir[filename_obj]

        extract_object_optimal(wavelength_map, trace_data, filename, gain, read_out_noise)



def run_extract_1d():
    logger.info("Running extract_1d")

    # load wavelength map
    wavelength_map = load_wavelength_map()

    trace_dir = load_object_traces()

    skysubbed_files = get_skysub_files()

    if len(skysubbed_files) != len(trace_dir):
        logger.error("Number of skysubbed files and object traces do not match.")
        logger.error("Re-run both procedures or remove left-over files.")
        exit()

    extract_objects(skysubbed_files, trace_dir, wavelength_map)

    logger.info("extract_1d done")

if __name__ == "__main__":
    run_extract_1d()