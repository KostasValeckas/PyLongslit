from parser import output_dir, detector_params
from utils import open_fits, list_files, get_skysub_files, show_frame
from logger import logger
import os
import numpy as np
from photutils.aperture import RectangularAperture
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, gaussian_fwhm_to_sigma
from scipy.interpolate import interp1d
from wavecalib import load_fit2d_REID_from_disc

#Weight function for optimal extraction
def gaussweight(x, mu, sig):
    return np.exp(-0.5*(x-mu)**2/sig**2) / (np.sqrt(2.*np.pi)*sig)

def load_wavelength_map():

    #TODO - not used. Keep while developing amd then removes
    
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


    

def extract_object_optimal(trace_data, skysubbed_frame, gain, read_out_noise):

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

    spec = np.array(spec) / 300
    spec_var = np.array(spec_var) / (30**2)

    plt.plot(pixel, spec, label = "Extracted spectrum")
    plt.plot(pixel, np.sqrt(spec_var), label = "Error")
    plt.legend()
    plt.show()

    return pixel, spec, spec_var


def extract_objects(skysubbed_files, trace_dir):

    logger.info("Extracting 1D spectra")

    # get gain and read out noise parameters
    gain = detector_params["gain"]
    read_out_noise = detector_params["read_out_noise"]

    for filename in skysubbed_files:
        logger.info(f"Extracting 1D spectrum from {filename}")

        filename_obj = filename.replace("skysub_", "obj_").replace(".fits", ".dat")

        trace_data = trace_dir[filename_obj]

        pixel, spec, spec_var = extract_object_optimal(trace_data, filename, gain, read_out_noise)

        wavelength_calibrate(pixel, trace_data[1], spec)


def wavelength_calibrate(pixels, centers, spec):

    fit2d_REID = load_fit2d_REID_from_disc()

    ap_wavelen = fit2d_REID(pixels, centers)

    wavelen_homogenous = np.linspace(ap_wavelen[0], ap_wavelen[-1], len(spec))
    
    f = interp1d(ap_wavelen, spec, fill_value="extrapolate", kind="cubic")
    spec_calibrated = f(wavelen_homogenous)

    plt.plot(wavelen_homogenous, spec_calibrated)
    plt.show()








def run_extract_1d():
    logger.info("Running extract_1d")

    trace_dir = load_object_traces()

    skysubbed_files = get_skysub_files()

    if len(skysubbed_files) != len(trace_dir):
        logger.error("Number of skysubbed files and object traces do not match.")
        logger.error("Re-run both procedures or remove left-over files.")
        exit()

    extract_objects(skysubbed_files, trace_dir)

    logger.info("extract_1d done")

if __name__ == "__main__":
    run_extract_1d()