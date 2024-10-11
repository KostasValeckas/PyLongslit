from parser import obj_trace_clone_params
import numpy as np
import matplotlib.pyplot as plt

def read_obj_trace_results():
    file_path = obj_trace_clone_params["archived_spec_root"]
    
    pixel, center, fwhm = np.loadtxt(file_path, unpack=True)

    plt.plot(pixel, center)
    plt.show()
    plt.plot(pixel, fwhm)
    plt.show()

    return pixel, center, fwhm

def run_obj_trace_clone():
   read_obj_trace_results()

if __name__ == "__main__":
    run_obj_trace_clone()