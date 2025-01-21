from parser import obj_trace_clone_params
import numpy as np
import matplotlib.pyplot as plt
from utils import open_fits, show_frame, hist_normalize
from logger import logger

def read_obj_trace_results():
    file_path = obj_trace_clone_params["archived_spec_root"]
    
    pixel, center, fwhm = np.loadtxt(file_path, unpack=True)

    #plt.plot(pixel, center)
    #plt.show()
    #plt.plot(pixel, fwhm)
    #plt.show()

    return pixel, center, fwhm

def load_skysubbed_frame():
    
    file_path = obj_trace_clone_params["skysubbed_frame_root"]

    hdul = open_fits("", file_path)

    data = hdul[0].data

    return data

def overlay_trace(pixel, center, fwhm_guess, skysubbed_frame):

    normalized_skysub = hist_normalize(skysubbed_frame)
    

    fig, ax = plt.subplots()
    ax.imshow(normalized_skysub, origin='lower')
    line_below, = ax.plot(pixel, center-fwhm_guess, label="Object Trace", color="red")
    line_above, = ax.plot(pixel, center+fwhm_guess, color="red")
    plt.title("Use arrow keys to move the aperture trace")
    plt.legend()

    # Event handler function
    def on_key(event):
        nonlocal center
        nonlocal fwhm_guess
        if event.key == 'up':
            center += 1  # Move the aptrace up by 1 pixel
        elif event.key == 'down':
            center -= 1  # Move the aptrace down by 1 pixel
        elif event.key == 'right':
            fwhm_guess += 1
        elif event.key == 'left':
            fwhm_guess -= 1
        line_below.set_ydata(center-fwhm_guess)  # Update the plot
        line_above.set_ydata(center+fwhm_guess)
        fig.canvas.draw()  # Redraw the figure

    # Connect the event handler to the figure
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    return center, fwhm_guess


def write_cloned_trace_to_file(pixel, center, fwhm):
    
    filename = obj_trace_clone_params["skysubbed_frame_root"]

    output_file = filename.replace("skysub_", "obj_").replace(".fits", ".dat")

    fwhm_array = np.full_like(center, fwhm)

    # write to the file
    with open(output_file, "w") as f:
        for x, center, fwhm in zip(pixel, center, fwhm_array):
            f.write(f"{x}\t{center}\t{fwhm}\n")

    logger.info("Cloned trace written to disk.")
    logger.info(f"Output file: {output_file}") 

def run_obj_trace_clone():
   pixel, center, fwhm = read_obj_trace_results()

   skysubbed_frame = load_skysubbed_frame()

    # for initial fwhm guess, just use the mean of the fwhm archived trace
   corrected_centers, fwhm = overlay_trace(pixel, center, np.mean(fwhm), skysubbed_frame)

   write_cloned_trace_to_file(pixel, corrected_centers, fwhm)

if __name__ == "__main__":
    run_obj_trace_clone()