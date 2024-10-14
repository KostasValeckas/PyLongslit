from parser import obj_trace_clone_params
import numpy as np
import matplotlib.pyplot as plt
from utils import open_fits, show_frame, hist_normalize

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

def overlay_trace(pixel, center, skysubbed_frame):

    normalized_skysub = hist_normalize(skysubbed_frame)
    

    fig, ax = plt.subplots()
    ax.imshow(normalized_skysub, origin='lower')
    line, = ax.plot(pixel, center, label="Object Trace", color="red")
    plt.title("Use arrow keys to move the aperture trace")
    plt.legend()

    # Event handler function
    def on_key(event):
        nonlocal center
        if event.key == 'up':
            center += 1  # Move the aptrace up by 1 pixel
        elif event.key == 'down':
            center -= 1  # Move the aptrace down by 1 pixel
        line.set_ydata(center)  # Update the plot
        fig.canvas.draw()  # Redraw the figure

    # Connect the event handler to the figure
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    return center

def initiate_fwhm_array(center):

    FWHM = obj_trace_clone_params["FWHM"]

    fwhm_array = np.full_like(center, FWHM)

    return fwhm_array


def write_cloned_trace_to_file(pixel, center, fwhm):
    
    filename = obj_trace_clone_params["skysubbed_frame_root"]

    output_file = filename.replace("skysub_", "obj_").replace(".fits", ".dat")

    # write to the file
    with open(output_file, "w") as f:
        for x, center, fwhm in zip(pixel, center, fwhm):
            f.write(f"{x}\t{center}\t{fwhm}\n")


def run_obj_trace_clone():
   pixel, center, _ = read_obj_trace_results()

   skysubbed_frame = load_skysubbed_frame()

   corrected_centers = overlay_trace(pixel, center, skysubbed_frame)

   fwhm_array = initiate_fwhm_array(corrected_centers)

   write_cloned_trace_to_file(pixel, corrected_centers, fwhm_array)

if __name__ == "__main__":
    run_obj_trace_clone()