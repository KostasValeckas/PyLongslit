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

def overlay_trace(pixel, center, fwhm, skysubbed_frame):

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
    
def run_obj_trace_clone():
   pixel, center, fwhm = read_obj_trace_results()

   skysubbed_frame = load_skysubbed_frame()

   overlay_trace(pixel, center, fwhm, skysubbed_frame)

   



if __name__ == "__main__":
    run_obj_trace_clone()