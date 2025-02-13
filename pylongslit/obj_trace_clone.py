import numpy as np
import matplotlib.pyplot as plt
import argparse



def read_obj_trace_results():
    from pylongslit.parser import obj_trace_clone_params
    
    file_path = obj_trace_clone_params["archived_spec_root"]

    pixel, center, fwhm = np.loadtxt(file_path, unpack=True)

    # plt.plot(pixel, center)
    # plt.show()
    # plt.plot(pixel, fwhm)
    # plt.show()

    return pixel, center, fwhm


def load_frame():

    from pylongslit.utils import PyLongslit_frame
    from pylongslit.parser import obj_trace_clone_params

    file_path = obj_trace_clone_params["frame_root"]

    frame = PyLongslit_frame.read_from_disc(file_path)

    data = frame.data

    return data


def overlay_trace(pixel, center, fwhm_guess, skysubbed_frame):

    from pylongslit.utils import hist_normalize

    normalized_skysub = hist_normalize(skysubbed_frame)

    fig, ax = plt.subplots()
    ax.imshow(normalized_skysub, origin="lower")
    (line_below,) = ax.plot(
        pixel, center - fwhm_guess, label="Object Trace", color="red"
    )
    (line_above,) = ax.plot(pixel, center + fwhm_guess, color="red")
    plt.title("Use arrow keys to move the aperture trace")
    plt.legend()

    # Event handler function
    def on_key(event):
        nonlocal center
        nonlocal fwhm_guess
        if event.key == "up":
            center += 1  # Move the aptrace up by 1 pixel
        elif event.key == "down":
            center -= 1  # Move the aptrace down by 1 pixel
        elif event.key == "right":
            fwhm_guess += 1
        elif event.key == "left":
            fwhm_guess -= 1
        line_below.set_ydata(center - fwhm_guess)  # Update the plot
        line_above.set_ydata(center + fwhm_guess)
        fig.canvas.draw()  # Redraw the figure

    # Connect the event handler to the figure
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    return center, fwhm_guess


def write_cloned_trace_to_file(pixel, center, fwhm):

    from pylongslit.logger import logger
    from pylongslit.parser import obj_trace_clone_params

    filename = obj_trace_clone_params["frame_root"]

    output_file = filename.replace("reduced_", "obj_").replace(".fits", ".dat")

    fwhm_array = np.full_like(center, fwhm)

    # write to the file
    with open(output_file, "w") as f:
        for x, center, fwhm in zip(pixel, center, fwhm_array):
            f.write(f"{x}\t{center}\t{fwhm}\n")

    logger.info("Cloned trace written to disk.")
    logger.info(f"Output file: {output_file}")


def run_obj_trace_clone():
    pixel, center, fwhm = read_obj_trace_results()

    skysubbed_frame = load_frame()

    # for initial fwhm guess, just use the mean of the fwhm archived trace
    corrected_centers, fwhm = overlay_trace(
        pixel, center, np.mean(fwhm), skysubbed_frame
    )

    write_cloned_trace_to_file(pixel, corrected_centers, fwhm)

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit cloned object trace procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_obj_trace_clone()


if __name__ == "__main__":
    main()
    
