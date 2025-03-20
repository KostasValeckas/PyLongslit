"""
PyLongslit module for cropping the reduced images.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse


def crop_image(frame, figsize=(10, 6)):
    """
    Interactive cropping of the image.

    Parameters
    ----------
    frame : PyLongslit_frame
        The frame to be cropped.

    figsize : tuple
        The size of the figure.

    Returns
    -------
    image : numpy.ndarray
        The cropped image.

    sigma : numpy.ndarray
        The cropped error image.

    cropped_y : tuple
        The top and bottom y-coordinates of the crop.
    """

    from pylongslit.utils import hist_normalize

    # copy the data and error to avoid altering the original frame
    image = frame.data.copy()
    error = frame.sigma.copy()
    header = frame.header.copy()

    # if this is a re-crop, this will be needed for bookkeeping
    old_crop_bottom = header.get("CROPY1", 0)

    # this is used to toggle between normalized and non-normalized images
    normalized = False

    # create the interactive plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img_display = ax.imshow(image, cmap="gray")

    # some sliders for the top and bottom of the crop
    axcolor = "lightgoldenrodyellow"
    axbottom = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    axtop = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

    stop = Slider(axtop, "Top", 0, image.shape[0], valinit=image.shape[0], valstep=1)
    sbottom = Slider(axbottom, "Bottom", 0, image.shape[0], valinit=0, valstep=1)

    # interactive update function
    def update(val):
        top = int(stop.val)
        bottom = int(sbottom.val)
        cropped_img = image[bottom:top, :]
        if normalized:
            cropped_img = hist_normalize(cropped_img)
            img_display.set_clim(0, 1)
        else:
            img_display.set_clim(cropped_img.min(), cropped_img.max())
        img_display.set_data(cropped_img)
        fig.canvas.draw()

    stop.on_changed(update)
    sbottom.on_changed(update)

    # toggle between normalized and non-normalized images
    def toggle_hist_normalization(event):
        if event.key == "h":
            nonlocal normalized
            normalized = not normalized
            update(None)

    fig.canvas.mpl_connect("key_press_event", toggle_hist_normalization)

    fig.suptitle(
        "Use the sliders to crop the image. Press 'h' to histagram normalize.\n"
        "Cropping some background away can improve further processing,\nbut ensre you have some background left on both sides of the object.\n"
        "When you are satisfied with the crop, close the window ('q') to continue."
    )

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

    # extract the user set limits and crop the images
    top = int(stop.val)
    bottom = int(sbottom.val)
    image = image[bottom:top, :]
    sigma = error[bottom:top, :]

    # if this a re-crop, we need to keep track of the total crop since the original image
    # this is needed for matching the wavelength solution
    cropped_y = old_crop_bottom + bottom, old_crop_bottom + top

    return image, sigma, cropped_y


def run_crop():
    """
    Driver for the cropping procedure.

    Loads frames, checks input, crops the images, and saves the cropped images.
    """
    from pylongslit.utils import PyLongslit_frame, get_reduced_frames
    from pylongslit.logger import logger

    logger.info("Fetching the reduced frames.")
    reduced_files = get_reduced_frames()
    if len(reduced_files) == 0:
        logger.error(
            "No reduced frames found. Please run the reduction procedure first."
        )
        exit()

    for i, file in enumerate(reduced_files):

        logger.info(f"Cropping frame {file}...")

        frame = PyLongslit_frame.read_from_disc(file)

        data, error, cropped_y = crop_image(frame)
        frame.data = data
        frame.sigma = error
        frame.header["CROPY1"] = cropped_y[0]
        frame.header["CROPY2"] = cropped_y[1]

        frame.show_frame()

        frame.write_to_disc()

    logger.info("Cropping routine done.")


def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit crop procedure.")
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_crop()


if __name__ == "__main__":
    main()
