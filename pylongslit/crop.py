import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import argparse


def crop_image(image):

    from pylongslit.utils import hist_normalize

    norm_image = hist_normalize(image)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img_display = ax.imshow(norm_image, cmap="gray")

    axcolor = "lightgoldenrodyellow"
    axtop = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    axbottom = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

    stop = Slider(axtop, "Top", 0, image.shape[0], valinit=0, valstep=1)
    sbottom = Slider(
        axbottom, "Bottom", 0, image.shape[0], valinit=image.shape[0], valstep=1
    )

    def update(val):
        top = int(stop.val)
        bottom = int(sbottom.val)
        cropped_img = norm_image[top:bottom, :]
        img_display.set_data(cropped_img)
        fig.canvas.draw_idle()

    stop.on_changed(update)
    sbottom.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

    plt.show()

    top = int(stop.val)
    bottom = int(sbottom.val)
    image = image[top:bottom, :]

    cropped_y = top, bottom

    plt.imshow(hist_normalize(image), cmap="gray")
    plt.show()

    return image, cropped_y


def crop_files(files):

    from pylongslit.parser import output_dir
    from pylongslit.utils import open_fits, write_to_fits

    for i, file in enumerate(files):
        hdul = open_fits(output_dir, file)
        header = hdul[0].header
        data = hdul[0].data
        data, cropped_y = crop_image(data)
        header["CROPY1"] = cropped_y[0]
        header["CROPY2"] = cropped_y[1]
        write_to_fits(data, header, file, output_dir)


def run_crop():

    from pylongslit.utils import get_reduced_frames

    reduced_files = get_reduced_frames()

    crop_files(reduced_files)

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit crop procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_crop()


if __name__ == "__main__":
    main()
