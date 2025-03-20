import matplotlib.pyplot as plt
import argparse
import numpy as np


def run_background_subtraction():
    """
    Subtract the background from the reduced files using A - B image pairs.
    The pairs are given in the config file.

    This method loads the files, checks if they have been reduced,
    and subtracts the background. The frames are altered in place.
    """

    from pylongslit.logger import logger
    from pylongslit.parser import background_params
    from pylongslit.utils import PyLongslit_frame, hist_normalize
    from pylongslit.utils import get_reduced_frames

    reduced_files = get_reduced_frames()

    # these are the pairs of object - background files, given in the config file
    # they are given as a dictionary of dictionaries, where the key is the pair number
    # and the value is a dictionary with keys "A" and "B" which are the object and background files
    file_pairs = background_params["pairs"]

    logger.info(f"Found {len(file_pairs)} object - background pairs.")

    # check that the files named in pairs have been reduced
    for i in range(len(file_pairs)):
        pair = file_pairs[str(i + 1)]

        for file in pair.values():
            if not any(file in s for s in reduced_files):
                logger.error(f"{file} not found in reduced files.")
                logger.error("Please reduce the file first.")
                exit()

    logger.info("All files found in reduced files.")

    # load the images
    images = {}
    sigmas = {}
    headers = {}

    # count how many files have already been background subtracted - 
    # a bit of a hack to prevent double subtraction
    already_subtracted = 0

    for file in reduced_files:
        with PyLongslit_frame.read_from_disc(file) as frame:
            if frame.header["BCGSUBBED"] == True:
                already_subtracted += 1
                logger.warning(
                    f"File {file} already had background subtracted. Skipping..."
                )
                continue
            # strp prefix to match original file names
            new_filename = file.replace("reduced_science_", "").replace(
                "reduced_std_", ""
            )
            images[new_filename] = frame.data
            sigmas[new_filename] = frame.sigma
            headers[new_filename] = frame.header

    if already_subtracted == len(file_pairs):
        logger.warning("All files have already been background subtracted.")
        logger.warning("Exiting...")
        exit()

    logger.info("Images loaded.")

    # these are the containers for the final results
    subtracted_images = {}
    new_sigmas = {}

    for i in range(len(file_pairs)):
        pair = file_pairs[str(i + 1)]

        logger.info(f"Subtracting background from {pair['A']} using {pair['B']}.")

        # check the dimensions of the images
        if images[pair["A"]].shape != images[pair["B"]].shape:
            logger.error(
                f"Image dimensions do not match for {pair['A']} and {pair['B']}."
            )
            logger.error("Please check the images.")
            logger.error(
                "Don't run the crop procedure before the background subtraction."
            )
            logger.error(
                "If you have already run the crop procedure, please re-reduce the images."
            )
            exit()

        # simple handling for skip-cases TODO: make this more robust
        try:
            subtracted_image = images[pair["A"]] - images[pair["B"]]
        except KeyError:
            continue

        subtracted_images[pair["A"]] = subtracted_image
        # propagating the errors
        new_sigmas[pair["A"]] = np.sqrt(sigmas[pair["A"]] ** 2 + sigmas[pair["B"]] ** 2)

        # for visualization purposes
        normalized = False

        def update(event):
            if normalized:
                axes[0].imshow(hist_normalize(images[pair["A"]]), cmap="gray")
                axes[1].imshow(hist_normalize(subtracted_image), cmap="gray")
            else:
                axes[0].imshow(images[pair["A"]], cmap="gray")
                axes[1].imshow(subtracted_image, cmap="gray")
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal normalized
            if event.key == "h":
                normalized = not normalized
                update(event)

        # Create a plot with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Show the image before and after subtraction
        axes[0].imshow(images[pair["A"]], cmap="gray")
        axes[0].set_title(f'Before Subtraction: {pair["A"]}')
        axes[0].axis("off")

        axes[1].imshow(subtracted_image, cmap="gray")
        axes[1].set_title(f'After Subtraction: {pair["A"]}')
        axes[1].axis("off")

        fig.suptitle(
            f"Background Subtraction for {pair['A']}. Press \"h\" to histogram normalize. \n"
            f"Ensure that the traces of the objects do not overlay each other.\n"
            f"If they do, reduce the images again and skip this procedure."
        )

        fig.canvas.mpl_connect("key_press_event", on_key)

        plt.show()

        # show 5 slices in spectral direction to check if the traces are separated
        fig, axes = plt.subplots(5, 1, figsize=(10, 6))
        # plot indices - 5 slices. Do not start at detector edges 
        indices = np.linspace(10, subtracted_image.shape[1] - 10, 7, dtype=int)[1:-1]
        for plot_index, spectral_index in enumerate(indices):
            axes[plot_index].plot(subtracted_image[:, spectral_index], label = f"Slice at pixel {spectral_index}")
            axes[plot_index].legend()
            axes[plot_index].set_xlabel("Spatial Pixel")
            axes[plot_index].set_ylabel("Counts")
        fig.suptitle(
            f"A sample of 5 spatial detector slices for {pair['A']}.\n"
            "Make sure the traces are separated completely - no overlap.\n"
            "If not, reduce the images again and skip this procedure."
        )
        plt.show()


    # save the subtracted images
    for filename in reduced_files:

        # crop the prefix to find the original file name
        new_filename = filename.replace("reduced_science_", "").replace(
            "reduced_std_", ""
        )

        # use the original file name to create the subtracted frame.
        if new_filename in subtracted_images:
            image = subtracted_images[new_filename]
            sigma = new_sigmas[new_filename]
            header = headers[new_filename]

            save_filename = filename.replace(".fits", "")

            frame = PyLongslit_frame(image, sigma, header, save_filename)
            # this header prevents double subtraction
            frame.header["BCGSUBBED"] = True

            frame.show_frame()

            logger.info(f"Saving subtracted image {save_filename} to disc.")
            frame.write_to_disc()

    logger.info("Subtracted images saved to disc.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit background-subtraction procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_background_subtraction()


if __name__ == "__main__":
    main()
