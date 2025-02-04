import matplotlib.pyplot as plt
import argparse

def subtract_background(reduced_files):

    from pylongslit.logger import logger
    from pylongslit.parser import output_dir, background_params
    from pylongslit.utils import hist_normalize, open_fits, write_to_fits

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
    headers = {}
    for file in reduced_files:
        with open_fits(output_dir, file) as hdul:
            new_filename = file.replace("reduced_science_", "").replace(
                "reduced_std_", ""
            )
            images[new_filename] = hdul[0].data
            headers[new_filename] = hdul[0].header

    subtracted_images = {}
    for i in range(len(file_pairs)):
        pair = file_pairs[str(i + 1)]

        subtracted_image = images[pair["A"]] - images[pair["B"]]

        subtracted_images[pair["A"]] = subtracted_image

        # Histogram equalize the images
        hist_eq_before = hist_normalize(images[pair["A"]])
        hist_eq_after = hist_normalize(subtracted_image)

        # Create a plot with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # Show the histogram equalized image before subtraction
        axes[0].imshow(hist_eq_before, cmap="gray")
        axes[0].set_title(f'Before Subtraction: {pair["A"]}')
        axes[0].axis("off")

        # Show the histogram equalized image after subtraction
        axes[1].imshow(hist_eq_after, cmap="gray")
        axes[1].set_title(f'After Subtraction: {pair["A"]}')
        axes[1].axis("off")

        # Display the plot
        plt.tight_layout()
        plt.show()

    # save the subtracted images
    for filename in reduced_files:
        new_filename = filename.replace("reduced_science_", "").replace(
            "reduced_std_", ""
        )
        if new_filename in subtracted_images:
            image = subtracted_images[new_filename]
            header = headers[new_filename]
            logger.info(f"Saving subtracted image {filename} to disc.")
            write_to_fits(image, header, filename, output_dir)

    logger.info("Subtracted images saved to disc.")


def run_background_subtraction():

    from pylongslit.utils import get_reduced_frames

    reduced_files = get_reduced_frames()
    subtract_background(reduced_files)

def main():
    parser = argparse.ArgumentParser(description="Run the pylongslit background-subtraction procedure.")
    parser.add_argument('config', type=str, help='Configuration file path')
    # Add more arguments as needed

    args = parser.parse_args()

    from pylongslit import set_config_file_path
    set_config_file_path(args.config)

    run_background_subtraction()

if __name__ == "__main__":
    main()