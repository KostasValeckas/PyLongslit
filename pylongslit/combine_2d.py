import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
PyLongslit module for combinning 2d reduced spectra.
"""

def run_combine_2dspec():
    """
    Main function to run the combination procedure.
    """
    from pylongslit.logger import logger
    from pylongslit.utils import PyLongslit_frame, get_reduced_frames

    logger.warning("UNDER DEVELOPMENT - NOT READY FOR USE")
    logger.warning("UNDER DEVELOPMENT - NOT READY FOR USE")
    logger.warning("UNDER DEVELOPMENT - NOT READY FOR USE")

    logger.info("Fetching the reduced frames.")
    reduced_files = get_reduced_frames()
    if len(reduced_files) == 0:
        logger.error(
            "No reduced frames found. Please run the reduction procedure first."
        )
        exit()

    data_container = {}

    for i, file in enumerate(reduced_files):

        logger.info(f"Loading frame {file}...")

        frame = PyLongslit_frame.read_from_disc(file)
        data_container[file] = (frame.data, frame.sigma)

    # create a weighted average of the frames
    logger.info("Combining frames using weighted average...")
    
    # Get the shape from the first frame
    first_frame_data = list(data_container.values())[0][0]
    frame_shape = first_frame_data.shape
    num_frames = len(data_container)
    
    # Create 3D arrays to hold all data: (height, width, num_frames)
    all_data = np.zeros((frame_shape[0], frame_shape[1], num_frames))
    all_variance = np.zeros((frame_shape[0], frame_shape[1], num_frames))
    
    # Stack all 2D frames into 3D arrays
    for i, (data, sigma) in enumerate(data_container.values()):
        all_data[:, :, i] = data
        all_variance[:, :, i] = sigma**2  # Convert sigma to variance
    
    # Calculate weighted mean and variance for each pixel
    # Weighted mean: sum(data/var) / sum(1/var)
    weights = 1.0 / all_variance
    combined_data = np.sum(all_data * weights, axis=2) / np.sum(weights, axis=2)
    combined_variance = 1.0 / np.sum(weights, axis=2)
    combined_sigma = np.sqrt(combined_variance)
    
    # Create combined frame
    combined_frame = PyLongslit_frame(
        data=combined_data,
        sigma=combined_sigma,
        header=frame.header,  # Use header from last frame as template
        name="combined_2d_spectrum"
    )
    
    logger.info("Frame combination completed.")
    

    combined_frame.write_to_disc()

    logger.info("2D spectrum combination routine done.")    


def main():
    parser = argparse.ArgumentParser(
        description="Run the pylongslit combine 2d reduced spectrum procedure."
    )
    parser.add_argument("config", type=str, help="Configuration file path")

    args = parser.parse_args()

    from pylongslit import set_config_file_path

    set_config_file_path(args.config)

    run_combine_2dspec()


if __name__ == "__main__":
    main()