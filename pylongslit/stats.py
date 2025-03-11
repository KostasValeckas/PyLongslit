"""
PyLongslit module for statistics.
"""

# TODO: some utils might be more appropriate to move here

import numpy as np
from tqdm import tqdm


def bootstrap_median_errors_framestack(framestack, nboot=1000):
    """
    Calculate the standard error of the median using bootstrapping.
    Assumes frames are stacked along the first axis. The median is calculated
    along the first axis.

    framestack : numpy array
        The framestack to calculate the median errors of.

    nboot : int, optional
        The number of bootstrap samples to take. Default is 1000.

    Returns
    -------
    median_errors : numpy array
        The standard error of the median of each pixel in the framestack.
    """

    from pylongslit.logger import logger

    # Check the input
    if not isinstance(framestack, np.ndarray):
        logger.critical("Input failure in error estimation. Contact the developers.")
        exit()

    # Initialize variables for incremental calculation
    shape = framestack.shape[1:]
    sum_medians = np.zeros(shape)
    sum_squares = np.zeros(shape)

    logger.info("Bootstrapping errors. This will take a while...")
    logger.info(
        f"You can turn off bootstrapping in the config file if you want to speed up the process."
    )
    for _ in tqdm(range(nboot), desc="Bootstrapping"):
        random_sample = np.random.choice(
            framestack.shape[0], framestack.shape[0], replace=True
        )
        sample_median = np.median(framestack[random_sample], axis=0)

        # Update running sum and sum of squares gradually to avoid memory issues
        sum_medians += sample_median
        sum_squares += sample_median**2

    # Calculate mean of medians
    mean_medians = sum_medians / nboot

    # Calculate variance of medians
    variance_medians = (sum_squares / nboot) - (mean_medians**2)

    # Calculate standard error of the median
    median_errors = np.sqrt(variance_medians) / np.sqrt(nboot)

    return median_errors


def safe_mean(array, axis=None):
    """
    Calculate the mean of an array, ignoring NaN and infs values.

    array : numpy array
        The array to calculate the mean of.

    axis : int, optional
        The axis along which to calculate the mean.
        If None, the mean of the whole array is calculated.

    Returns
    -------
    mean : float
        The mean of the array.
    """
    return np.nanmean(array[np.isfinite(array)], axis=axis)
