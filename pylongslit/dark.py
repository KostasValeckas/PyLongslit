"""
PyLongslit module for dark current estimation.

For now this module only supports constant dark current estimation, 
as no instrument has been tested where dark current has a noticable effect. 
"""

import numpy as np
from astropy.io import fits


def estimate_dark(frame, exptime, supress_warning=False):
    """
    Simple function to estimate the dark current in a frame by
    assuming a constant dark current for every pixel and
    scaling it by the exposure time.

    Parameters
    ----------
    frame : numpy.ndarray
        Any frame that matches the needed detector shape.

    exptime : float
        The exposure time in seconds.
        The dark current will be scaled by this value.

    supress_warning : bool
        If True, the function will not print a warning if the dark current is set to 0.

    Returns
    -------
    dark_frame : pylongslit.utils.PyLongslit_frame
        A PyLongslit_frame object that contains the dark frame.
    """

    from pylongslit.utils import PyLongslit_frame
    from pylongslit.parser import detector_params
    from pylongslit.logger import logger

    # TODO: For now, this is a simple solution with constant DARK currents
    # that are assumed to be growing linearly with exptime.
    # Later on, add option to support DARK frames

    gain = detector_params["gain"]  # e/ADU

    dark = detector_params["dark_current"]  # e/s/pixel

    # in electrons:
    dark_frame_e = np.ones_like(frame) * dark * exptime
    # convert to ADU
    dark_frame_counts = dark_frame_e / gain

    # poisson noise for darks, from
    #     From:
    # Richard Berry, James Burnell - The Handbook of Astronomical Image Processing
    # -Willmann-Bell (2005), p. 45 - 46
    dark_noise_error = np.sqrt(dark_frame_counts)

    # Create an empty FITS header
    header = fits.Header()

    dark_frame = PyLongslit_frame(dark_frame_counts, dark_noise_error, header, "dark")

    if not dark == 0:
        dark_frame.write_to_disc()

    elif not supress_warning:
        logger.warning(
            "Dark current is set to 0 in the config file. No dark frame will be created."
        )
        logger.warning(
            "This is wanted behaviour for some instruments, but ensure that this is the case for your instrument."
        )

    return dark_frame
