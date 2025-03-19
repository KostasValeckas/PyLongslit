"""
PyLongslit test module for the A-B background subtraction procedure.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(17)
def test_AB_SDSS():
    """
    Test the A-B background subtraction procedure procedure on the SDSSJ213510+2728 dataset.
    """

    # this resets the memory - needs to be called explicitly by every test
    # therefore no method created for this repeated code

    for key in list(sys.modules.keys()):
        if key.startswith("pylongslit"):
            del sys.modules[key]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    CONFIG_FILE = "SDSS_J213510+2728.json"
    from pylongslit import set_config_file_path

    set_config_file_path(CONFIG_FILE)
    from pylongslit.subtract_background import run_background_subtraction
    from pylongslit.utils import PyLongslit_frame

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the A-B background subtraction procedure
    run_background_subtraction()

    # load the frames to assert their header keyword changes
    frame1 = PyLongslit_frame.read_from_disc("reduced_science_ALHh080251.fits")
    frame2 = PyLongslit_frame.read_from_disc("reduced_science_ALHh080252.fits")

    assert frame1.header["BCGSUBBED"]
    assert frame2.header["BCGSUBBED"]
