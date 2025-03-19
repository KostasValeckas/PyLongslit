"""
PyLongslit test module for the cosmic-ray removal procedure.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(15)
def test_crr_GQ():
    """
    Test the crr procedure on the GQ1218+0832 dataset.
    """

    # this resets the memory - needs to be called explicitly by every test
    # therefore no method created for this repeated code

    for key in list(sys.modules.keys()):
        if key.startswith("pylongslit"):
            del sys.modules[key]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    CONFIG_FILE = "GQ1218+0832.json"
    from pylongslit import set_config_file_path

    set_config_file_path(CONFIG_FILE)
    from pylongslit.crremoval import run_crremoval
    from pylongslit.utils import PyLongslit_frame

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the crr procedure
    run_crremoval()

    # load the frames to assert their crr header keyword changes
    frame1 = PyLongslit_frame.read_from_disc(
        "reduced_science_0003881272-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits"
    )
    frame2 = PyLongslit_frame.read_from_disc(
        "reduced_science_0003881273-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits"
    )
    frame3 = PyLongslit_frame.read_from_disc(
        "reduced_standard_0003881849-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits"
    )

    assert frame1.header["CRRREMOVD"]
    assert frame2.header["CRRREMOVD"]
    assert frame3.header["CRRREMOVD"]


@pytest.mark.order(16)
def test_crr_SDSS():
    """
    Test the crr procedure on the SDSSJ213510+2728 dataset.
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
    from pylongslit.crremoval import run_crremoval
    from pylongslit.utils import PyLongslit_frame

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the crr procedure
    run_crremoval()

    # load the frames to assert their crr header keyword changes
    frame1 = PyLongslit_frame.read_from_disc("reduced_science_ALHh080251.fits")
    frame2 = PyLongslit_frame.read_from_disc("reduced_science_ALHh080252.fits")
    frame3 = PyLongslit_frame.read_from_disc("reduced_standard_ALHh050097.fits")

    assert frame1.header["CRRREMOVD"]
    assert frame2.header["CRRREMOVD"]
    assert frame3.header["CRRREMOVD"]
