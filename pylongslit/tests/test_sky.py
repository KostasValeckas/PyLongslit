"""
PyLongslit test module for the moduled sky-subtraction procedure.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(18)
def test_sky_GQ():
    """
    Test the modelled sky-subraction on the GQ1218+0832 dataset.
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
    from pylongslit.skysubtract import remove_sky_background
    from pylongslit.utils import PyLongslit_frame

    matplotlib.use("Agg")  # Use non-interactive backend

    # these would normally be produced by interactive center choosing:
    center_dict = {
        "reduced_science_0003881272-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits": (
            1258,
            745,
        ),
        "reduced_science_0003881273-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits": (
            1391,
            734,
        ),
        "reduced_standard_0003881849-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits": (
            1316,
            745,
        ),
    }

    # Run the sky-subtraction procedure
    remove_sky_background(center_dict)

    # load the frames to assert their header keyword changes
    frame1 = PyLongslit_frame.read_from_disc(
        "reduced_science_0003881272-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits"
    )
    frame2 = PyLongslit_frame.read_from_disc(
        "reduced_science_0003881273-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits"
    )

    assert frame1.header["SKYSUBBED"]
    assert frame2.header["SKYSUBBED"]


@pytest.mark.order(19)
def test_sky_SDSS():
    """
    Test the modelled sky-subraction on the SDSSJ213510+2728 dataset.
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
    from pylongslit.skysubtract import remove_sky_background
    from pylongslit.utils import PyLongslit_frame

    matplotlib.use("Agg")  # Use non-interactive backend

    # this would normally be produced by interactive center choosing:
    center_dict = {"reduced_standard_ALHh050097.fits": (1057, 250)}

    # Run the sky-subtraction procedure
    remove_sky_background(center_dict)

    # load the frame to assert the header keyword changes
    frame = PyLongslit_frame.read_from_disc("reduced_standard_ALHh050097.fits")

    assert frame.header["SKYSUBBED"]
