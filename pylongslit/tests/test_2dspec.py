"""
PyLongslit test module for the 2d-spec viewer.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(20)
def test_2dspec_GQ():
    """
    Test the 2dspec viewer on the GQ1218+0832 dataset.
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
    from pylongslit.spec_viewer import run_2dspec

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the 2dspec viewer
    run_2dspec()


@pytest.mark.order(21)
def test_2dsped_SDSS():
    """
    Test the 2dspec viewer on the SDSSJ213510+2728 dataset.
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
    from pylongslit.spec_viewer import run_2dspec

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the 2dspec viewer
    run_2dspec()
