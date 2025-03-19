"""
PyLongslit test module for the configuration file checker.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(3)
def test_cofig_GQ():
    """
    Test the config checker on the GQ1218+0832 dataset.
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
    from pylongslit.check_config import run_config_checks

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the config_file_checker
    run_config_checks()


@pytest.mark.order(4)
def test_config_SDSS():
    """
    Test the config checker on the SDSSJ213510+2728 dataset.
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
    from pylongslit.check_config import run_config_checks

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the config_file_checker
    run_config_checks()
