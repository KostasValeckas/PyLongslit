"""
PyLongslit test module for the sensitivity function module.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(28)
def test_sensfunc_GQ():
    """
    Test the sensitivity function procedure on the GQ1218+0832 dataset.
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
    from pylongslit.sensitivity_function import run_sensitivity_function
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the sensitivity function procedure
    run_sensitivity_function()

    assert os.path.exists(os.path.join(output_dir, "sensfunc.dat"))


@pytest.mark.order(29)
def test_extract_SDSS():
    """
    Test the sensitivity function procedure on the SDSSJ213510+2728 dataset.
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
    from pylongslit.sensitivity_function import run_sensitivity_function
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the sensitivity function procedure
    run_sensitivity_function()

    assert os.path.exists(os.path.join(output_dir, "sensfunc.dat"))
