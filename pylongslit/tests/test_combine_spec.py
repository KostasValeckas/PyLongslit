"""
PyLongslit test module for the fluxed spectrum combination procedure.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(32)
def test_flux_combine_GQ():
    """
    Test the fluxed spectrum combination procedure on the GQ1218+0832 dataset.
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
    from pylongslit.combine import run_combine_spec
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the combination procedure
    run_combine_spec()

    assert os.path.exists(os.path.join(output_dir, "GQ1218+0832_combined.dat"))


@pytest.mark.order(33)
def test_flux_combine_SDSS():
    """
    Test the fluxed spectrum combination procedure on the SDSSJ213510+2728 dataset.
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
    from pylongslit.combine import run_combine_spec
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the combination procedure
    run_combine_spec()

    assert os.path.exists(os.path.join(output_dir, "SDSS_J213510+2728_combined.dat"))
