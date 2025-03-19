"""
PyLongslit test module for the flats module.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(11)
def test_flats_GQ():
    """
    Test the falt-fielding on the GQ1218+0832 dataset.
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
    from pylongslit.mkspecflat import run_flats
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the flat procedure
    run_flats()

    assert os.path.exists(os.path.join(output_dir, "master_flat.fits"))


@pytest.mark.order(12)
def test_flats_SDSS():
    """
    Test the falt-fielding on the SDSSJ213510+2728 dataset.
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
    from pylongslit.mkspecflat import run_flats
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the flat procedure
    run_flats()

    assert os.path.exists(os.path.join(output_dir, "master_flat.fits"))
