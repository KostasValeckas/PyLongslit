""" 
PyLongslit test module for the combine arcs model.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(7)
def test_arcs_GQ():
    """
    Test the combine arcs function on the GQ1218+0832 dataset.
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
    from pylongslit.combine_arcs import combine_arcs
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Combine arcs procedure
    combine_arcs()

    assert os.path.exists(os.path.join(output_dir, "master_arc.fits"))


@pytest.mark.order(8)
def test_arcs_SDSS():
    """
    Test the combine arcs function on the SDSSJ213510+2728 dataset.
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
    from pylongslit.combine_arcs import combine_arcs
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Combine arcs procedure
    combine_arcs()

    assert os.path.exists(os.path.join(output_dir, "master_arc.fits"))
