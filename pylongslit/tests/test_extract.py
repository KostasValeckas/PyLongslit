"""
PyLongslit test module for the main extraction procedure.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(26)
def test_extract_GQ():
    """
    Test the extraction procedure on the GQ1218+0832 dataset.
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
    from pylongslit.extract_1d import run_extract_1d
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the extraction procedure
    run_extract_1d()

    assert os.path.exists(
        os.path.join(
            output_dir,
            "1d_science_0003881272-20230707-OSIRIS-OsirisLongSlitSpectroscopy.dat",
        )
    )
    assert os.path.exists(
        os.path.join(
            output_dir,
            "1d_science_0003881273-20230707-OSIRIS-OsirisLongSlitSpectroscopy.dat",
        )
    )
    assert os.path.exists(
        os.path.join(
            output_dir,
            "1d_standard_0003881849-20230707-OSIRIS-OsirisLongSlitSpectroscopy.dat",
        )
    )


@pytest.mark.order(27)
def test_extract_SDSS():
    """
    Test the simple extraction procedure on the SDSSJ213510+2728 dataset.
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
    from pylongslit.extract_1d import run_extract_1d
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the extraction procedure
    run_extract_1d()

    assert os.path.exists(os.path.join(output_dir, "1d_science_ALHh080251.dat"))
    assert os.path.exists(os.path.join(output_dir, "1d_science_ALHh080252.dat"))
    assert os.path.exists(os.path.join(output_dir, "1d_standard_ALHh050097.dat"))
