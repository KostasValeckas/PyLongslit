"""
PyLongslit test module for the reduction procedure.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(13)
def test_reduce_GQ():
    """
    Test the reduction procedure on the GQ1218+0832 dataset.
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
    from pylongslit.reduce import reduce_all
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the reduction procedure
    reduce_all()

    assert os.path.exists(
        os.path.join(
            output_dir,
            "reduced_science_0003881272-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits",
        )
    )
    assert os.path.exists(
        os.path.join(
            output_dir,
            "reduced_science_0003881273-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits",
        )
    )
    assert os.path.exists(
        os.path.join(
            output_dir,
            "reduced_standard_0003881849-20230707-OSIRIS-OsirisLongSlitSpectroscopy.fits",
        )
    )


@pytest.mark.order(14)
def test_reduce_SDSS():
    """
    Test the reduction procedure on the SDSSJ213510+2728 dataset.
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
    from pylongslit.reduce import reduce_all
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the reduction procedure
    reduce_all()

    assert os.path.exists(os.path.join(output_dir, "reduced_science_ALHh080251.fits"))
    assert os.path.exists(os.path.join(output_dir, "reduced_science_ALHh080252.fits"))
    assert os.path.exists(os.path.join(output_dir, "reduced_standard_ALHh050097.fits"))
