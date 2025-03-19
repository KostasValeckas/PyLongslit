""" 
PyLongslit test module for the wavelength calibration model.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(9)
def test_wavecalib_GQ():
    """
    Test the wavecalibration on the GQ1218+0832 dataset.
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
    from pylongslit.wavecalib import run_wavecalib
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the wavecalibration
    run_wavecalib()

    assert os.path.exists(os.path.join(output_dir, "reidentified_lines.pkl"))
    assert os.path.exists(os.path.join(output_dir, "wavelen_fit.pkl"))
    assert os.path.exists(os.path.join(output_dir, "tilt_fit.pkl"))
    assert os.path.exists(os.path.join(output_dir, "good_lines.pkl"))
    assert os.path.exists(os.path.join(output_dir, "wavelength_map.fits"))


@pytest.mark.order(10)
def test_wavecalib_SDSS():
    """
    Test the wavecalibration on the SDSSJ213510+2728 dataset.
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
    from pylongslit.wavecalib import run_wavecalib
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the wavecalibration
    run_wavecalib()

    assert os.path.exists(os.path.join(output_dir, "reidentified_lines.pkl"))
    assert os.path.exists(os.path.join(output_dir, "wavelen_fit.pkl"))
    assert os.path.exists(os.path.join(output_dir, "tilt_fit.pkl"))
    assert os.path.exists(os.path.join(output_dir, "good_lines.pkl"))
    assert os.path.exists(os.path.join(output_dir, "wavelength_map.fits"))
