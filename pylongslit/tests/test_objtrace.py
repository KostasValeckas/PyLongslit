"""
PyLongslit test module for the object tracing procedure.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(22)
def test_obj_trace_GQ():
    """
    Test the object tracing procedure on the GQ1218+0832 dataset.
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
    from pylongslit.obj_trace import find_obj, write_obj_trace_results
    from pylongslit.parser import output_dir

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

    # Run the interactive part of the object tracing procedure
    obj_dict = find_obj(center_dict)
    write_obj_trace_results(obj_dict)

    # assert objects have been written

    assert os.path.exists(
        os.path.join(
            output_dir,
            "obj_science_0003881272-20230707-OSIRIS-OsirisLongSlitSpectroscopy.dat",
        )
    )
    assert os.path.exists(
        os.path.join(
            output_dir,
            "obj_science_0003881273-20230707-OSIRIS-OsirisLongSlitSpectroscopy.dat",
        )
    )
    assert os.path.exists(
        os.path.join(
            output_dir,
            "obj_standard_0003881849-20230707-OSIRIS-OsirisLongSlitSpectroscopy.dat",
        )
    )


@pytest.mark.order(23)
def test_objtrace_SDSS():
    """
    Test the object tracing procedure on the SDSSJ213510+2728 dataset.
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
    from pylongslit.obj_trace import find_obj, write_obj_trace_results
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # these would normally be produced by interactive center choosing:
    center_dict = {
        "reduced_science_ALHh080251.fits": (1013, 249),
        "reduced_science_ALHh080252.fits": (1057, 226),
        "reduced_standard_ALHh050097.fits": (1114, 250),
    }

    # Run the interactive part of the object tracing procedure
    obj_dict = find_obj(center_dict)
    write_obj_trace_results(obj_dict)

    # assert objects have been written

    assert os.path.exists(os.path.join(output_dir, "obj_science_ALHh080251.dat"))
    assert os.path.exists(os.path.join(output_dir, "obj_science_ALHh080252.dat"))
    assert os.path.exists(os.path.join(output_dir, "obj_standard_ALHh050097.dat"))
