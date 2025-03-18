import os
import matplotlib
import sys
import pytest

@pytest.mark.order(3)
def test_bias_GQ():
    """
    Test the bias function.
    """

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for key in list(sys.modules.keys()):
        if key.startswith("pylongslit"):
            del sys.modules[key]


    CONFIG_FILE = "GQ1218+0832.json"
    from pylongslit import set_config_file_path 
    set_config_file_path(CONFIG_FILE)
    from pylongslit.mkspecbias import run_bias
    from pylongslit.parser import output_dir

    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Run the bias function
    run_bias()

        
    assert os.path.exists(os.path.join(output_dir, "master_bias.fits"))

@pytest.mark.order(4)
def test_bias_SDSS():
    """
    Test the bias function.
    """

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for key in list(sys.modules.keys()):
        if key.startswith("pylongslit"):
            del sys.modules[key]


    CONFIG_FILE_1 = "SDSS_J213510+2728.json"
    from pylongslit import set_config_file_path 
    set_config_file_path(CONFIG_FILE_1)
    from pylongslit.mkspecbias import run_bias
    from pylongslit.parser import output_dir

    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Run the bias function
    run_bias()

        
    assert os.path.exists(os.path.join(output_dir, "master_bias.fits"))




