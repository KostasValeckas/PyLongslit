from pylongslit.mkspecbias import run_bias
from pylongslit import set_config_file_path
import matplotlib
matplotlib.use('Agg') 

def test_bias():
    """
    Test the bias function.
    """

    set_config_file_path("/home/kostas/Documents/PyLongslit_dev/SDSS_J213510+2728/SDSS_J213510+2728.json")

    run_bias()

