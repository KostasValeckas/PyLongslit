import os
import sys
import requests
import zipfile
import pytest

@pytest.mark.order(1)
def test_data_GQ(
    url="https://github.com/KostasValeckas/PyLongslit_dev/archive/refs/heads/main.zip",
    output_dir="."):
    """
    Download and extract a ZIP file from GitHub.

    Not really a test, but good to know if it fails

    Parameters
    ----------
    url : str
        The URL of the ZIP file to download.
    output_dir : str
        The directory where the ZIP file should be extracted.

    Returns
    -------
    str
        The path to the extracted directory.
    """

    for key in list(sys.modules.keys()):
        if key.startswith("pylongslit"):
            del sys.modules[key]

    CONFIG_FILE = "GQ1218+0832.json"
    from pylongslit import set_config_file_path 
    set_config_file_path(CONFIG_FILE)
    from pylongslit.check_config import check_dirs

    # check if the directories exist for the provided test json file, if not 
    # download the test suite: 

    # bool for bookeeping errors
    any_errors = False
    
    any_errors = check_dirs(any_errors)
        
    if any_errors:

        # if there are directory errors, this might be due to that this 
        # is the first time the test is run, so download the test suite


        # Download the ZIP file
        zip_path = os.path.join(output_dir, "temp.zip")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Remove the ZIP file
        os.remove(zip_path)

    # now check if this has fixed the directory errors

    any_errors = False


    any_errors = check_dirs(any_errors)

    assert not any_errors

    if any_errors:
        raise ValueError(
            "Error in test suite data. All other tests will fail. "
            "Check the configuration files placed in the test directory. "
            "See the docs for more information, or contact the developers. "
        )

@pytest.mark.order(2)
def test_data_SDSS(
    url="https://github.com/KostasValeckas/PyLongslit_dev/archive/refs/heads/main.zip",
    output_dir="."):
    """
    Download and extract a ZIP file from GitHub.

    Not really a test, but good to know if it fails

    Parameters
    ----------
    url : str
        The URL of the ZIP file to download.
    output_dir : str
        The directory where the ZIP file should be extracted.

    Returns
    -------
    str
        The path to the extracted directory.
    """

    for key in list(sys.modules.keys()):
        if key.startswith("pylongslit"):
            del sys.modules[key]

    CONFIG_FILE = "SDSS_J213510+2728.json"
    from pylongslit import set_config_file_path 
    set_config_file_path(CONFIG_FILE)
    from pylongslit.check_config import check_dirs

    # check if the directories exist for the provided test json file, if not 
    # download the test suite: 

    # bool for bookeeping errors
    any_errors = False
    
    any_errors = check_dirs(any_errors)
        
    if any_errors:

        # if there are directory errors, this might be due to that this 
        # is the first time the test is run, so download the test suite


        # Download the ZIP file
        zip_path = os.path.join(output_dir, "temp.zip")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Remove the ZIP file
        os.remove(zip_path)

    # now check if this has fixed the directory errors

    any_errors = False


    any_errors = check_dirs(any_errors)

    assert not any_errors

    if any_errors:
        raise ValueError(
            "Error in test suite data. All other tests will fail. "
            "Check the configuration files placed in the test directory. "
            "See the docs for more information, or contact the developers. "
        )
