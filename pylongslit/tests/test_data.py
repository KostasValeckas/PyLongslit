"""
PyLongslit test module for checking the availability of the test data, 
and downloading it if necessary.
"""

import os
import sys
import requests
import zipfile
import pytest


@pytest.mark.order(1)
def test_data_GQ(
    url="https://github.com/KostasValeckas/PyLongslit_dev/archive/refs/heads/main.zip",
    output_dir=".",
):
    """
    Checks that the test data is available for the GQ1218+0832 dataset.

    If not, downloads the test suite from the GitHub repository.

    Parameters
    ----------
    url : str, optional
        The URL of the ZIP file to download.
        Default is the main branch of the PyLongslit_dev repository.

    output_dir : str, optional
        The directory where the ZIP file should be extracted.
        Default is the current directory.
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
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
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
    output_dir=".",
):
    """
    Checks that the test data is available for the SDSSJ213510+2728 dataset.

    If not, downloads the test suite from the GitHub repository.

    Parameters
    ----------
    url : str, optional
        The URL of the ZIP file to download.
        Default is the main branch of the PyLongslit_dev repository.

    output_dir : str, optional
        The directory where the ZIP file should be extracted.
        Default is the current directory.
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
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
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
