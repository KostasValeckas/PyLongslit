"""
This module is called any time the package is initialized. Only functtion
it has is to set the path to the config file.
"""

CONFIG_FILE_PATH = "config.json"  # Default config file path


def set_config_file_path(path):
    global CONFIG_FILE_PATH
    # loading the config file
    CONFIG_FILE_PATH = path
