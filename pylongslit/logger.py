"""
PyLongslit module for logging messages to a file and the console.
The script initializes a logger object and configures it to log messages to a file and the console.
The file name is derived from the configuration file name and is saved in the same directory.

This script is designed only to be read by other scripts in the PyLongslit package, 
executing it by itself does not make much sense.
"""

import logging
from colorama import Fore, Style, init
from pylongslit import CONFIG_FILE_PATH
import pathlib

# Initialize colorama
init(autoreset=True)

# Create a custom logger
logger = logging.getLogger("PyLongslit")

# Close and remove leftover handlers
for handler in logger.handlers:
    handler.close()
    logger.removeHandler(handler)

# Configure logging level
logger.setLevel(logging.INFO)

config_file_dir = pathlib.Path(CONFIG_FILE_PATH).parent
config_file_name = pathlib.Path(CONFIG_FILE_PATH).stem + ".log"

log_file_path = config_file_dir / config_file_name

# we add different information when we print to console and when we write to
# file, so we need two handlers
fh = logging.FileHandler(log_file_path)
ch = logging.StreamHandler()


# Create a custom formatter - this allows different colors for different log levels
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            record.levelname = f"{Fore.GREEN}{record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{Fore.YELLOW}{record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR:
            record.levelname = f"{Fore.RED}{record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.CRITICAL:
            record.levelname = (
                f"{Fore.RED}{record.levelname}{Style.BRIGHT}{Style.RESET_ALL}"
            )
        return super().format(record)


# For filelogging we add the dates, for console logging we don't
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
color_formatter = CustomFormatter(
    "%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

fh.setFormatter(formatter)
ch.setFormatter(color_formatter)

# Add both handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# Log messages
logger.info("Logger initialized. Log will be saved in " + fh.baseFilename + ".")
