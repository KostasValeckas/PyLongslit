"""
Module for logging messages to a file and the console.
"""

import logging
from colorama import Fore, Style, init

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

# Create a file handler
# TODO: Change the file path to object name
fh = logging.FileHandler("pylogslit.log")

# Create a console handler
ch = logging.StreamHandler()


# Create a custom formatter
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


# Create a formatter and set it for both handlers
formatter = logging.Formatter(
    "%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
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
